# finetuner.py
import time
from typing import Dict, Any, Optional, Tuple, List
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .checkpoint import CheckpointManager
from .metrics import MetricsTracker
from .optimization import create_optimizer, create_scheduler
from .loss import contrastive_loss_fn

class FineTuningTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        user_tag_encoder: Dict[str, int],
        collection_label_encoder: Dict[str, int]
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.loss_fn = contrastive_loss_fn
        self.user_tag_encoder = user_tag_encoder
        self.collection_label_encoder = collection_label_encoder
        
        self.use_amp = config['training'].get('use_amp', False) and device.type == 'cuda'
        self.grad_clip_norm = config['training'].get('grad_clip_norm', 1.0)
        self.grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.metrics_tracker = MetricsTracker()

        print(f"FineTuningTrainer initialized - AMP: {self.use_amp}, Device: {device}, Grad Accum: {self.grad_accum_steps}")
        if self.train_dataloader.batch_sampler is not None:
            effective_batch_size = self.train_dataloader.batch_sampler.batch_size * self.grad_accum_steps
        else:
            effective_batch_size = self.train_dataloader.batch_size * self.grad_accum_steps
        print(f"Effective batch size: {effective_batch_size}")


    def _encode_labels(self, labels: List[List[str]], encoder: Dict[str, int], num_classes: int) -> torch.Tensor:
        batch_size = len(labels)
        encoded_tensor = torch.zeros(batch_size, num_classes, device=self.device)
        for i, sample_labels in enumerate(labels):
            if not sample_labels:
                continue
            for label in sample_labels:
                if label in encoder:
                    encoded_tensor[i, encoder[label]] = 1.0
        return encoded_tensor
    
    def _prepare_batch(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        vectors, attention_mask, metadata, difficulty_ratings, collection_labels, user_tags = batch
        
        vectors = vectors.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)
        metadata = metadata.to(self.device, non_blocking=True)
        
        labels_dict = {
            'difficulty_ratings': difficulty_ratings.to(self.device, non_blocking=True)
        }
        
        if any(collection_labels):
            # Check for model attribute before accessing it
            num_collection_classes = self.model.module.collection_label_head.out_features if isinstance(self.model, nn.DataParallel) else self.model.collection_label_head.out_features
            encoded_collections = self._encode_labels(collection_labels, self.collection_label_encoder, num_collection_classes)
            labels_dict['collection_labels'] = encoded_collections

        is_user_tag_head_present = hasattr(self.model, 'user_tag_head') or (isinstance(self.model, nn.DataParallel) and hasattr(self.model.module, 'user_tag_head'))
        if is_user_tag_head_present and any(user_tags):
            num_user_tag_classes = self.model.module.user_tag_head.out_features if isinstance(self.model, nn.DataParallel) else self.model.user_tag_head.out_features
            encoded_tags = self._encode_labels(user_tags, self.user_tag_encoder, num_user_tag_classes)
            labels_dict['user_tags'] = encoded_tags
            
        return vectors, attention_mask, metadata, labels_dict

    def _run_step(self, batch: Tuple, is_train: bool) -> Dict[str, float]:
        # This function is now only for validation and a single micro-step
        vectors, attention_mask, metadata, labels = self._prepare_batch(batch)
        
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                predictions = self.model(vectors, metadata, attention_mask)
                losses = self.loss_fn(predictions, labels, self.config)
        
        if is_train:
             # Scale the loss by accumulation steps
            scaled_loss = losses['total_loss'] / self.grad_accum_steps
            self.scaler.scale(scaled_loss).backward()
        
        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True) # Zero grad at the beginning of the epoch
        
        epoch_losses = {}
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {epoch+1} [Train]", 
            dynamic_ncols=True
        )
        
        for i, batch in enumerate(progress_bar):
            # This is now a micro-batch
            step_losses = self._run_step(batch, is_train=True)
            
            # Accumulate losses for logging
            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            
            # Perform optimizer step after accumulating gradients
            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(self.train_dataloader):
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.scheduler: self.scheduler.step()

            progress_bar.set_postfix({
                "Loss": f"{step_losses['total_loss']:.4f}",
                "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
        avg_losses = {k: v / len(self.train_dataloader) for k, v in epoch_losses.items()}
        avg_losses['learning_rate'] = self.optimizer.param_groups[0]['lr']
        return avg_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        epoch_losses = {}
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_dataloader, 
                desc=f"Validation", 
                dynamic_ncols=True,
                leave=False
            )
            for batch in progress_bar:
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    # We can use the original _run_step for validation as it doesn't backprop
                    vectors, attention_mask, metadata, labels = self._prepare_batch(batch)
                    predictions = self.model(vectors, metadata, attention_mask)
                    step_losses = self.loss_fn(predictions, labels, self.config)
                
                    for k, v in step_losses.items():
                            epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
                
        avg_losses = {k: v / len(self.val_dataloader) for k, v in epoch_losses.items()}
        return avg_losses

    def train(self, start_epoch: int = 0):
        # This part remains mostly the same
        num_epochs = self.config['training']['num_epochs']
        print(f"Starting fine-tuning from epoch {start_epoch+1}/{num_epochs}...")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch()
            
            self.metrics_tracker.log_epoch(epoch, train_metrics, val_metrics)
            
            epoch_duration = time.time() - epoch_start_time

            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.scaler,
                epoch, val_metrics, suffix=f"epoch_{epoch+1}"
            )
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.scaler,
                epoch, val_metrics, suffix="latest"
            )

            print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_duration:.2f}s | "
                  f"Train Loss: {train_metrics['total_loss']:.4f} | "

                  f"Val Loss: {val_metrics['total_loss']:.4f} | "
                  f"Checkpoint saved to {checkpoint_path}")
        
        print("Fine-tuning finished.")
        return self.metrics_tracker

def setup_finetuning(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    user_tag_encoder: Dict[str, int],
    collection_label_encoder: Dict[str, int]
) -> Tuple[FineTuningTrainer, CheckpointManager]:
    # This part remains the same
    total_steps = len(train_dataloader) * config['training']['num_epochs']
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    checkpoint_dir = config['training']['checkpoint_dir']
    model_name = config['model'].get('type', 'model') + "_finetuned"
    checkpoint_manager = CheckpointManager(checkpoint_dir, model_name)
    
    trainer = FineTuningTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        checkpoint_manager=checkpoint_manager,
        user_tag_encoder=user_tag_encoder,
        collection_label_encoder=collection_label_encoder
    )
    
    return trainer, checkpoint_manager