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
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.metrics_tracker = MetricsTracker()

        print(f"FineTuningTrainer initialized - AMP: {self.use_amp}, Device: {device}")

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
            num_collection_classes = self.model.collection_label_head.out_features
            encoded_collections = self._encode_labels(collection_labels, self.collection_label_encoder, num_collection_classes)
            labels_dict['collection_labels'] = encoded_collections

        if hasattr(self.model, 'user_tag_head') and any(user_tags):
            num_user_tag_classes = self.model.user_tag_head.out_features
            encoded_tags = self._encode_labels(user_tags, self.user_tag_encoder, num_user_tag_classes)
            labels_dict['user_tags'] = encoded_tags
            
        return vectors, attention_mask, metadata, labels_dict

    def _run_step(self, batch: Tuple, is_train: bool) -> Dict[str, float]:
        vectors, attention_mask, metadata, labels = self._prepare_batch(batch)
        
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                predictions = self.model(vectors, metadata, attention_mask)
                losses = self.loss_fn(predictions, labels, self.config)
        
        if is_train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses['total_loss']).backward()
            
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler: self.scheduler.step()
        
        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {}
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {epoch+1} [Train]", 
            dynamic_ncols=True
        )
        
        for batch in progress_bar:
            step_losses = self._run_step(batch, is_train=True)
            
            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            
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
        
        for batch in self.val_dataloader:
            step_losses = self._run_step(batch, is_train=False)
            
            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                
        avg_losses = {k: v / len(self.val_dataloader) for k, v in epoch_losses.items()}
        return avg_losses

    def train(self, start_epoch: int = 0):
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