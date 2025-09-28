# pretrainer.py
import time
import math
from core.data.types import HitObjectVector
from typing import Dict, Any, Optional, Tuple, Callable
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from core.training.checkpoint import CheckpointManager
from core.training.metrics import MetricsTracker, PretrainEpochMetrics 
from core.training.optimization import create_optimizer, create_scheduler
from .loss import mlm_loss_fn
from core.logger import TrainingLogger
from ..data.transforms import BeatmapNormalizer

class MLMTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_manager: CheckpointManager,
        normalizer: BeatmapNormalizer,
        loss_fn: Callable = mlm_loss_fn
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.normalizer = normalizer
        self.loss_fn = loss_fn
        
        self.use_amp = config['pretraining'].get('use_amp', False) and device.type == 'cuda'
        self.grad_clip_norm = config['pretraining'].get('grad_clip_norm', 1.0)
        self.grad_accum_steps = config['pretraining'].get('gradient_accumulation_steps', 1)

        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        self.metrics_tracker = MetricsTracker()
        self.feature_info = HitObjectVector.get_feature_info()

        self.standard_cont_names = [name for name in self.feature_info['continuous'] if 'angle' not in name]
        self.angle_pair_names = ['flow_angle', 'entry_angle'] 
        self.cat_feat_names = list(self.feature_info['categorical'].keys())
        
        self.val_metrics = PretrainEpochMetrics(
            feature_info=self.feature_info,
            standard_cont_names=self.standard_cont_names,
            angle_pair_names=self.angle_pair_names,
            cat_feat_names=self.cat_feat_names,
            device=self.device
        )
        
        self.logger = TrainingLogger(
            standard_cont_names=self.standard_cont_names,
            angle_pair_names=self.angle_pair_names,
            cat_feat_names=self.cat_feat_names
        )
        
        print(f"Trainer initialized - AMP: {self.use_amp}, Device: {device}, Grad Accum: {self.grad_accum_steps}")
        if self.train_dataloader.batch_sampler is not None:
            effective_batch_size = self.train_dataloader.batch_sampler.batch_size * self.grad_accum_steps
        else:
            effective_batch_size = self.train_dataloader.batch_size * self.grad_accum_steps
        print(f"Effective batch size: {effective_batch_size}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        
        num_update_steps = math.ceil(len(self.train_dataloader) / self.grad_accum_steps)
        progress_bar = tqdm(
            total=num_update_steps,
            desc=f"Epoch {epoch+1} [Train]",
            dynamic_ncols=True
        )
        
        data_iter = iter(self.train_dataloader)
        for i in range(len(self.train_dataloader)):
            vectors, attention_mask, metadata = next(data_iter)
            vectors, attention_mask, metadata = vectors.to(self.device), attention_mask.to(self.device), metadata.to(self.device)
            
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                predictions, targets, mask = self.model(vectors, metadata, attention_mask)
                loss = self.loss_fn(predictions, targets, mask)
                scaled_loss = loss / self.grad_accum_steps
            
            self.scaler.scale(scaled_loss).backward()
            
            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(self.train_dataloader):
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.scheduler is not None: self.scheduler.step()
                
                progress_bar.update(1)
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
        progress_bar.close()
        avg_loss = total_loss / len(self.train_dataloader)
        return {'loss': avg_loss, 'learning_rate': self.optimizer.param_groups[0]['lr']}
    
    def validate_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.eval()
        total_loss, num_batches = 0.0, 0
        
        self.val_metrics.reset()
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_dataloader,
                desc=f"Epoch {epoch+1} [Validate]",
                dynamic_ncols=True,
                leave=False
            )
            for vectors, attention_mask, metadata in progress_bar:
                vectors, attention_mask, metadata = vectors.to(self.device), attention_mask.to(self.device), metadata.to(self.device)
                
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    predictions, targets, mask = self.model(vectors, metadata, attention_mask)
                    loss = self.loss_fn(predictions, targets, mask)
                
                total_loss += loss.item()
                num_batches += 1

                self.val_metrics.update(predictions, targets, mask)
        
        results = self.val_metrics.compute()
        results['loss'] = total_loss / max(num_batches, 1)
        
        return results

    def train(self, start_epoch: int = 0) -> MetricsTracker:
        num_epochs = self.config['pretraining']['num_epochs']
        
        self.logger.log_training_start(start_epoch, num_epochs, self.config)
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            self.metrics_tracker.log_epoch(epoch, train_metrics, val_metrics)
            
            epoch_duration = time.time() - epoch_start_time

            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.scaler,
                epoch, val_metrics,
                vector_stats=self.normalizer.get_vector_stats(),
                meta_stats=self.normalizer.get_metadata_stats()
            )
            
            self.logger.log_epoch_end(
                epoch=epoch,
                num_epochs=num_epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                duration=epoch_duration,
                checkpoint_path=checkpoint_path
            )
        
        self.logger.log_training_end()
        return self.metrics_tracker


def setup_training(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    normalizer: BeatmapNormalizer
) -> Tuple[MLMTrainer, CheckpointManager]:
    grad_accum_steps = config['pretraining'].get('gradient_accumulation_steps', 1)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    total_steps = num_update_steps_per_epoch * config['pretraining']['num_epochs']

    optimizer = create_optimizer(model, config, 'pretraining')
    scheduler = create_scheduler(optimizer, config, total_steps, 'pretraining')
    
    checkpoint_dir = config['pretraining']['checkpoint_dir']
    model_name = config['model'].get('type', 'model')
    checkpoint_manager = CheckpointManager(checkpoint_dir, model_name)
    
    trainer = MLMTrainer(
        model, train_dataloader, val_dataloader,
        optimizer, scheduler, config, device, checkpoint_manager,
        normalizer=normalizer
    )
    
    return trainer, checkpoint_manager