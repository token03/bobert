import os
import time
import math
from typing import Dict, Any, Optional, Tuple, Callable
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.min_lr + (self.base_lr - self.min_lr) * self.last_epoch / self.warmup_steps
                   for _ in self.optimizer.param_groups]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
                   for _ in self.optimizer.param_groups]

def masked_mlm_loss_fn(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    """Calculates MSE loss only on masked tokens."""
    num_masked = torch.sum(mask)
    if num_masked == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    masked_predictions = predictions[mask]
    masked_targets = targets[mask]
    
    return nn.functional.mse_loss(masked_predictions, masked_targets)

def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    training_config = config['training']
    
    optimizer_type = training_config.get('optimizer', 'adamw')
    lr = float(training_config['learning_rate'])  
    weight_decay = float(training_config.get('weight_decay', 0.0))  
    
    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        momentum = float(training_config.get('momentum', 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: Optimizer, 
    config: Dict[str, Any], 
    total_steps: int
) -> Optional[_LRScheduler]:
    training_config = config['training']
    
    scheduler_type = training_config.get('scheduler', 'cosine_warmup')
    
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'cosine_warmup':
        warmup_ratio = float(training_config.get('warmup_ratio', 0.05))
        warmup_steps = int(warmup_ratio * total_steps)
        base_lr = float(training_config['learning_rate'])
        min_lr = float(training_config.get('min_lr', 1e-6))
        
        return CosineWarmupScheduler(
            optimizer, warmup_steps, total_steps, base_lr, min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class CheckpointManager:
    """Manages model checkpointing and loading."""
    
    def __init__(self, checkpoint_dir: str, model_name: str = "model"):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def get_checkpoint_path(self, suffix: str = "latest") -> str:
        """Gets the path for a checkpoint file."""
        return os.path.join(self.checkpoint_dir, f"{self.model_name}_{suffix}.pth")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        metrics: Dict[str, float],
        suffix: str = "latest"
    ):
        """Saves a training checkpoint."""
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
        if scaler is not None:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
            
        checkpoint_path = self.get_checkpoint_path(suffix)
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        suffix: str = "latest",
        device: torch.device = torch.device('cpu')
    ) -> Tuple[int, Dict[str, float]]:
        """
        Loads a training checkpoint.
        
        Returns:
            Tuple of (epoch, metrics)
        """
        checkpoint_path = self.get_checkpoint_path(suffix)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint['epoch'], checkpoint.get('metrics', {})
    
    def checkpoint_exists(self, suffix: str = "latest") -> bool:
        """Checks if a checkpoint exists."""
        return os.path.exists(self.get_checkpoint_path(suffix))


class MetricsTracker:
    """Tracks training and validation metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.epoch_metrics = []
    
    def update(self, phase: str, **kwargs):
        """Updates metrics for a given phase."""
        if phase not in self.metrics:
            self.metrics[phase] = {}
        
        for key, value in kwargs.items():
            if key not in self.metrics[phase]:
                self.metrics[phase][key] = []
            self.metrics[phase][key].append(value)
    
    def get_latest(self, phase: str, metric: str) -> Optional[float]:
        """Gets the latest value for a metric."""
        if phase in self.metrics and metric in self.metrics[phase]:
            return self.metrics[phase][metric][-1]
        return None
    
    def get_average(self, phase: str, metric: str, last_n: int = 1) -> Optional[float]:
        """Gets the average of the last N values for a metric."""
        if phase in self.metrics and metric in self.metrics[phase]:
            values = self.metrics[phase][metric][-last_n:]
            return sum(values) / len(values) if values else None
        return None
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float] = None):
        """Logs metrics for an epoch."""
        epoch_data = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics or {}
        }
        self.epoch_metrics.append(epoch_data)


def mlm_loss_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if predictions.numel() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    return nn.functional.mse_loss(predictions, targets)


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
        loss_fn: Callable = masked_mlm_loss_fn
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        self.loss_fn = loss_fn
        
        self.use_amp = config['training'].get('use_amp', False) and device.type == 'cuda'
        self.grad_clip_norm = config['training'].get('grad_clip_norm', 1.0)
        
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)

        self.metrics_tracker = MetricsTracker()
        
        print(f"Trainer initialized - AMP: {self.use_amp}, Device: {device}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Trains for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {epoch+1} [Train]", 
            dynamic_ncols=True
        )
        
        for vectors, attention_mask, metadata in progress_bar:
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                all_predictions, targets, mask = self.model(vectors, metadata, attention_mask)
                loss = self.loss_fn(all_predictions, targets, mask)
            
            self.scaler.scale(loss).backward()
            
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / max(num_batches, 1)
        return {
            'loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validates for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for vectors, attention_mask, metadata in self.val_dataloader:
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    all_predictions, targets, mask = self.model(vectors, metadata, attention_mask)
                    loss = self.loss_fn(all_predictions, targets, mask)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}
    
    def train(self, start_epoch: int = 0) -> MetricsTracker:
        """
        Main training loop.
        
        Args:
            start_epoch: Epoch to start training from
            
        Returns:
            MetricsTracker with training history
        """
        num_epochs = self.config['training']['num_epochs']
        
        print(f"\n--- Starting Training ---")
        print(f"Epochs: {start_epoch + 1} to {num_epochs}")
        print(f"Batch Size: {self.config['training']['batch_size']}")
        print(f"Learning Rate: {self.config['training']['learning_rate']}")
        print("-" * 40)
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            train_metrics = self.train_epoch(epoch)
            
            val_metrics = self.validate_epoch(epoch)
            
            self.metrics_tracker.log_epoch(epoch, train_metrics, val_metrics)
            
            epoch_duration = time.time() - epoch_start_time
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"LR: {train_metrics['learning_rate']:.2e} | "
                f"Time: {epoch_duration:.2f}s"
            )
            
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.scaler,
                epoch, val_metrics
            )
            print(f"Checkpoint saved to {checkpoint_path}")
            print("-" * 40)
        
        print("\n--- Training Finished ---")
        return self.metrics_tracker


def setup_training(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[MLMTrainer, CheckpointManager]:
    optimizer = create_optimizer(model, config)
    
    total_steps = len(train_dataloader) * config['training']['num_epochs']
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    checkpoint_dir = config['training']['checkpoint_dir']
    model_name = config['model'].get('type', 'model')
    checkpoint_manager = CheckpointManager(checkpoint_dir, model_name)
    
    trainer = MLMTrainer(
        model, train_dataloader, val_dataloader,
        optimizer, scheduler, config, device, checkpoint_manager
    )
    
    return trainer, checkpoint_manager