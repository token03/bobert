# trainer.py
import os
import time
import math
import numpy as np
from collections import defaultdict, Counter
from core.data.types import HitObjectVector
from typing import Dict, Any, Optional, Tuple, Callable
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from sklearn.metrics import precision_recall_fscore_support

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

def mlm_loss_fn(
    predictions: Dict[str, Any], 
    targets: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    num_masked = torch.sum(mask)
    if num_masked == 0:
        return torch.tensor(0.0, device=targets.device, requires_grad=True)

    feature_info = HitObjectVector.get_feature_info()
    total_loss = torch.tensor(0.0, device=targets.device)

    standard_cont_names = [name for name in feature_info['continuous'] if 'angle' not in name]
    standard_cont_indices = [feature_info['continuous'][name] for name in standard_cont_names]
    
    if standard_cont_indices:
        cont_preds = predictions['standard_continuous']
        cont_targets = targets[..., standard_cont_indices]
        cont_loss = F.mse_loss(cont_preds, cont_targets, reduction='none')
        masked_cont_loss = cont_loss[mask].sum()
        total_loss += masked_cont_loss

    angle_names = sorted([name for name in feature_info['continuous'] if 'angle' in name])
    angle_indices = [feature_info['continuous'][name] for name in angle_names]
    
    angle_preds = predictions['angle'] 
    angle_targets = targets[..., angle_indices]
    
    angle_targets_reshaped = angle_targets.view(*angle_targets.shape[:-1], -1, 2)
    angle_targets_norm = F.normalize(angle_targets_reshaped, p=2, dim=-1)
    angle_targets_norm = angle_targets_norm.view_as(angle_targets)
    
    angle_loss = F.mse_loss(angle_preds, angle_targets_norm, reduction='none')
    masked_angle_loss = angle_loss[mask].sum()
    total_loss += masked_angle_loss

    for name, info in feature_info['categorical'].items():
        cat_logits = predictions['categorical'][name]
        cat_targets = targets[..., info['index']].long()
        
        flat_logits = cat_logits.view(-1, info['cardinality'])
        flat_targets = cat_targets.view(-1)
        
        cat_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        cat_loss = cat_loss.view(mask.shape) 
        
        masked_cat_loss = cat_loss[mask].sum()
        total_loss += masked_cat_loss

    return total_loss / num_masked


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
    def __init__(self, checkpoint_dir: str, model_name: str = "model"):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def get_checkpoint_path(self, suffix: str = "latest") -> str:
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
        checkpoint_path = self.get_checkpoint_path(suffix)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
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
    def __init__(self):
        self.metrics = {}
        self.epoch_metrics = []
    
    def update(self, phase: str, **kwargs):
        if phase not in self.metrics:
            self.metrics[phase] = {}
        
        for key, value in kwargs.items():
            if key not in self.metrics[phase]:
                self.metrics[phase][key] = []
            self.metrics[phase][key].append(value)
    
    def get_latest(self, phase: str, metric: str) -> Optional[float]:
        if phase in self.metrics and metric in self.metrics[phase]:
            return self.metrics[phase][metric][-1]
        return None
    
    def get_average(self, phase: str, metric: str, last_n: int = 1) -> Optional[float]:
        if phase in self.metrics and metric in self.metrics[phase]:
            values = self.metrics[phase][metric][-last_n:]
            return sum(values) / len(values) if values else None
        return None
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float] = None):
        epoch_data = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics or {}
        }
        self.epoch_metrics.append(epoch_data)

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
        self.loss_fn = loss_fn
        
        self.use_amp = config['training'].get('use_amp', False) and device.type == 'cuda'
        self.grad_clip_norm = config['training'].get('grad_clip_norm', 1.0)
        
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.metrics_tracker = MetricsTracker()
        self.feature_info = HitObjectVector.get_feature_info()

        self.standard_cont_names = [name for name in self.feature_info['continuous'] if 'angle' not in name]
        self.angle_names = sorted([name for name in self.feature_info['continuous'] if 'angle' in name])
        self.cat_feat_names = list(self.feature_info['categorical'].keys())
        
        print(f"Trainer initialized - AMP: {self.use_amp}, Device: {device}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {epoch+1} [Train]", 
            dynamic_ncols=True
        )
        
        for vectors, attention_mask, metadata in progress_bar:
            vectors, attention_mask, metadata = vectors.to(self.device), attention_mask.to(self.device), metadata.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                predictions, targets, mask = self.model(vectors, metadata, attention_mask)
                loss = self.loss_fn(predictions, targets, mask)
            
            self.scaler.scale(loss).backward()
            
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler is not None: self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss, 'learning_rate': self.optimizer.param_groups[0]['lr']}
    
    def validate_epoch(self, epoch: int) -> Dict[str, Any]:
        self.model.eval()
        total_loss, num_batches, total_masked_count = 0.0, 0, 0
        
        total_abs_error_standard_cont = torch.zeros(len(self.standard_cont_names), device=self.device)
        total_angle_error_rad = torch.zeros(len(self.angle_names) // 2, device=self.device)
        
        all_masked_standard_cont_targets = []
        all_masked_cat_preds, all_masked_cat_targets = defaultdict(list), defaultdict(list)

        with torch.no_grad():
            for vectors, attention_mask, metadata in self.val_dataloader:
                vectors, attention_mask, metadata = vectors.to(self.device), attention_mask.to(self.device), metadata.to(self.device)
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                    predictions, targets, mask = self.model(vectors, metadata, attention_mask)
                    loss = self.loss_fn(predictions, targets, mask)
                total_loss += loss.item()
                num_batches += 1

                num_masked_in_batch = torch.sum(mask)
                if num_masked_in_batch > 0:
                    total_masked_count += num_masked_in_batch
                    
                    standard_cont_indices = [self.feature_info['continuous'][name] for name in self.standard_cont_names]
                    masked_standard_cont_preds = predictions['standard_continuous'][mask]
                    masked_standard_cont_targets = targets[mask][:, standard_cont_indices]
                    total_abs_error_standard_cont += torch.sum(torch.abs(masked_standard_cont_preds - masked_standard_cont_targets), dim=0)
                    all_masked_standard_cont_targets.append(masked_standard_cont_targets)
                    
                    angle_indices = [self.feature_info['continuous'][name] for name in self.angle_names]
                    masked_angle_preds = predictions['angle'][mask]
                    masked_angle_targets = targets[mask][:, angle_indices]

                    for i in range(len(self.angle_names) // 2):
                        pair_slice = slice(i*2, (i+1)*2)
                        preds_pair = masked_angle_preds[:, pair_slice] 
                        targets_pair = F.normalize(masked_angle_targets[:, pair_slice], p=2, dim=-1)

                        dot_product = torch.sum(preds_pair * targets_pair, dim=-1).clamp(-1.0, 1.0)
                        angle_errors_rad = torch.acos(dot_product)
                        total_angle_error_rad[i] += torch.sum(angle_errors_rad)

                    for name, info in self.feature_info['categorical'].items():
                        pred_classes = torch.argmax(predictions['categorical'][name][mask], dim=-1)
                        target_classes = targets[mask][:, info['index']].long()
                        all_masked_cat_preds[name].append(pred_classes)
                        all_masked_cat_targets[name].append(target_classes)
        
        avg_loss = total_loss / max(num_batches, 1)
        results = {'loss': avg_loss}
        if total_masked_count > 0:
            cont_metrics = {}
            mae_per_cont = (total_abs_error_standard_cont / total_masked_count).cpu().tolist()
            all_cont_targets_tensor = torch.cat(all_masked_standard_cont_targets, dim=0)
            mean_per_cont, std_per_cont = all_cont_targets_tensor.mean(dim=0).cpu().tolist(), all_cont_targets_tensor.std(dim=0).cpu().tolist()

            for i, name in enumerate(self.standard_cont_names):
                cont_metrics[name] = {'mae': mae_per_cont[i], 'mean': mean_per_cont[i], 'std': std_per_cont[i]}
            
            avg_angle_errors_rad = (total_angle_error_rad / total_masked_count).cpu().tolist()
            angle_pair_names = ['movement_angle', 'inner_angle'] 
            for i, name in enumerate(angle_pair_names):
                 cont_metrics[name] = {'mae_degrees': math.degrees(avg_angle_errors_rad[i])}
            
            results['continuous_metrics'] = cont_metrics

            cat_metrics = {}
            for name in self.cat_feat_names:
                preds, targets = torch.cat(all_masked_cat_preds[name]).cpu().numpy(), torch.cat(all_masked_cat_targets[name]).cpu().numpy()
                accuracy = np.mean(preds == targets)
                precision, recall, _, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
                cat_metrics[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'distribution': Counter(targets)}
            results['categorical_metrics'] = cat_metrics
        
        return results

    def train(self, start_epoch: int = 0) -> MetricsTracker:
        num_epochs = self.config['training']['num_epochs']
        
        print(f"\n--- Starting Training ---")
        print(f"Epochs: {start_epoch + 1} to {num_epochs}")
        print(f"Batch Size: {self.config['training']['batch_size']}")
        print(f"Learning Rate: {self.config['training']['learning_rate']}")
        print("-" * 60)
        
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
            
            print("=" * 70)
            print(f"{' ' * 21} DETAILED VALIDATION REPORT {' ' * 22}")

            if 'continuous_metrics' in val_metrics:
                print("-" * 70)
                print(" CONTINUOUS FEATURES:")
                header = f"  {'Feature':<22} | {'MAE / Error':<15} | {'Mean (True)':<12} | {'Std (True)':<12}"
                print(header)
                print(f"  {'-'*22}-+-{'-'*15}-+-{'-'*12}-+-{'-'*12}")
                for name in self.standard_cont_names:
                     if name in val_metrics['continuous_metrics']:
                        metrics = val_metrics['continuous_metrics'][name]
                        row = f"  {name:<22} | {metrics['mae']:<15.4f} | {metrics['mean']:<12.4f} | {metrics['std']:<12.4f}"
                        print(row)
                angle_pair_names = ['movement_angle', 'inner_angle']
                for name in angle_pair_names:
                    if name in val_metrics['continuous_metrics']:
                        metrics = val_metrics['continuous_metrics'][name]
                        row = f"  {name:<22} | {metrics['mae_degrees']:<15.4f} (deg) | {'-':<12} | {'-':<12}"
                        print(row)

            if 'categorical_metrics' in val_metrics:
                print("-" * 70)
                print(" CATEGORICAL FEATURES:")
                header = f"  {'Feature':<22} | {'Accuracy':<10} | {'Precision':<12} | {'Recall':<12}"
                print(header)
                print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
                for name, metrics in val_metrics['categorical_metrics'].items():
                    row = f"  {name:<22} | {metrics['accuracy']:<10.2%} | {metrics['precision']:<12.4f} | {metrics['recall']:<12.4f}"
                    print(row)
                    dist_data = metrics['distribution']
                    total_count = sum(dist_data.values())
                    if total_count == 0: continue
                    sorted_dist = sorted(dist_data.items(), key=lambda item: item[1], reverse=True)
                    dist_str_parts, limit = [], 5
                    if len(sorted_dist) > limit:
                        top_items = sorted_dist[:limit]
                        other_count = sum(count for _, count in sorted_dist[limit:])
                        for class_idx, count in top_items: dist_str_parts.append(f"{class_idx}:{count/total_count:.1%}")
                        if other_count > 0: dist_str_parts.append(f"Other:{other_count/total_count:.1%}")
                    else:
                        for class_idx, count in sorted_dist: dist_str_parts.append(f"{class_idx}:{count/total_count:.1%}")
                    print(f"    └─ True Dist: {', '.join(dist_str_parts)}")

            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.scaler,
                epoch, val_metrics
            )
            print("-" * 70)
            print(f"Checkpoint saved to {checkpoint_path}")
            print("=" * 70)
        
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