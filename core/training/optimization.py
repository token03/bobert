import math
from typing import Any, Dict, Optional
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