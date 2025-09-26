from pytorch_optimizer import get_wsd_schedule
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

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
) -> Optional[LRScheduler]:
    training_config = config['training']
    
    base_lr = float(training_config['learning_rate'])
    min_lr = float(training_config.get('min_lr', 1e-6))
    
    warmup_ratio = float(training_config.get('warmup_ratio', 0.05))
    stable_ratio = float(training_config.get('stable_ratio', 0.1))
    
    cooldown_type = training_config.get('cooldown_type', 'cosine') 
    num_cycles = float(training_config.get('num_cycles', 0.5)) 
    
    num_warmup_steps = int(warmup_ratio * total_steps)
    num_stable_steps = int(stable_ratio * total_steps)
    
    if num_warmup_steps + num_stable_steps >= total_steps:
            raise ValueError("The sum of warmup and stable steps must be less than total_steps.")

    num_decay_steps = total_steps - num_warmup_steps - num_stable_steps
    
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    print(f"Scheduler: WSD with {num_warmup_steps} warmup, {num_stable_steps} stable, {num_decay_steps} decay steps.")
    print(f"Cooldown type: {cooldown_type}, Min LR Ratio: {min_lr_ratio:.4f}")
    
    return get_wsd_schedule(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=min_lr_ratio,
        cooldown_type=cooldown_type,
        num_cycles=num_cycles
    )