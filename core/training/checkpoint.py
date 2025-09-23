# checkpoint.py
import os
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, Tuple

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
