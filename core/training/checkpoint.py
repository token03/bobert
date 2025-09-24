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
        suffix: str = "latest",
        vector_stats: Optional[Dict[str, Any]] = None,
        meta_stats: Optional[Dict[str, Any]] = None
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
            
        if vector_stats is not None:
            checkpoint_data['vector_stats'] = vector_stats
        
        if meta_stats is not None:
            checkpoint_data['meta_stats'] = meta_stats
            
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
    ) -> Tuple[int, Dict[str, float], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        checkpoint_path = self.get_checkpoint_path(suffix)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        state_dict = checkpoint['model_state_dict']
        
        compiled_model_prefix = '_orig_mod.'
        is_checkpoint_compiled = any(key.startswith(compiled_model_prefix) for key in state_dict.keys())
        is_model_compiled = getattr(model, 'is_compiled', False)
        
        if is_checkpoint_compiled and not is_model_compiled:
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith(compiled_model_prefix):
                    new_key = key[len(compiled_model_prefix):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        elif not is_checkpoint_compiled and is_model_compiled:
            new_state_dict = {}
            for key, value in state_dict.items():
                compiled_key = compiled_model_prefix + key
                new_state_dict[compiled_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        vector_stats = checkpoint.get('vector_stats')
        meta_stats = checkpoint.get('meta_stats')
        
        return checkpoint['epoch'], checkpoint.get('metrics', {}), vector_stats, meta_stats
    
    def load_normalization_stats(self, suffix: str = "latest") -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Loads only the normalization stats from a checkpoint."""
        checkpoint_path = self.get_checkpoint_path(suffix)
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for stats not found at {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        vector_stats = checkpoint.get('vector_stats')
        meta_stats = checkpoint.get('meta_stats')
        
        if vector_stats and meta_stats:
            return vector_stats, meta_stats
        
        print(f"Warning: Normalization stats not found in checkpoint {checkpoint_path}")
        return None

    def checkpoint_exists(self, suffix: str = "latest") -> bool:
        """Checks if a checkpoint exists."""
        return os.path.exists(self.get_checkpoint_path(suffix))