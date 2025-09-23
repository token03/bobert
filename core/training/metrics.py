# metrics.py
import torch
from typing import Dict, Any, Optional, List

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