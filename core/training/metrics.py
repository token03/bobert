# metrics.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import Dict, Any, Optional, List

from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric, CatMetric
from torchmetrics.classification import Accuracy, Precision, Recall

class PretrainEpochMetrics(nn.Module):
    def __init__(self, feature_info: Dict[str, Any], standard_cont_names: List[str], 
                 angle_pair_names: List[str], cat_feat_names: List[str], device: torch.device):
        super().__init__()
        self.feature_info = feature_info
        self.standard_cont_names = standard_cont_names
        self.angle_names = sorted([name for name in self.feature_info['continuous'] if 'angle' in name])
        self.angle_pair_names = angle_pair_names
        self.cat_feat_names = cat_feat_names

        self.standard_cont_metrics = MetricCollection({
            name: MeanMetric() for name in self.standard_cont_names
        }).to(device)

        self.cont_target_aggregator = CatMetric().to(device)

        self.angle_metrics = MetricCollection({
            name: MeanMetric() for name in self.angle_pair_names
        }).to(device)

        self.cat_metrics = nn.ModuleDict()
        for name, info in self.feature_info['categorical'].items():
            num_classes = info['cardinality']
            self.cat_metrics[name] = MetricCollection({
                'accuracy': Accuracy(task="multiclass", num_classes=num_classes),
                'precision': Precision(task="multiclass", num_classes=num_classes, average='macro', zero_division=0),
                'recall': Recall(task="multiclass", num_classes=num_classes, average='macro', zero_division=0),
                'target_aggregator': CatMetric(),
            }).to(device)

    def update(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor, mask: torch.Tensor):
        if not torch.any(mask):
            return

        standard_cont_indices = [self.feature_info['continuous'][name] for name in self.standard_cont_names]
        masked_preds = predictions['standard_continuous'][mask]
        masked_targets = targets[mask][:, standard_cont_indices]
        
        abs_errors = torch.abs(masked_preds - masked_targets)
        for i, name in enumerate(self.standard_cont_names):
            self.standard_cont_metrics[name].update(abs_errors[:, i])
        
        self.cont_target_aggregator.update(masked_targets)

        angle_indices = [self.feature_info['continuous'][name] for name in self.angle_names]
        masked_angle_preds = predictions['angle'][mask]
        masked_angle_targets = targets[mask][:, angle_indices]

        for i in range(len(self.angle_pair_names)):
            pair_slice = slice(i*2, (i+1)*2)
            preds_pair = masked_angle_preds[:, pair_slice]
            targets_pair = F.normalize(masked_angle_targets[:, pair_slice], p=2, dim=-1)
            
            dot_product = torch.sum(preds_pair * targets_pair, dim=-1).clamp(-1.0, 1.0)
            angle_errors_rad = torch.acos(dot_product)
            self.angle_metrics[self.angle_pair_names[i]].update(angle_errors_rad)

        for name, info in self.feature_info['categorical'].items():
            pred_logits = predictions['categorical'][name][mask]
            pred_classes = torch.argmax(pred_logits, dim=-1)
            target_classes = targets[mask][:, info['index']].long()
            self.cat_metrics[name]['accuracy'].update(pred_classes, target_classes)
            self.cat_metrics[name]['precision'].update(pred_classes, target_classes)
            self.cat_metrics[name]['recall'].update(pred_classes, target_classes)
            self.cat_metrics[name]['target_aggregator'].update(target_classes)

    def compute(self) -> Dict[str, Any]:
        results = {}
        
        cont_metrics = {}
        all_cont_targets = self.cont_target_aggregator.compute()
        if all_cont_targets.numel() > 0:
            mae_results = self.standard_cont_metrics.compute()
            mean_per_cont = all_cont_targets.mean(dim=0).cpu().tolist()
            std_per_cont = all_cont_targets.std(dim=0).cpu().tolist()
            for i, name in enumerate(self.standard_cont_names):
                cont_metrics[name] = {
                    'mae': mae_results[name].item(),
                    'mean': mean_per_cont[i],
                    'std': std_per_cont[i]
                }

        angle_results = self.angle_metrics.compute()
        for name, value in angle_results.items():
            if value.numel() > 0:
                cont_metrics[name] = {'mae_degrees': math.degrees(value.item())}
        
        if cont_metrics:
            results['continuous_metrics'] = cont_metrics

        cat_metrics = {}
        for name, collection in self.cat_metrics.items():
            cat_results = collection.compute()
            targets_for_dist = cat_results.pop('target_aggregator')
            
            if targets_for_dist.numel() > 0:
                cat_metrics[name] = {k: v.item() for k, v in cat_results.items()}
                cat_metrics[name]['distribution'] = Counter(targets_for_dist.cpu().numpy())
        
        if cat_metrics:
            results['categorical_metrics'] = cat_metrics
            
        return results
    
    def reset(self):
        self.standard_cont_metrics.reset()
        self.cont_target_aggregator.reset()
        self.angle_metrics.reset()
        for collection in self.cat_metrics.values():
            collection.reset()


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