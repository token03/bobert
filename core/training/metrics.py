# metrics.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import Dict, Any, Optional, List

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
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


class FineTuneEpochMetrics(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.k_values = config['finetuning'].get('validation_k_values', [1, 5, 10])

        self.batch_metrics = MetricCollection({
            'total_loss': MeanMetric(),
            'user_tag_loss': MeanMetric(),
            'collection_label_loss': MeanMetric(),
            'difficulty_rating_loss': MeanMetric(),
            'user_tag_contrastive_loss': MeanMetric(),
            'collection_label_contrastive_loss': MeanMetric(),
            'difficulty_contrastive_loss': MeanMetric(),
        })

    def update_batch_metrics(self, losses: Dict[str, float]):
        for name, value in losses.items():
            if name in self.batch_metrics:
                if not math.isnan(value):
                    self.batch_metrics[name].update(value)

    def compute_batch_metrics(self) -> Dict[str, float]:
        results = {}
        for name, metric in self.batch_metrics.items():
            if metric.update_count > 0:
                computed_value = metric.compute()
                if not torch.isnan(computed_value) and computed_value > 0:
                    results[name] = computed_value.item()
        return results

    def reset_batch_metrics(self):
        self.batch_metrics.reset()

    def compute_embedding_metrics(
        self,
        all_embeddings: torch.Tensor,
        all_ratings: torch.Tensor,
        all_labels: List[List[str]],
    ) -> Dict[str, float]:
        print(f"Calculating embedding metrics on {len(all_embeddings)} validation samples...")
        metrics = {}
        device = all_embeddings.device

        all_embeddings = F.normalize(all_embeddings.to(torch.float32), p=2, dim=1)
        sim_matrix = torch.matmul(all_embeddings, all_embeddings.T)

        rating_diffs = torch.abs(all_ratings.unsqueeze(0) - all_ratings.unsqueeze(1))
        
        difficulty_mask = rating_diffs <= self.config['finetuning']['positive_difficulty_threshold']
        label_mask = torch.zeros_like(difficulty_mask, dtype=torch.bool, device=device)
        for i in range(len(all_labels)):
            set_i = set(all_labels[i])
            if not set_i: continue
            for j in range(i, len(all_labels)):
                set_j = set(all_labels[j])
                if set_i.intersection(set_j):
                    label_mask[i, j] = True
                    label_mask[j, i] = True
        
        true_positives_mask = difficulty_mask & label_mask
        torch.diagonal(true_positives_mask).fill_(False)

        continuous_relevance = torch.zeros_like(rating_diffs, dtype=torch.float32)
        continuous_relevance.masked_scatter_(
            label_mask,
            1.0 / (1.0 + rating_diffs[label_mask])
        )
        torch.diagonal(continuous_relevance).fill_(0)

        sim_matrix_for_ranking = sim_matrix.clone()
        sim_matrix_for_ranking.fill_diagonal_(-torch.inf)
        self._compute_recall_at_k(metrics, sim_matrix_for_ranking, true_positives_mask)

        self._compute_spearman_rho(metrics, sim_matrix, all_ratings)

        self._compute_ndcg_at_k(metrics, sim_matrix, continuous_relevance)

        return metrics

    def _compute_recall_at_k(self, metrics: Dict, sim_matrix: torch.Tensor, true_positives_mask: torch.Tensor):
        num_queries_with_positives = (true_positives_mask.sum(dim=1) > 0).sum().item()
        if num_queries_with_positives == 0:
            for k in self.k_values: metrics[f'Recall@{k}'] = 0.0
            return

        _, topk_indices = torch.topk(sim_matrix, max(self.k_values), dim=1)
        for k in self.k_values:
            hits_at_k = torch.gather(true_positives_mask, 1, topk_indices[:, :k]).any(dim=1)
            valid_queries_mask = true_positives_mask.any(dim=1)
            recall_at_k = hits_at_k[valid_queries_mask].sum().item() / num_queries_with_positives
            metrics[f'Recall@{k}'] = recall_at_k

    def _compute_spearman_rho(self, metrics: Dict, sim_matrix: torch.Tensor, all_ratings: torch.Tensor):
        num_samples = len(all_ratings)
        num_pairs = min(num_samples * 10, 100000)
        idx1 = torch.randint(0, num_samples, (num_pairs,), device=sim_matrix.device)
        idx2 = torch.randint(0, num_samples, (num_pairs,), device=sim_matrix.device)
        
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        sim_scores = sim_matrix[idx1, idx2] 
        rating_distances = torch.abs(all_ratings[idx1] - all_ratings[idx2])

        rho, _ = spearmanr(sim_scores.cpu().numpy(), -rating_distances.cpu().numpy()) # Higher sim should correlate with lower distance
        metrics['SpearmanRho'] = rho if not np.isnan(rho) else 0.0

    def _compute_ndcg_at_k(self, metrics: Dict, sim_matrix: torch.Tensor, continuous_relevance: torch.Tensor):
        num_queries = sim_matrix.shape[0]
        num_queries_to_eval = min(5000, num_queries)
        
        if num_queries_to_eval == 0:
            for k in self.k_values: metrics[f'nDCG@{k}'] = 0.0
            return
        
        query_indices = torch.randperm(num_queries, device=sim_matrix.device)[:num_queries_to_eval]

        total_ndcg = {k: 0.0 for k in self.k_values}
        sim_matrix_cpu = sim_matrix.cpu().numpy()
        continuous_relevance_cpu = continuous_relevance.cpu().numpy()

        for i in query_indices:
            true_relevances_for_query = continuous_relevance_cpu[i].reshape(1, -1)
            pred_scores_for_query = sim_matrix_cpu[i].reshape(1, -1)
            for k in self.k_values:
                total_ndcg[k] += ndcg_score(true_relevances_for_query, pred_scores_for_query, k=k)
        
        for k in self.k_values:
            metrics[f'nDCG@{k}'] = total_ndcg[k] / num_queries_to_eval


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