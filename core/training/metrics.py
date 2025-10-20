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
                 cat_feat_names: List[str], device: torch.device):
        super().__init__()
        self.feature_info = feature_info
        self.standard_cont_names = standard_cont_names
        self.cat_feat_names = cat_feat_names

        self.standard_cont_metrics = MetricCollection({
            name: MeanMetric() for name in self.standard_cont_names
        }).to(device)
        self.cont_target_aggregator = CatMetric().to(device)
        self.cat_metrics = nn.ModuleDict()
        for name, info in self.feature_info['categorical'].items():
            num_classes = info['cardinality']
            self.cat_metrics[name] = MetricCollection({
                'accuracy': Accuracy(task="multiclass", num_classes=num_classes),
                'precision': Precision(task="multiclass", num_classes=num_classes, average='macro', zero_division=0),
                'recall': Recall(task="multiclass", num_classes=num_classes, average='macro', zero_division=0),
                'target_aggregator': CatMetric(),
            }).to(device)
        
        self.difficulty_metrics = MetricCollection({
            'stars_mae': MeanMetric(),
            'aim_mae': MeanMetric(),
            'speed_mae': MeanMetric(),
            'slider_factor_mae': MeanMetric()
        }).to(device)

    def update(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor, mask: torch.Tensor, difficulty_labels: Dict[str, torch.Tensor]):
        if torch.any(mask):
            cont_preds_masked = predictions['mlm']['continuous'][mask]
            cont_names_ordered = sorted(self.feature_info['continuous'].keys(), key=lambda k: self.feature_info['continuous'][k])
            
            std_cont_target_indices = [self.feature_info['continuous'][name] for name in self.standard_cont_names]
            std_cont_targets = targets[mask][:, std_cont_target_indices]
            
            std_cont_pred_indices = [cont_names_ordered.index(name) for name in self.standard_cont_names]
            std_cont_preds = cont_preds_masked[:, std_cont_pred_indices]

            abs_errors = torch.abs(std_cont_preds - std_cont_targets)
            for i, name in enumerate(self.standard_cont_names):
                self.standard_cont_metrics[name].update(abs_errors[:, i])
            self.cont_target_aggregator.update(std_cont_targets)

            for name, info in self.feature_info['categorical'].items():
                pred_logits = predictions['mlm']['categorical'][name][mask]
                pred_classes = torch.argmax(pred_logits, dim=-1)
                target_classes = targets[mask][:, info['index']].long()
                self.cat_metrics[name]['accuracy'].update(pred_classes, target_classes)
                self.cat_metrics[name]['precision'].update(pred_classes, target_classes)
                self.cat_metrics[name]['recall'].update(pred_classes, target_classes)
                self.cat_metrics[name]['target_aggregator'].update(target_classes)

        for key, preds in predictions['difficulty'].items():
            if key in difficulty_labels:
                mae = torch.abs(preds - difficulty_labels[key])
                self.difficulty_metrics[f'{key}_mae'].update(mae)

    def compute(self) -> Dict[str, Any]:
        results = {}
        
        cont_metrics = {}
        all_cont_targets = self.cont_target_aggregator.compute()
        if all_cont_targets.numel() > 0:
            mae_results = self.standard_cont_metrics.compute()
            targets_np = all_cont_targets.cpu().numpy()

            for i, name in enumerate(self.standard_cont_names):
                target_values = targets_np[:, i]
                mae = mae_results[name].item()

                median = np.median(target_values)
                q25, q75 = np.percentile(target_values, [25, 75])
                iqr = q75 - q25
                mape = np.mean(np.abs((target_values - median) / (median + 1e-8))) * 100

                cont_metrics[name] = {
                    'mae': mae,
                    'median': median,
                    'iqr': iqr,
                    'mape': mape,
                    'range_min': np.min(target_values),
                    'range_max': np.max(target_values)
                }
        if cont_metrics:
            results['continuous_metrics'] = cont_metrics

        cat_metrics = {}
        for name, collection in self.cat_metrics.items():
            cat_results = collection.compute()
            targets_for_dist = cat_results.pop('target_aggregator')
            if targets_for_dist.numel() > 0:
                cat_metrics[name] = {k: v.item() for k, v in cat_results.items()}
                targets_np = targets_for_dist.cpu().numpy()
                unique_classes, class_counts = np.unique(targets_np, return_counts=True)
                class_balance = class_counts / len(targets_np)
                cat_metrics[name]['class_balance_entropy'] = -np.sum(class_balance * np.log(class_balance + 1e-8))
                cat_metrics[name]['num_active_classes'] = len(unique_classes)
        if cat_metrics:
            results['categorical_metrics'] = cat_metrics
            
        diff_results = self.difficulty_metrics.compute()
        difficulty_metrics = {k: v.item() for k, v in diff_results.items() if v.numel() > 0}
        if difficulty_metrics:
            results['difficulty_metrics'] = difficulty_metrics
            
        return results
    
    def reset(self):
        self.standard_cont_metrics.reset()
        self.cont_target_aggregator.reset()
        for collection in self.cat_metrics.values():
            collection.reset()
        self.difficulty_metrics.reset()


class FineTuneEpochMetrics(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.k_values = config['finetuning'].get('recall_k_values', [1, 5, 10])

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

        if all_ratings.dim() > 1 and all_ratings.shape[1] > 1:
            stars_ratings = all_ratings[:, 0]
        else:
            stars_ratings = all_ratings

        rating_diffs = torch.abs(stars_ratings.unsqueeze(0) - stars_ratings.unsqueeze(1))

        difficulty_mask = rating_diffs <= self.config['finetuning']['recall_difficulty_threshold']
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

        self._compute_ndcg_at_k(metrics, sim_matrix, continuous_relevance)

        return metrics

    def _compute_recall_at_k(self, metrics: Dict, sim_matrix: torch.Tensor, true_positives_mask: torch.Tensor):
        num_queries_with_positives = (true_positives_mask.sum(dim=1) > 0).sum().item()
        if num_queries_with_positives == 0:
            for k in self.k_values: metrics[f'Recall@{k}'] = 0.0
            return

        max_k = min(max(self.k_values), sim_matrix.shape[1] - 1)
        if max_k == 0:
            for k in self.k_values: metrics[f'Recall@{k}'] = 0.0
            return

        _, topk_indices = torch.topk(sim_matrix, max_k, dim=1)
        for k in self.k_values:
            if k > max_k:
                metrics[f'Recall@{k}'] = 0.0
                continue
            hits_at_k = torch.gather(true_positives_mask, 1, topk_indices[:, :k]).any(dim=1)
            valid_queries_mask = true_positives_mask.any(dim=1)
            recall_at_k = hits_at_k[valid_queries_mask].sum().item() / num_queries_with_positives
            metrics[f'Recall@{k}'] = recall_at_k

    def _compute_ndcg_at_k(self, metrics: Dict, sim_matrix: torch.Tensor, continuous_relevance: torch.Tensor):
        num_queries = sim_matrix.shape[0]
        num_queries_to_eval = min(2000, num_queries)  

        if num_queries_to_eval == 0:
            for k in self.k_values: metrics[f'nDCG@{k}'] = 0.0
            return

        query_indices = torch.randperm(num_queries, device=sim_matrix.device)[:num_queries_to_eval]

        total_ndcg = {k: 0.0 for k in self.k_values}
        sim_matrix_cpu = sim_matrix.cpu().numpy()
        continuous_relevance_cpu = continuous_relevance.cpu().numpy()

        max_k = min(max(self.k_values), sim_matrix.shape[1] - 1)

        for i in query_indices:
            true_relevances_for_query = continuous_relevance_cpu[i].reshape(1, -1)
            pred_scores_for_query = sim_matrix_cpu[i].reshape(1, -1)
            for k in self.k_values:
                if k <= max_k:
                    total_ndcg[k] += ndcg_score(true_relevances_for_query, pred_scores_for_query, k=k)

        for k in self.k_values:
            if k <= max_k:
                metrics[f'nDCG@{k}'] = total_ndcg[k] / num_queries_to_eval
            else:
                metrics[f'nDCG@{k}'] = 0.0


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