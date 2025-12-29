import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import Accuracy, Precision, Recall

from core.data.types import DIFFICULTY_ATTRIBUTES


class MLMMetrics(nn.Module):
    def __init__(self, feature_info: Dict[str, Any], device: torch.device):
        super().__init__()
        self.feature_info = feature_info
        self._device = device

        self.cont_names = sorted(
            feature_info["continuous"].keys(),
            key=lambda k: feature_info["continuous"][k],
        )

        slider_features = set(feature_info.get("slider", {}).keys())
        self.standard_cont_names = [
            n for n in self.cont_names if n not in slider_features
        ]

        self.cont_metrics = MetricCollection(
            {name: MeanMetric() for name in self.standard_cont_names}
        ).to(device)

        self.cat_metrics = nn.ModuleDict()
        for name, info in feature_info["categorical"].items():
            num_classes = info["cardinality"]
            self.cat_metrics[name] = MetricCollection(
                {
                    "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
                    "precision": Precision(
                        task="multiclass",
                        num_classes=num_classes,
                        average="weighted",
                        zero_division=0,
                    ),
                    "recall": Recall(
                        task="multiclass",
                        num_classes=num_classes,
                        average="weighted",
                        zero_division=0,
                    ),
                }
            ).to(device)

        self.loss_metric = MeanMetric().to(device)

    def update(
        self,
        predictions: Dict[str, Any],
        targets: torch.Tensor,
        mask: torch.Tensor,
        loss: Optional[float] = None,
    ):
        if loss is not None:
            self.loss_metric.update(loss)

        if not torch.any(mask):
            return

        cont_preds_masked = predictions["continuous"][mask]

        std_cont_target_indices = [
            self.feature_info["continuous"][name] for name in self.standard_cont_names
        ]
        std_cont_targets = targets[mask][:, std_cont_target_indices]

        std_cont_pred_indices = [
            self.cont_names.index(name) for name in self.standard_cont_names
        ]
        std_cont_preds = cont_preds_masked[:, std_cont_pred_indices]

        abs_errors = torch.abs(std_cont_preds - std_cont_targets)
        for i, name in enumerate(self.standard_cont_names):
            self.cont_metrics[name].update(abs_errors[:, i])

        for name, info in self.feature_info["categorical"].items():
            pred_logits = predictions["categorical"][name][mask]
            pred_classes = torch.argmax(pred_logits, dim=-1)
            target_classes = targets[mask][:, info["index"]].long()
            self.cat_metrics[name]["accuracy"].update(pred_classes, target_classes)
            self.cat_metrics[name]["precision"].update(pred_classes, target_classes)
            self.cat_metrics[name]["recall"].update(pred_classes, target_classes)

    def compute(self) -> Dict[str, Any]:
        results = {}

        cont_results = self.cont_metrics.compute()
        continuous_metrics = {}
        for name in self.standard_cont_names:
            if name in cont_results:
                continuous_metrics[name] = {"mae": cont_results[name].item()}
        if continuous_metrics:
            results["continuous_metrics"] = continuous_metrics

        categorical_metrics = {}
        for name, collection in self.cat_metrics.items():
            cat_results = collection.compute()
            if any(v.numel() > 0 for v in cat_results.values()):
                categorical_metrics[name] = {
                    k: v.item() for k, v in cat_results.items()
                }
        if categorical_metrics:
            results["categorical_metrics"] = categorical_metrics

        if self.loss_metric.update_count > 0:
            results["mlm_loss"] = self.loss_metric.compute().item()

        return results

    def reset(self):
        self.cont_metrics.reset()
        for collection in self.cat_metrics.values():
            collection.reset()
        self.loss_metric.reset()


class DifficultyMetrics(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self._device = device

        self.attr_metrics = MetricCollection(
            {f"{name}_mae": MeanMetric() for name in DIFFICULTY_ATTRIBUTES}
        ).to(device)

        self.loss_metric = MeanMetric().to(device)

    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        loss: Optional[float] = None,
    ):
        if loss is not None:
            self.loss_metric.update(loss)

        for key, preds in predictions.items():
            if key in labels:
                mae = torch.abs(preds - labels[key])
                self.attr_metrics[f"{key}_mae"].update(mae)

    def compute(self) -> Dict[str, Any]:
        results = {}

        attr_results = self.attr_metrics.compute()
        difficulty_metrics = {
            k: v.item() for k, v in attr_results.items() if v.numel() > 0
        }
        if difficulty_metrics:
            results["difficulty_metrics"] = difficulty_metrics

        if self.loss_metric.update_count > 0:
            results["difficulty_loss"] = self.loss_metric.compute().item()

        return results

    def reset(self):
        self.attr_metrics.reset()
        self.loss_metric.reset()


class ContrastiveMetrics(nn.Module):
    def __init__(self, k_values: List[int], device: torch.device):
        super().__init__()
        self.k_values = k_values
        self._device = device
        self.loss_metric = MeanMetric().to(device)

    def update(
        self, embeddings: torch.Tensor, labels: Any, loss: Optional[float] = None
    ):
        if loss is not None:
            self.loss_metric.update(loss)

    def compute(self) -> Dict[str, float]:
        results = {}
        if self.loss_metric.update_count > 0:
            results["contrastive_loss"] = self.loss_metric.compute().item()
        for k in self.k_values:
            results[f"Recall@{k}"] = 0.0
            results[f"nDCG@{k}"] = 0.0
        return results

    def reset(self):
        self.loss_metric.reset()
