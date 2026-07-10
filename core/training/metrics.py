import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import FBetaScore

from core.data.schema import DIFFICULTY_ATTRIBUTES
from core.data.normalizer import BeatmapNormalizer


class MLMMetrics(nn.Module):
    def __init__(self, feature_info: Dict[str, Any], device: torch.device):
        super().__init__()
        self.feature_info = feature_info
        self._device = device

        self.cont_names = sorted(
            feature_info["continuous"].keys(),
            key=lambda k: feature_info["continuous"][k],
        )

        self.slider_feature_names = set(feature_info.get("slider", {}).keys())

        self.cont_metrics = MetricCollection(
            {name: MeanMetric() for name in self.cont_names}
        ).to(device)

        self.cat_metrics = nn.ModuleDict()
        for name, info in feature_info["categorical"].items():
            num_classes = info["cardinality"]
            self.cat_metrics[name] = MetricCollection(
                {
                    "f2": FBetaScore(
                        task="multiclass",
                        num_classes=num_classes,
                        beta=2.0,
                        average="weighted",
                        zero_division=0,
                    )
                }
            ).to(device)

        self.loss_metric = MeanMetric().to(device)

    def update(
        self,
        predictions: Dict[str, Any],
        targets: torch.Tensor,
        loss: Optional[float] = None,
    ):
        from core.data.schema import OBJECT_TYPE_SLIDER_HEAD

        if loss is not None:
            self.loss_metric.update(loss)

        if targets.shape[0] == 0:
            return

        masked_targets = targets
        continuous_predictions = predictions["continuous"]
        categorical_predictions = predictions["categorical"]
        object_type_idx = self.feature_info["categorical"]["object_type"]["index"]
        object_types = masked_targets[:, object_type_idx].long()

        for i, name in enumerate(self.cont_names):
            target_idx = self.feature_info["continuous"][name]
            pred_idx = i

            if name in self.slider_feature_names:
                slider_mask = object_types == OBJECT_TYPE_SLIDER_HEAD
                if not torch.any(slider_mask):
                    continue
                preds = continuous_predictions[slider_mask, pred_idx]
                targs = masked_targets[slider_mask, target_idx]
            else:
                preds = continuous_predictions[:, pred_idx]
                targs = masked_targets[:, target_idx]

            abs_error = torch.abs(preds - targs)
            self.cont_metrics[name].update(abs_error)

        for name, info in self.feature_info["categorical"].items():
            pred_logits = categorical_predictions[name]
            pred_classes = torch.argmax(pred_logits, dim=-1)
            target_classes = masked_targets[:, info["index"]].long()
            self.cat_metrics[name]["f2"].update(pred_classes, target_classes)

    def compute(self) -> Dict[str, Any]:
        results = {}

        cont_results = self.cont_metrics.compute()
        continuous_metrics = {}
        for name in self.cont_names:
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
    def __init__(
        self, device: torch.device, normalizer: Optional[BeatmapNormalizer] = None
    ):
        super().__init__()
        self._device = device
        self.normalizer = normalizer

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
                label = labels[key]
                if self.normalizer is not None:
                    preds = self.normalizer.denormalize_attribute(key, preds)
                    label = self.normalizer.denormalize_attribute(key, label)
                mae = torch.abs(preds - label)
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
    def __init__(self, device: torch.device):
        super().__init__()
        self._device = device
        self.loss_metric = MeanMetric().to(device)

    def update(self, loss: Optional[float] = None):
        if loss is not None:
            self.loss_metric.update(loss)

    def compute(self) -> Dict[str, float]:
        results = {}
        if self.loss_metric.update_count > 0:
            results["contrastive_loss"] = self.loss_metric.compute().item()
        return results

    def reset(self):
        self.loss_metric.reset()
