import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from core.data.schema import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER

from torchmetrics import MetricCollection
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import FBetaScore


class GeometryMetrics(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("map_sum", torch.zeros(dim), persistent=False)
        self.register_buffer("map_outer", torch.zeros(dim, dim), persistent=False)
        self.register_buffer("map_unit_sum", torch.zeros(dim), persistent=False)
        self.register_buffer("token_unit_sum", torch.zeros(dim), persistent=False)
        self.register_buffer(
            "map_count", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "token_count", torch.zeros((), dtype=torch.long), persistent=False
        )

    def update(self, packed_output: torch.Tensor, cu_seqlens: torch.Tensor):
        output = packed_output.float()
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        maps = torch.segment_reduce(output, reduce="mean", lengths=lengths)

        self.map_sum.add_(maps.sum(dim=0))
        self.map_outer.add_(maps.T @ maps)
        self.map_unit_sum.add_(torch.nn.functional.normalize(maps, dim=-1).sum(dim=0))
        self.token_unit_sum.add_(
            torch.nn.functional.normalize(output, dim=-1).sum(dim=0)
        )
        self.map_count.add_(maps.shape[0])
        self.token_count.add_(output.shape[0])

    def compute(self) -> Dict[str, float]:
        map_count = int(self.map_count.item())
        token_count = int(self.token_count.item())
        if map_count < 2 or token_count == 0:
            return {}

        map_sum = self.map_sum.double().cpu()
        covariance = (
            self.map_outer.double().cpu() - torch.outer(map_sum, map_sum) / map_count
        )
        eigenvalues = torch.linalg.eigvalsh(covariance / (map_count - 1)).clamp_min(0)
        total = eigenvalues.sum().clamp_min(1e-30)
        probabilities = eigenvalues / total
        probabilities = probabilities[probabilities > 0]
        effective_rank = torch.exp(-(probabilities * probabilities.log()).sum())

        return {
            "map_effective_rank": float(effective_rank),
            "map_pc1_ratio": float(eigenvalues[-1] / total),
            "map_anisotropy": float(
                (self.map_unit_sum / map_count).square().sum().item()
            ),
            "token_anisotropy": float(
                (self.token_unit_sum / token_count).square().sum().item()
            ),
        }

    def reset(self):
        self.map_sum.zero_()
        self.map_outer.zero_()
        self.map_unit_sum.zero_()
        self.token_unit_sum.zero_()
        self.map_count.zero_()
        self.token_count.zero_()


class MLMMetrics(nn.Module):
    def __init__(self, feature_info: Dict[str, Any], device: torch.device):
        super().__init__()
        self.feature_info = feature_info
        self._device = device

        self.groups = ("common", "slider", "spinner")
        self.cont_names = {
            group: tuple(
                name
                for name in feature_info[group]
                if name in feature_info["continuous"]
            )
            for group in self.groups
        }
        self.cat_names = {
            group: tuple(
                name
                for name in feature_info[group]
                if name in feature_info["categorical"]
            )
            for group in self.groups
        }

        self.cont_metrics = MetricCollection(
            {name: MeanMetric() for names in self.cont_names.values() for name in names}
        ).to(device)

        self.cat_metrics = nn.ModuleDict()
        for name in (name for names in self.cat_names.values() for name in names):
            info = feature_info["categorical"][name]
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
        if loss is not None:
            self.loss_metric.update(loss)

        if targets.shape[0] == 0:
            return

        object_type_idx = self.feature_info["categorical"]["object_type"]["index"]
        object_types = targets[:, object_type_idx].long()
        masks = {
            "common": torch.ones_like(object_types, dtype=torch.bool),
            "slider": object_types == OBJECT_TYPE_SLIDER,
            "spinner": object_types == OBJECT_TYPE_SPINNER,
        }

        for group, mask in masks.items():
            if not torch.any(mask):
                continue

            output = predictions[group]
            for pred_idx, name in enumerate(self.cont_names[group]):
                target_idx = self.feature_info["continuous"][name]
                abs_error = torch.abs(
                    output["continuous"][mask, pred_idx] - targets[mask, target_idx]
                )
                self.cont_metrics[name].update(abs_error)

            for name in self.cat_names[group]:
                info = self.feature_info["categorical"][name]
                pred_classes = torch.argmax(output["categorical"][name][mask], dim=-1)
                target_classes = targets[mask, info["index"]].long()
                self.cat_metrics[name]["f2"].update(pred_classes, target_classes)

    def compute(self) -> Dict[str, Any]:
        results = {}

        cont_results = self.cont_metrics.compute()
        continuous_metrics = {}
        for names in self.cont_names.values():
            for name in names:
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
