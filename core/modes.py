from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import torch

from . import catch, mania, taiko
from .features import Feature
from .osu import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER

ONSET_FEATURES = ("log_onset_ioi_ms", "onset_rhythm_cos", "onset_rhythm_sin")


@dataclass(frozen=True)
class ModeSpec:
    features: tuple[Feature, ...]
    groups: tuple[tuple[str, ...], ...]
    right_border: tuple[str, ...]
    gates: dict[str, tuple[str, tuple[int, ...]]] = field(default_factory=dict)
    negate: tuple[str, ...] = ()
    mirror: tuple[str, ...] = ()

    @cached_property
    def index(self) -> dict[str, int]:
        return {feature.name: i for i, feature in enumerate(self.features)}

    @cached_property
    def continuous(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features if f.cardinality is None)

    @cached_property
    def categorical(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (f.name, f.cardinality) for f in self.features if f.cardinality is not None
        )

    def gate_mask(self, x: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
        columns = []
        for name in names:
            gate = self.gates.get(name)
            if gate is None:
                columns.append(
                    torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
                )
                continue
            values = x[..., self.index[gate[0]]].round().long()
            columns.append(torch.isin(values, torch.tensor(gate[1], device=x.device)))
        return torch.stack(columns, dim=-1)

    def fit_stats(
        self, train_data: Sequence[torch.Tensor], epsilon: float = 1e-8
    ) -> torch.Tensor:
        names = [f.name for f in self.features if f.standardize]
        columns = [self.index[name] for name in names]
        counts = torch.zeros(len(names), dtype=torch.float64)
        totals = torch.zeros(len(names), dtype=torch.float64)
        totals_sq = torch.zeros(len(names), dtype=torch.float64)
        for start in range(0, len(train_data), 256):
            vectors = torch.cat(tuple(train_data[start : start + 256])).double()
            active = self.gate_mask(vectors, names)
            values = vectors[:, columns] * active
            counts += active.sum(dim=0)
            totals += values.sum(dim=0)
            totals_sq += values.square().sum(dim=0)
        mean = totals / counts.clamp_min(1)
        variance = (totals_sq - totals.square() / counts.clamp_min(1)) / (
            counts - 1
        ).clamp_min(1)
        stats = torch.stack(
            (torch.zeros(len(self.features)), torch.ones(len(self.features)))
        )
        stats[0, columns] = mean.float()
        stats[1, columns] = variance.clamp_min(0.0).sqrt().clamp_min(epsilon).float()
        return stats

    def augment(self, vector: torch.Tensor) -> torch.Tensor:
        if self.negate and bool(torch.rand(()) < 0.5):
            vector = vector.clone()
            vector[:, [self.index[name] for name in self.negate]] *= -1
        if self.mirror and bool(torch.rand(()) < 0.5):
            columns = [self.index[name] for name in self.mirror]
            vector = vector.clone()
            vector[:, columns] = vector[:, columns[::-1]]
        return vector


def conditional_gates(
    features: Sequence[Feature], gate: str, slider: int, spinner: int
) -> dict[str, tuple[str, tuple[int, ...]]]:
    values = {"slider": (slider,), "spinner": (spinner,)}
    return {
        f.name: (gate, values[f.conditional])
        for f in features
        if f.conditional is not None
    }


MODE_SPECS = {
    "taiko": ModeSpec(
        features=taiko.FEATURES,
        groups=(
            ONSET_FEATURES,
            ("log_drumroll_duration_ms", "drumroll_rhythm_cos", "drumroll_rhythm_sin"),
            ("log_swell_duration_ms",),
        ),
        right_border=(*ONSET_FEATURES, "onset_state"),
        gates=conditional_gates(
            taiko.FEATURES,
            "note_type",
            slider=taiko.NOTE_DRUMROLL,
            spinner=taiko.NOTE_SWELL,
        ),
    ),
    "catch": ModeSpec(
        features=catch.FEATURES,
        groups=(
            ("norm_x",),
            ("log_movement_distance", "movement_direction"),
            ONSET_FEATURES,
            ("log_stream_duration_ms", "stream_rhythm_cos", "stream_rhythm_sin"),
            ("stream_end_dx", "stream_residual_1_dx", "stream_residual_2_dx"),
            ("log_banana_duration_ms",),
        ),
        right_border=(
            "log_movement_distance",
            "movement_direction",
            *ONSET_FEATURES,
            "incoming_state",
        ),
        gates={
            **conditional_gates(
                catch.FEATURES,
                "object_type",
                slider=OBJECT_TYPE_SLIDER,
                spinner=OBJECT_TYPE_SPINNER,
            ),
            "log_movement_distance": ("incoming_state", (1,)),
            "movement_direction": ("incoming_state", (1,)),
        },
        negate=(
            "norm_x",
            "movement_direction",
            "stream_end_dx",
            "stream_residual_1_dx",
            "stream_residual_2_dx",
        ),
    ),
    "mania": ModeSpec(
        features=mania.FEATURES,
        groups=(ONSET_FEATURES,),
        right_border=(*ONSET_FEATURES, "onset_state"),
        mirror=mania.SLOT_FIELDS,
    ),
}
