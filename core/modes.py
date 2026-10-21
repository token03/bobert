from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from types import ModuleType

import torch

from . import catch, features, mania, osu, taiko
from .features import Feature, beatmap_frames
from .osu import OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER, RawBeatmap

ONSET_FEATURES = ("log_onset_ioi_ms", "onset_rhythm_cos", "onset_rhythm_sin")


@dataclass(frozen=True)
class Mode:
    name: str
    mode_int: int
    label: str
    features: tuple[Feature, ...]
    columns: tuple[str, ...]
    build: Callable
    sources: tuple[ModuleType, ...]
    groups: tuple[tuple[str, ...], ...]
    right_border: tuple[str, ...]
    strains: dict[str, float]
    gates: dict[str, tuple[str, tuple[int, ...]]] = field(default_factory=dict)
    flips: tuple[tuple[str, ...], ...] = ()
    mirror: tuple[str, ...] = ()

    @cached_property
    def field_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @cached_property
    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.field_names)}

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
        for names in self.flips:
            if bool(torch.rand(()) < 0.5):
                vector = vector.clone()
                vector[:, [self.index[name] for name in names]] *= -1
        if self.mirror and bool(torch.rand(()) < 0.5):
            columns = [self.index[name] for name in self.mirror]
            vector = vector.clone()
            vector[:, columns] = vector[:, columns[::-1]]
        return vector

    def beatmap_tensor(
        self, beatmap: RawBeatmap, max_seq_len: int | None = None
    ) -> torch.Tensor:
        if beatmap.mode != self.mode_int:
            raise ValueError(f"only {self.name} beatmaps are supported")
        vectors, _ = self.build(*beatmap_frames([beatmap], self.columns), max_seq_len)
        if not vectors:
            raise ValueError("could not engineer hitobject features")
        return vectors[0]


def conditional_gates(
    features: Sequence[Feature], gate: str, slider: int, spinner: int
) -> dict[str, tuple[str, tuple[int, ...]]]:
    values = {"slider": (slider,), "spinner": (spinner,)}
    return {
        f.name: (gate, values[f.conditional])
        for f in features
        if f.conditional is not None
    }


JUMP_FEATURES = ("log_jump_distance", "jump_direction_cos", "jump_direction_sin")

MODES = {
    "std": Mode(
        name="std",
        mode_int=0,
        label="osu",
        features=features.FEATURES,
        columns=features.HITOBJECT_COLUMNS,
        build=features.build_feature_tensors,
        sources=(features, osu),
        groups=(
            ("norm_x", "norm_y"),
            JUMP_FEATURES,
            ONSET_FEATURES,
            ("log_span_duration_ms", "span_rhythm_cos", "span_rhythm_sin"),
            ("log_spinner_duration_ms", "spinner_rhythm_cos", "spinner_rhythm_sin"),
            ("log_span_length",),
            (
                "log_span_end_distance",
                "span_end_direction_cos",
                "span_end_direction_sin",
            ),
            ("curve_residual_1_dx", "curve_residual_1_dy"),
            ("curve_residual_2_dx", "curve_residual_2_dy"),
            ("log_gray_anchor_count",),
        ),
        right_border=(*JUMP_FEATURES, *ONSET_FEATURES, "incoming_motion_valid"),
        strains={
            "aim": 1.0,
            "speed": 1.0,
            "slider": 0.25,
            "snap": 0.25,
            "flow": 0.25,
            "agility": 0.25,
            "tap": 0.5,
            "rhythm": 0.5,
        },
        gates={
            **conditional_gates(
                features.FEATURES,
                "object_type",
                slider=OBJECT_TYPE_SLIDER,
                spinner=OBJECT_TYPE_SPINNER,
            ),
            **{name: ("incoming_motion_valid", (1,)) for name in JUMP_FEATURES},
        },
        flips=(
            (
                "norm_x",
                "jump_direction_cos",
                "span_end_direction_cos",
                "curve_residual_1_dx",
                "curve_residual_2_dx",
            ),
            (
                "norm_y",
                "jump_direction_sin",
                "span_end_direction_sin",
                "curve_residual_1_dy",
                "curve_residual_2_dy",
            ),
        ),
    ),
    "taiko": Mode(
        name="taiko",
        mode_int=1,
        label="taiko",
        features=taiko.FEATURES,
        columns=taiko.HITOBJECT_COLUMNS,
        build=taiko.build_feature_tensors,
        sources=(taiko, features, osu),
        groups=(
            ONSET_FEATURES,
            ("log_drumroll_duration_ms", "drumroll_rhythm_cos", "drumroll_rhythm_sin"),
            ("log_swell_duration_ms",),
        ),
        right_border=(*ONSET_FEATURES, "onset_state"),
        strains={"stars": 1.0},
        gates=conditional_gates(
            taiko.FEATURES,
            "note_type",
            slider=taiko.NOTE_DRUMROLL,
            spinner=taiko.NOTE_SWELL,
        ),
    ),
    "catch": Mode(
        name="catch",
        mode_int=2,
        label="fruits",
        features=catch.FEATURES,
        columns=catch.HITOBJECT_COLUMNS,
        build=catch.build_feature_tensors,
        sources=(catch, features, osu),
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
        strains={"stars": 1.0},
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
        flips=(
            (
                "norm_x",
                "movement_direction",
                "stream_end_dx",
                "stream_residual_1_dx",
                "stream_residual_2_dx",
            ),
        ),
    ),
    "mania": Mode(
        name="mania",
        mode_int=3,
        label="mania",
        features=mania.FEATURES,
        columns=mania.HITOBJECT_COLUMNS,
        build=mania.build_feature_tensors,
        sources=(mania, features, osu),
        groups=(ONSET_FEATURES,),
        right_border=(*ONSET_FEATURES, "onset_state"),
        strains={"stars": 1.0},
        mirror=mania.SLOT_FIELDS,
    ),
}
MODE_NAMES = {mode.mode_int: name for name, mode in MODES.items()}
