from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import batched, chain, repeat
from typing import Literal

import numpy as np
import polars as pl
import torch

from .osu import (
    OBJECT_TYPE_CIRCLE,
    OBJECT_TYPE_HOLD,
    OBJECT_TYPE_SLIDER,
    OBJECT_TYPE_SPINNER,
    RawBeatmap,
)

STAGE_WIDTH = 512
STAGE_HEIGHT = 384
PRE_START_MS = 200.0
BREAK_MS = 5000
MIN_OBJECTS = 10
SPAN_COUNT_BINS = 5

Gate = tuple[str, tuple[int, ...]]


@dataclass(frozen=True)
class Feature:
    name: str
    expr: pl.Expr = field(repr=False)
    standardize: bool = False
    cardinality: int | None = None
    gate: Gate | None = None


BEATMAP_SCHEMA = {
    "beatmap_id": pl.Int64,
    "hp_drain": pl.Float32,
    "cs": pl.Float32,
    "od": pl.Float32,
    "ar": pl.Float32,
    "slider_multiplier": pl.Float32,
    "slider_tick": pl.Float32,
    "difficulty_rating": pl.Float32,
}

HITOBJECT_SCHEMA = {
    "beatmap_id": pl.Int64,
    "object_index": pl.Int32,
    "x": pl.Int32,
    "y": pl.Int32,
    "time": pl.Int32,
    "object_type": pl.Int8,
    "is_new_combo": pl.Int8,
    "hit_sound": pl.Int32,
    "end_time": pl.Int32,
    "pixel_length": pl.Float32,
    "bpm": pl.Float32,
    "end_bpm": pl.Float32,
    "num_anchors": pl.Int32,
    "slider_repeats": pl.Int32,
    "hard_anchor_ratio": pl.Float32,
    "slider_path_valid": pl.Int8,
    "span_end_dx": pl.Float32,
    "span_end_dy": pl.Float32,
    "curve_residual_1_dx": pl.Float32,
    "curve_residual_1_dy": pl.Float32,
    "curve_residual_2_dx": pl.Float32,
    "curve_residual_2_dy": pl.Float32,
}

TIMING_COLUMNS = (
    "beatmap_id",
    "object_index",
    "time",
    "object_type",
    "end_time",
    "bpm",
    "end_bpm",
)


def beatmap_frames(
    beatmaps: Sequence[RawBeatmap], columns: Sequence[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    beatmap_frame = pl.DataFrame(
        {
            name: [getattr(beatmap, name) for beatmap in beatmaps]
            for name in BEATMAP_SCHEMA
        },
        schema=BEATMAP_SCHEMA,
    )
    series = []
    for name in columns:
        if name == "beatmap_id":
            values = chain.from_iterable(
                repeat(beatmap.beatmap_id, len(beatmap.hit_objects))
                for beatmap in beatmaps
            )
        else:
            values = chain.from_iterable(
                getattr(beatmap.hit_objects, name) for beatmap in beatmaps
            )
        series.append(pl.Series(name, list(values), dtype=HITOBJECT_SCHEMA[name]))
    return beatmap_frame, pl.DataFrame(series)


def per_map(expr: pl.Expr) -> pl.Expr:
    return expr.over("beatmap_id")


def previous(name: str) -> pl.Expr:
    return per_map(pl.col(name).shift(1))


def gated(mask: pl.Expr, value: pl.Expr | float) -> pl.Expr:
    return pl.when(mask).then(value).otherwise(0.0)


def centered(value: pl.Expr, extent: int) -> pl.Expr:
    center = extent / 2.0
    return ((value - center) / center).clip(-1.0, 1.0)


def beat_angle(duration_ms: pl.Expr) -> pl.Expr:
    beats = duration_ms * pl.col("bpm") / 60000.0
    return beats.clip(lower_bound=1e-6).log() * (2 * np.pi / np.log(2))


def duration_ms() -> pl.Expr:
    return (pl.col("end_time") - pl.col("time")).clip(lower_bound=0)


def bpm_is_valid(column: str) -> pl.Expr:
    bpm = pl.col(column)
    return bpm.is_finite() & (bpm > 0) & (bpm <= 1000)


def with_valid_timing(objects: pl.LazyFrame) -> pl.LazyFrame:
    valid_maps = (
        objects.group_by("beatmap_id")
        .agg(
            pl.len().alias("_objects"),
            (~(bpm_is_valid("bpm") & bpm_is_valid("end_bpm")))
            .fill_null(True)
            .any()
            .alias("_bad_bpm"),
        )
        .filter((pl.col("_objects") >= MIN_OBJECTS) & ~pl.col("_bad_bpm"))
        .select("beatmap_id")
    )
    return objects.join(valid_maps, on="beatmap_id", how="semi")


def in_play_order(objects: pl.LazyFrame, max_seq_len: int | None) -> pl.LazyFrame:
    objects = objects.sort("beatmap_id", "time", "object_index")
    if max_seq_len is not None:
        objects = objects.group_by("beatmap_id", maintain_order=True).head(max_seq_len)
    return objects


def with_onsets(frame: pl.LazyFrame) -> pl.LazyFrame:
    gap = pl.col("time") - pl.col("_prev_time")
    return frame.with_columns(previous("time").alias("_prev_time")).with_columns(
        (pl.col("time") - pl.col("_prev_time").fill_null(pl.col("time") - PRE_START_MS))
        .clip(0, BREAK_MS)
        .alias("_onset_ioi_ms"),
        pl.when(pl.col("_prev_time").is_null())
        .then(0)
        .when(gap >= BREAK_MS)
        .then(2)
        .otherwise(1)
        .cast(pl.Int32)
        .alias("onset_state"),
    )


def onset_features() -> tuple[Feature, ...]:
    ioi = pl.col("_onset_ioi_ms")
    return (
        Feature("log_onset_ioi_ms", ioi.log1p(), standardize=True),
        Feature("onset_rhythm_cos", beat_angle(ioi).cos()),
        Feature("onset_rhythm_sin", beat_angle(ioi).sin()),
    )


def timed_features(
    prefix: str, mask: pl.Expr, gate: Gate, rhythm: bool = True
) -> tuple[Feature, ...]:
    duration = duration_ms()
    log_duration = Feature(
        f"log_{prefix}_duration_ms",
        gated(mask, duration.log1p()),
        standardize=True,
        gate=gate,
    )
    if not rhythm:
        return (log_duration,)
    return (
        log_duration,
        Feature(
            f"{prefix}_rhythm_cos", gated(mask, beat_angle(duration).cos()), gate=gate
        ),
        Feature(
            f"{prefix}_rhythm_sin", gated(mask, beat_angle(duration).sin()), gate=gate
        ),
    )


def span_count() -> pl.Expr:
    return pl.col("slider_repeats").clip(lower_bound=0) + 1


def exit_position(axis: str, offset: str) -> pl.Expr:
    is_slider = pl.col("object_type") == OBJECT_TYPE_SLIDER
    is_spinner = pl.col("object_type") == OBJECT_TYPE_SPINNER
    ends_at_tail = span_count() % 2 == 1
    return (
        pl.when(is_slider & (pl.col("slider_path_valid") > 0))
        .then(pl.col(axis) + pl.when(ends_at_tail).then(pl.col(offset)).otherwise(0.0))
        .when(~is_slider & ~is_spinner)
        .then(pl.col(axis))
        .otherwise(None)
    )


def span_count_bin(gate: Gate) -> Feature:
    spans = span_count()
    return Feature(
        "span_count_bin",
        pl.when(pl.col("object_type") != OBJECT_TYPE_SLIDER)
        .then(0)
        .when(spans <= 3)
        .then(spans - 1)
        .when(spans % 2 == 0)
        .then(3)
        .otherwise(4)
        .cast(pl.Int32),
        cardinality=SPAN_COUNT_BINS,
        gate=gate,
    )


def object_gates(column: str, slider: int, spinner: int) -> tuple[Gate, Gate]:
    return (column, (slider,)), (column, (spinner,))


def pack_vectors(
    frame: pl.LazyFrame, features: Sequence[Feature]
) -> tuple[list[torch.Tensor], np.ndarray]:
    packed = frame.select(
        "beatmap_id",
        *(
            pl.when(feature.expr.cast(pl.Float16).is_finite())
            .then(feature.expr.cast(pl.Float16))
            .otherwise(0.0)
            .alias(feature.name)
            for feature in features
        ),
    ).collect(engine="streaming")
    if packed.is_empty():
        return [], np.empty(0, dtype=np.int64)
    ids = packed["beatmap_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    vectors = torch.from_numpy(packed.drop("beatmap_id").to_numpy(order="c"))
    lengths = np.diff(np.r_[starts, len(ids)])
    return list(torch.split(vectors, lengths.tolist())), ids[starts]


SLIDER = pl.col("object_type") == OBJECT_TYPE_SLIDER
SPINNER = pl.col("object_type") == OBJECT_TYPE_SPINNER


def std_objects(
    beatmaps: pl.LazyFrame, objects: pl.LazyFrame, max_seq_len: int | None
) -> pl.LazyFrame:
    on_stage = pl.col("x").is_between(0, STAGE_WIDTH) & pl.col("y").is_between(
        0, STAGE_HEIGHT
    )
    objects = in_play_order(with_valid_timing(objects.filter(on_stage)), max_seq_len)
    return with_onsets(
        objects.with_columns(
            exit_position("x", "span_end_dx").alias("_exit_x"),
            exit_position("y", "span_end_dy").alias("_exit_y"),
        )
    ).with_columns(
        previous("end_time").alias("_prev_end_time"),
        previous("object_type").alias("_prev_type"),
        previous("_exit_x").alias("_prev_exit_x"),
        previous("_exit_y").alias("_prev_exit_y"),
    )


def std_features() -> tuple[Feature, ...]:
    slider, spinner = object_gates(
        "object_type", OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER
    )
    is_break = pl.col("_prev_time").is_not_null() & (
        pl.col("time") - pl.col("_prev_time") >= BREAK_MS
    )
    inside_previous_slider = (pl.col("_prev_type") == OBJECT_TYPE_SLIDER) & (
        pl.col("time") < pl.col("_prev_end_time")
    )
    incoming = (
        pl.col("_prev_time").is_not_null()
        & pl.col("_prev_exit_x").is_not_null()
        & (pl.col("time") > pl.col("_prev_time"))
        & ~inside_previous_slider
        & ~is_break
    )
    jump_dx = pl.col("x") - pl.col("_prev_exit_x")
    jump_dy = pl.col("y") - pl.col("_prev_exit_y")
    jump = (jump_dx.pow(2) + jump_dy.pow(2)).sqrt()
    jump_has_direction = incoming & (jump > 0)
    span_dx, span_dy = pl.col("span_end_dx"), pl.col("span_end_dy")
    span_distance = (span_dx.pow(2) + span_dy.pow(2)).sqrt()
    span_has_direction = SLIDER & (span_distance > 0)
    anchors = pl.col("num_anchors").clip(0, 4096)
    red_anchors = (pl.col("hard_anchor_ratio") * anchors).round().clip(0, 4096)
    motion: Gate = ("incoming_motion_valid", (1,))

    def residual(name: str, extent: int) -> Feature:
        return Feature(
            name, gated(SLIDER, pl.col(name) / extent), standardize=True, gate=slider
        )

    return (
        Feature("norm_x", centered(pl.col("x"), STAGE_WIDTH)),
        Feature("norm_y", centered(pl.col("y"), STAGE_HEIGHT)),
        Feature(
            "log_jump_distance",
            gated(incoming, jump.log1p()),
            standardize=True,
            gate=motion,
        ),
        Feature(
            "jump_direction_cos", gated(jump_has_direction, jump_dx / jump), gate=motion
        ),
        Feature(
            "jump_direction_sin", gated(jump_has_direction, jump_dy / jump), gate=motion
        ),
        *onset_features(),
        *timed_features("span", SLIDER, slider),
        Feature(
            "log_span_length",
            gated(SLIDER, pl.col("pixel_length").clip(lower_bound=0).log1p()),
            standardize=True,
            gate=slider,
        ),
        Feature(
            "log_span_end_distance",
            gated(SLIDER, span_distance.log1p()),
            standardize=True,
            gate=slider,
        ),
        Feature(
            "span_end_direction_cos",
            gated(span_has_direction, span_dx / span_distance),
            gate=slider,
        ),
        Feature(
            "span_end_direction_sin",
            gated(span_has_direction, span_dy / span_distance),
            gate=slider,
        ),
        residual("curve_residual_1_dx", STAGE_WIDTH),
        residual("curve_residual_1_dy", STAGE_HEIGHT),
        residual("curve_residual_2_dx", STAGE_WIDTH),
        residual("curve_residual_2_dy", STAGE_HEIGHT),
        *timed_features("spinner", SPINNER, spinner),
        Feature("object_type", pl.col("object_type"), cardinality=3),
        Feature("is_new_combo", pl.col("is_new_combo"), cardinality=2),
        Feature(
            "incoming_motion_valid",
            pl.when(is_break)
            .then(2)
            .otherwise(incoming.fill_null(False).cast(pl.Int32)),
            cardinality=3,
        ),
        span_count_bin(slider),
        Feature(
            "log_gray_anchor_count",
            gated(SLIDER, anchors.log1p()),
            standardize=True,
            gate=slider,
        ),
        Feature(
            "red_anchor_bin",
            pl.when(SLIDER).then(red_anchors.clip(0, 3)).otherwise(0).cast(pl.Int32),
            cardinality=4,
            gate=slider,
        ),
    )


NOTE_DON, NOTE_KAT, NOTE_DRUMROLL, NOTE_SWELL = range(4)
HITSOUND_WHISTLE, HITSOUND_FINISH, HITSOUND_CLAP = 2, 4, 8


def taiko_objects(
    beatmaps: pl.LazyFrame, objects: pl.LazyFrame, max_seq_len: int | None
) -> pl.LazyFrame:
    return with_onsets(in_play_order(with_valid_timing(objects), max_seq_len))


def taiko_features() -> tuple[Feature, ...]:
    drumroll, swell = object_gates("note_type", NOTE_DRUMROLL, NOTE_SWELL)
    hit_sound = pl.col("hit_sound")
    is_kat = (hit_sound & (HITSOUND_WHISTLE | HITSOUND_CLAP)) != 0
    return (
        *onset_features(),
        *timed_features("drumroll", SLIDER, drumroll),
        *timed_features("swell", SPINNER, swell, rhythm=False),
        Feature(
            "note_type",
            pl.when(SPINNER)
            .then(NOTE_SWELL)
            .when(SLIDER)
            .then(NOTE_DRUMROLL)
            .when(is_kat)
            .then(NOTE_KAT)
            .otherwise(NOTE_DON)
            .cast(pl.Int32),
            cardinality=4,
        ),
        Feature(
            "is_big",
            (~SPINNER & ((hit_sound & HITSOUND_FINISH) != 0)).cast(pl.Int32),
            cardinality=2,
        ),
        Feature("onset_state", pl.col("onset_state"), cardinality=3),
    )


def catch_objects(
    beatmaps: pl.LazyFrame, objects: pl.LazyFrame, max_seq_len: int | None
) -> pl.LazyFrame:
    on_stage = pl.col("x").is_between(0, STAGE_WIDTH)
    objects = in_play_order(with_valid_timing(objects.filter(on_stage)), max_seq_len)
    has_previous_exit = pl.col("_prev_exit_x").is_not_null()
    after_previous = pl.col("time") >= pl.col("_prev_end_time")
    return (
        with_onsets(objects)
        .with_columns(exit_position("x", "span_end_dx").alias("_exit_x"))
        .with_columns(
            previous("_exit_x").alias("_prev_exit_x"),
            previous("end_time").alias("_prev_end_time"),
        )
        .with_columns(
            pl.when(pl.col("onset_state") == 2)
            .then(2)
            .when((pl.col("onset_state") == 1) & has_previous_exit & after_previous)
            .then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias("incoming_state"),
        )
    )


def catch_features() -> tuple[Feature, ...]:
    slider, spinner = object_gates(
        "object_type", OBJECT_TYPE_SLIDER, OBJECT_TYPE_SPINNER
    )
    moving = pl.col("incoming_state") == 1
    movement = pl.col("x") - pl.col("_prev_exit_x")
    has_path = SLIDER & (pl.col("slider_path_valid") > 0)
    movement_gate: Gate = ("incoming_state", (1,))

    def stream_offset(name: str, column: str, standardize: bool) -> Feature:
        return Feature(
            name,
            gated(has_path, pl.col(column) / STAGE_WIDTH),
            standardize=standardize,
            gate=slider,
        )

    return (
        Feature("norm_x", centered(pl.col("x"), STAGE_WIDTH)),
        Feature(
            "log_movement_distance",
            gated(moving, movement.abs().log1p()),
            standardize=True,
            gate=movement_gate,
        ),
        Feature(
            "movement_direction",
            pl.when(moving).then(movement.sign()).otherwise(0).cast(pl.Float32),
            gate=movement_gate,
        ),
        *onset_features(),
        *timed_features("stream", SLIDER, slider),
        stream_offset("stream_end_dx", "span_end_dx", standardize=False),
        stream_offset("stream_residual_1_dx", "curve_residual_1_dx", standardize=True),
        stream_offset("stream_residual_2_dx", "curve_residual_2_dx", standardize=True),
        *timed_features("banana", SPINNER, spinner, rhythm=False),
        Feature("object_type", pl.col("object_type"), cardinality=3),
        span_count_bin(slider),
        Feature("incoming_state", pl.col("incoming_state"), cardinality=3),
    )


MAX_KEYS = 10
HAND_SLOTS = 5
SLOT_EMPTY, SLOT_TAP, SLOT_HOLD, SLOT_HELD, SLOT_RELEASE = range(5)
SLOT_STATES = 5
SLOT_FIELDS = (
    *(f"left_{finger}" for finger in range(HAND_SLOTS, 0, -1)),
    "center",
    *(f"right_{finger}" for finger in range(1, HAND_SLOTS + 1)),
)


def mania_slot(column: pl.Expr, keys: pl.Expr) -> pl.Expr:
    half = keys // 2
    return (
        pl.when(column < half)
        .then(HAND_SLOTS - (half - column))
        .when(column >= keys - half)
        .then(HAND_SLOTS + column - (keys - half) + 1)
        .otherwise(HAND_SLOTS)
    )


def mania_notes(beatmaps: pl.LazyFrame, objects: pl.LazyFrame) -> pl.LazyFrame:
    keys = (
        beatmaps.select("beatmap_id", pl.col("cs").round().alias("_keys"))
        .filter(pl.col("_keys").is_between(1, MAX_KEYS))
        .with_columns(pl.col("_keys").cast(pl.Int32))
    )
    notes = objects.filter(
        pl.col("object_type").is_in([OBJECT_TYPE_CIRCLE, OBJECT_TYPE_HOLD])
    ).join(keys, on="beatmap_id", how="inner")
    is_hold = (pl.col("object_type") == OBJECT_TYPE_HOLD) & (
        pl.col("end_time") > pl.col("time")
    )
    return with_valid_timing(notes).with_columns(
        (pl.col("x") * pl.col("_keys") // STAGE_WIDTH)
        .clip(0, pl.col("_keys") - 1)
        .cast(pl.Int32)
        .alias("column"),
        is_hold.alias("_is_hold"),
    )


def mania_events(notes: pl.LazyFrame, max_seq_len: int | None) -> pl.LazyFrame:
    presses = notes.select(
        "beatmap_id", "time", "bpm", "_keys", pl.lit(0).alias("_release")
    )
    releases = notes.filter(pl.col("_is_hold")).select(
        "beatmap_id",
        pl.col("end_time").alias("time"),
        pl.col("end_bpm").alias("bpm"),
        "_keys",
        pl.lit(1).alias("_release"),
    )
    events = (
        pl.concat([presses, releases])
        .group_by("beatmap_id", "time")
        .agg(
            pl.col("bpm").sort_by("_release", "bpm").first(),
            pl.col("_keys").first(),
        )
        .sort("beatmap_id", "time")
    )
    if max_seq_len is not None:
        events = events.group_by("beatmap_id", maintain_order=True).head(max_seq_len)
    return events


def mania_slot_states(notes: pl.LazyFrame, events: pl.LazyFrame) -> pl.LazyFrame:
    slot_key = ["beatmap_id", "time", "column"]
    heads = notes.group_by(slot_key).agg(
        pl.when(pl.col("_is_hold").any())
        .then(SLOT_HOLD)
        .otherwise(SLOT_TAP)
        .alias("_head")
    )
    holds = notes.filter(pl.col("_is_hold")).select(
        "beatmap_id",
        "column",
        pl.col("time").alias("_hold_start"),
        pl.col("end_time").alias("_hold_end"),
    )
    releases = holds.select(
        "beatmap_id",
        "column",
        pl.col("_hold_end").alias("time"),
        pl.lit(True).alias("_release"),
    ).unique()
    active_holds = (
        holds.group_by("beatmap_id", "column", "_hold_start")
        .agg(pl.col("_hold_end").max())
        .sort("beatmap_id", "column", "_hold_start")
        .with_columns(pl.col("_hold_end").cum_max().over("beatmap_id", "column"))
    )
    grid = (
        events.select(
            "beatmap_id",
            "time",
            "_keys",
            pl.int_ranges(0, pl.col("_keys"), dtype=pl.Int32).alias("column"),
        )
        .explode("column")
        .sort("beatmap_id", "column", "time")
        .join_asof(
            active_holds,
            left_on="time",
            right_on="_hold_start",
            by=["beatmap_id", "column"],
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .join(heads, on=slot_key, how="left")
        .join(releases, on=slot_key, how="left")
        .with_columns(
            mania_slot(pl.col("column"), pl.col("_keys")).alias("_slot"),
            pl.when(pl.col("_head").is_not_null())
            .then(pl.col("_head"))
            .when(pl.col("_release").fill_null(False))
            .then(SLOT_RELEASE)
            .when(pl.col("_hold_end") > pl.col("time"))
            .then(SLOT_HELD)
            .otherwise(SLOT_EMPTY)
            .alias("_state"),
        )
    )
    slot_digit = pl.col("_state").cast(pl.Int64) * pl.lit(
        SLOT_STATES, dtype=pl.Int64
    ).pow(pl.col("_slot"))
    return (
        grid.group_by("beatmap_id", "time")
        .agg(slot_digit.sum().alias("_packed"))
        .select(
            "beatmap_id",
            "time",
            *(
                (pl.col("_packed") // SLOT_STATES**index % SLOT_STATES).alias(name)
                for index, name in enumerate(SLOT_FIELDS)
            ),
        )
    )


def mania_objects(
    beatmaps: pl.LazyFrame, objects: pl.LazyFrame, max_seq_len: int | None
) -> pl.LazyFrame:
    notes = mania_notes(beatmaps, objects)
    events = mania_events(notes, max_seq_len)
    slots = mania_slot_states(notes, events)
    return with_onsets(
        events.join(slots, on=["beatmap_id", "time"], how="left").sort(
            "beatmap_id", "time"
        )
    )


def mania_features() -> tuple[Feature, ...]:
    return (
        *onset_features(),
        Feature("onset_state", pl.col("onset_state"), cardinality=3),
        Feature("key_count", pl.col("_keys") - 1, cardinality=MAX_KEYS),
        *(Feature(name, pl.col(name), cardinality=SLOT_STATES) for name in SLOT_FIELDS),
    )


Prepare = Callable[[pl.LazyFrame, pl.LazyFrame, int | None], pl.LazyFrame]


@dataclass(frozen=True)
class Mode:
    name: str
    mode_int: int
    label: str
    columns: tuple[str, ...]
    prepare: Prepare = field(repr=False)
    features: tuple[Feature, ...] = field(repr=False)
    groups: tuple[tuple[str, ...], ...]
    right_border: tuple[str, ...]
    strains: dict[str, float]
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

    @cached_property
    def gates(self) -> dict[str, Gate]:
        return {f.name: f.gate for f in self.features if f.gate is not None}

    def build(
        self,
        beatmaps: pl.DataFrame,
        objects: pl.DataFrame,
        max_seq_len: int | None = None,
    ) -> tuple[list[torch.Tensor], np.ndarray]:
        frame = self.prepare(beatmaps.lazy(), objects.lazy(), max_seq_len)
        return pack_vectors(frame, self.features)

    def gate_mask(self, x: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
        columns = []
        for name in names:
            gate = self.gates.get(name)
            if gate is None:
                columns.append(
                    torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
                )
                continue
            column, accepted = gate
            values = x[..., self.index[column]].round().long()
            columns.append(torch.isin(values, torch.tensor(accepted, device=x.device)))
        return torch.stack(columns, dim=-1)

    def fit_stats(
        self, train_data: Iterable[torch.Tensor], epsilon: float = 1e-8
    ) -> torch.Tensor:
        names = [f.name for f in self.features if f.standardize]
        columns = [self.index[name] for name in names]
        counts = torch.zeros(len(names), dtype=torch.float64)
        totals = torch.zeros(len(names), dtype=torch.float64)
        totals_sq = torch.zeros(len(names), dtype=torch.float64)
        for chunk in batched(train_data, 256):
            vectors = torch.cat(chunk).double()
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


ONSETS = ("log_onset_ioi_ms", "onset_rhythm_cos", "onset_rhythm_sin")
JUMPS = ("log_jump_distance", "jump_direction_cos", "jump_direction_sin")

ModeName = Literal["std", "taiko", "catch", "mania"]
MODES = {
    "std": Mode(
        name="std",
        mode_int=0,
        label="osu",
        columns=(
            *TIMING_COLUMNS,
            "x",
            "y",
            "is_new_combo",
            "pixel_length",
            "slider_repeats",
            "slider_path_valid",
            "span_end_dx",
            "span_end_dy",
            "curve_residual_1_dx",
            "curve_residual_1_dy",
            "curve_residual_2_dx",
            "curve_residual_2_dy",
            "num_anchors",
            "hard_anchor_ratio",
        ),
        prepare=std_objects,
        features=std_features(),
        groups=(
            ("norm_x", "norm_y"),
            JUMPS,
            ONSETS,
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
        right_border=(*JUMPS, *ONSETS, "incoming_motion_valid"),
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
        columns=(*TIMING_COLUMNS, "hit_sound"),
        prepare=taiko_objects,
        features=taiko_features(),
        groups=(
            ONSETS,
            ("log_drumroll_duration_ms", "drumroll_rhythm_cos", "drumroll_rhythm_sin"),
            ("log_swell_duration_ms",),
        ),
        right_border=(*ONSETS, "onset_state"),
        strains={"stars": 1.0},
    ),
    "catch": Mode(
        name="catch",
        mode_int=2,
        label="fruits",
        columns=(
            *TIMING_COLUMNS,
            "x",
            "slider_repeats",
            "slider_path_valid",
            "span_end_dx",
            "curve_residual_1_dx",
            "curve_residual_2_dx",
        ),
        prepare=catch_objects,
        features=catch_features(),
        groups=(
            ("norm_x",),
            ("log_movement_distance", "movement_direction"),
            ONSETS,
            ("log_stream_duration_ms", "stream_rhythm_cos", "stream_rhythm_sin"),
            ("stream_end_dx", "stream_residual_1_dx", "stream_residual_2_dx"),
            ("log_banana_duration_ms",),
        ),
        right_border=(
            "log_movement_distance",
            "movement_direction",
            *ONSETS,
            "incoming_state",
        ),
        strains={"stars": 1.0},
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
        columns=(*TIMING_COLUMNS, "x"),
        prepare=mania_objects,
        features=mania_features(),
        groups=(ONSETS,),
        right_border=(*ONSETS, "onset_state"),
        strains={"stars": 1.0},
        mirror=SLOT_FIELDS,
    ),
}
MODE_NAMES = {mode.mode_int: name for name, mode in MODES.items()}
