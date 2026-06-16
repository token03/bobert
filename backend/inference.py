from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf

from core.data.batch import pack_batch
from core.data.beatmap import MAP_FEATURE_ATTRIBUTES
from core.data.feature import build_feature_tensors, calculate_drain_times
from core.data.normalizer import BeatmapNormalizer
from core.data.parser import RawBeatmap, parse_osu_file
from core.model.bobert import BobertForAlignment
from core.model.checkpoint import configure_from_checkpoint, normalize_checkpoint_state


MIN_OBJECTS_PER_MAP = 1


class CpuInferencer:
    def __init__(self, config_path: Path, model_path: Path):
        self.config_path = config_path
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.config: Any | None = None
        self.model: BobertForAlignment | None = None
        self.normalizer: BeatmapNormalizer | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.config_path.exists():
            raise FileNotFoundError(f"config not found: {self.config_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")

        base_config_path = self.config_path.with_name("config.yaml")
        if self.config_path.name != "config.yaml" and base_config_path.exists():
            config = OmegaConf.merge(
                OmegaConf.load(base_config_path), OmegaConf.load(self.config_path)
            )
        else:
            config = OmegaConf.load(self.config_path)
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        state = normalize_checkpoint_state(checkpoint.get("state_dict", checkpoint))
        config = configure_from_checkpoint(config, checkpoint, state, alignment=True)
        config.components.compile_model = False
        config.components.compile_dynamic = False
        config.components.activation_checkpointing = False
        config.alignment.query_pool_use_flash = False
        config.alignment.precision = 32

        model = BobertForAlignment.from_config(config, self.device)
        model_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        model.load_state_dict(compatible_state, strict=False)
        model.to(self.device).float().eval()

        self.config = config
        self.model = model
        self.normalizer = BeatmapNormalizer(
            vector_stats=checkpoint["vector_stats"],
            attribute_stats=checkpoint.get("attribute_stats", {}),
        )

    def embed_osu_bytes(self, content: bytes) -> np.ndarray:
        self.load()
        assert self.config is not None
        assert self.model is not None
        assert self.normalizer is not None

        with tempfile.NamedTemporaryFile(suffix=".osu") as tmp:
            tmp.write(content)
            tmp.flush()
            vectors, raw_map_features = _beatmap_inputs_from_osu(
                Path(tmp.name), self.config.data.max_seq_len
            )

        vectors = self.normalizer.normalize_vectors(vectors)
        map_features = torch.tensor(
            [
                self.normalizer.normalize_attribute(name, raw_map_features.get(name, 0.0))
                for name in MAP_FEATURE_ATTRIBUTES
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        vector_dim = vectors.shape[1]
        packed, cu_seqlens, max_seqlen = pack_batch(
            [vectors], self.config.data.max_seq_len, vector_dim
        )

        with torch.inference_mode():
            embedding = self.model.embed_packed(
                packed.to(self.device),
                cu_seqlens.to(self.device),
                max_seqlen,
                map_features.to(self.device),
            )

        vector = embedding.float().cpu().numpy()[0]
        norm = np.linalg.norm(vector)
        return (vector / max(norm, 1e-12)).astype(np.float32)


def _beatmap_inputs_from_osu(path: Path, max_seq_len: int):
    raw_beatmap = parse_osu_file(str(path))
    if not _validate_beatmap(raw_beatmap):
        raise ValueError(f"could not parse a valid beatmap from {path}")

    assert raw_beatmap is not None
    beatmaps_df = pl.DataFrame([_extract_beatmap_record(raw_beatmap)])
    hitobjects_df = pl.DataFrame(_extract_hitobject_records(raw_beatmap))
    drain_times = calculate_drain_times(beatmaps_df, hitobjects_df)
    beatmaps_df = beatmaps_df.join(drain_times, on="beatmap_id", how="left")
    if "drain_time" not in beatmaps_df.columns:
        beatmaps_df = beatmaps_df.with_columns(
            pl.lit(0.0).cast(pl.Float32).alias("drain_time")
        )
    else:
        beatmaps_df = beatmaps_df.with_columns(
            pl.col("drain_time").fill_null(0.0).cast(pl.Float32)
        )
    vectors, _ids, _ = build_feature_tensors(beatmaps_df, hitobjects_df)
    if not vectors:
        raise ValueError(f"could not engineer hitobject features for {path}")

    expanded_count = vectors[0].shape[0]
    truncate_len = min(expanded_count, max_seq_len or expanded_count)
    row = beatmaps_df.row(0, named=True)
    map_features = {
        name: float(row.get(name, 0.0) or 0.0) for name in MAP_FEATURE_ATTRIBUTES
    }
    return vectors[0][:truncate_len], map_features


def _validate_beatmap(beatmap: RawBeatmap | None) -> bool:
    return (
        beatmap is not None
        and len(beatmap.hit_objects) > 0
        and MIN_OBJECTS_PER_MAP < len(beatmap.hit_objects)
    )


def _extract_beatmap_record(beatmap: RawBeatmap) -> dict[str, Any]:
    return {
        "beatmap_id": beatmap.beatmap_id,
        "category": beatmap.category,
        "hp_drain": beatmap.hp_drain,
        "cs": beatmap.cs,
        "od": beatmap.od,
        "ar": beatmap.ar,
        "slider_multiplier": beatmap.slider_multiplier,
        "slider_tick": beatmap.slider_tick,
        "difficulty_rating": beatmap.difficulty_rating,
    }


def _extract_hitobject_records(beatmap: RawBeatmap) -> list[dict[str, Any]]:
    records = []
    for ho in beatmap.hit_objects:
        records.append(
            {
                "beatmap_id": beatmap.beatmap_id,
                "category": beatmap.category,
                "x": ho.x,
                "y": ho.y,
                "time": ho.time,
                "object_type": ho.object_type,
                "is_new_combo": ho.is_new_combo,
                "hit_sound": ho.hit_sound,
                "end_time": ho.end_time,
                "pixel_length": ho.pixel_length or 0.0,
                "bpm": ho.bpm,
                "curve_type_char": ho.curve_type or "",
                "num_anchors": ho.num_anchors,
                "kiai_time": ho.kiai_time,
                "slider_repeats": (ho.slides - 1) if ho.slides is not None else 0,
                "hard_anchor_ratio": ho.hard_anchor_ratio,
                "slider_end_x": ho.slider_end_x,
                "slider_end_y": ho.slider_end_y,
            }
        )
    return records
