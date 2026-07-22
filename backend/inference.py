from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from core.data.feature import build_feature_tensors
from core.data.normalizer import BeatmapNormalizer
from core.data.parser import RawBeatmap, parse_osu_file
from core.model.bobert import BobertEncoder


MIN_OBJECTS_PER_MAP = 1


class CpuInferencer:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.model: BobertEncoder | None = None
        self.normalizer: BeatmapNormalizer | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")

        model, normalizer = BobertEncoder.from_pretrained(self.model_path, self.device)
        model.to(self.device).float().eval()

        self.model = model
        self.normalizer = normalizer

    def embed_osu_bytes(self, content: bytes) -> np.ndarray:
        self.load()
        assert self.model is not None
        assert self.normalizer is not None

        with tempfile.NamedTemporaryFile(suffix=".osu") as tmp:
            tmp.write(content)
            tmp.flush()
            vectors = _beatmap_inputs_from_osu(Path(tmp.name), self.model.max_seq_len)

        vectors = self.normalizer.normalize_vectors(vectors)
        packed_vectors = vectors[: self.model.max_seq_len].contiguous()
        max_seqlen = packed_vectors.shape[0]
        cu_seqlens = torch.tensor([0, max_seqlen], dtype=torch.int32)

        with torch.inference_mode():
            embedding = self.model.embed_packed(
                packed_vectors.to(self.device),
                cu_seqlens.to(self.device),
                max_seqlen,
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
    vectors, _ids, _ = build_feature_tensors(
        beatmaps_df,
        hitobjects_df,
        max_seq_len=max_seq_len,
        return_original_counts=False,
    )
    if not vectors:
        raise ValueError(f"could not engineer hitobject features for {path}")

    expanded_count = vectors[0].shape[0]
    truncate_len = min(expanded_count, max_seq_len)
    return vectors[0][:truncate_len]


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
                "object_index": ho.object_index,
                "x": ho.x,
                "y": ho.y,
                "time": ho.time,
                "object_type": ho.object_type,
                "is_new_combo": ho.is_new_combo,
                "hit_sound": ho.hit_sound,
                "end_time": ho.end_time,
                "pixel_length": ho.pixel_length or 0.0,
                "bpm": ho.bpm,
                "timing_origin": ho.timing_origin,
                "end_bpm": ho.end_bpm,
                "end_timing_origin": ho.end_timing_origin,
                "curve_type_char": ho.curve_type or "",
                "num_anchors": ho.num_anchors,
                "kiai_time": ho.kiai_time,
                "slider_repeats": (ho.slides - 1) if ho.slides is not None else 0,
                "hard_anchor_ratio": ho.hard_anchor_ratio,
                "slider_end_x": ho.slider_end_x,
                "slider_end_y": ho.slider_end_y,
                "slider_path_valid": ho.slider_path_valid,
                "span_end_dx": ho.span_end_dx,
                "span_end_dy": ho.span_end_dy,
                "curve_residual_1_dx": ho.curve_residual_1_dx,
                "curve_residual_1_dy": ho.curve_residual_1_dy,
                "curve_residual_2_dx": ho.curve_residual_2_dx,
                "curve_residual_2_dy": ho.curve_residual_2_dy,
            }
        )
    return records
