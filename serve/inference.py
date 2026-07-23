from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import polars as pl
import torch

from core.features import VectorStats, build_feature_tensors, normalize
from core.model import BobertEncoder
from core.osu import (
    RawBeatmap,
    extract_beatmap_record,
    extract_hitobject_records,
    parse_osu_file,
)


MIN_OBJECTS_PER_MAP = 1


class CpuInferencer:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.model: BobertEncoder | None = None
        self.vector_stats: VectorStats | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")

        model, vector_stats = BobertEncoder.from_pretrained(
            self.model_path, self.device
        )
        model.to(self.device).float().eval()

        self.model = model
        self.vector_stats = vector_stats

    def embed_osu_bytes(self, content: bytes) -> np.ndarray:
        self.load()
        assert self.model is not None
        assert self.vector_stats is not None

        with tempfile.NamedTemporaryFile(suffix=".osu") as tmp:
            tmp.write(content)
            tmp.flush()
            vectors = _beatmap_inputs_from_osu(Path(tmp.name), self.model.max_seq_len)

        vectors = normalize(vectors, self.vector_stats)
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
    beatmaps_df = pl.DataFrame([extract_beatmap_record(raw_beatmap)])
    hitobjects_df = pl.DataFrame(extract_hitobject_records(raw_beatmap))
    vectors, _ids = build_feature_tensors(
        beatmaps_df,
        hitobjects_df,
        max_seq_len=max_seq_len,
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
