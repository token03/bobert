from __future__ import annotations

import re
import time
from math import isnan
from pathlib import Path

import numpy as np
import polars as pl
import requests
from core.data.beatmap import MAP_FEATURE_ATTRIBUTES
from scripts.common.osu import (
    API_TIERS,
    DOWNLOAD_HEADERS,
    get_sharded_path,
    is_valid_osu_file,
)
from scripts.common.paths import PROJECT_ROOT


DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings.parquet"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "beatmaps.parquet"
DEFAULT_BEATMAPS_DIR = PROJECT_ROOT / "data" / "beatmaps"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def extract_beatmap_id(raw_input: str) -> int:
    text = raw_input.strip()
    if not text:
        raise ValueError("empty input")

    if "osu.ppy.sh" in text:
        hash_match = re.search(
            r"beatmapsets/\d+#(?:osu|taiko|fruits|mania)/(\d+)", text
        )
        if hash_match:
            return int(hash_match.group(1))
        beatmaps_match = re.search(r"/(?:beatmaps|b)/(\d+)", text)
        if beatmaps_match:
            return int(beatmaps_match.group(1))

    id_match = re.search(r"(\d+)\s*$", text)
    if id_match:
        return int(id_match.group(1))

    raise ValueError(f"could not find a beatmap id in: {raw_input}")


def extract_beatmapset_id(raw_input: str) -> int | None:
    match = re.search(r"beatmapsets/(\d+)", raw_input.strip())
    return int(match.group(1)) if match else None


def load_embeddings(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Embeddings parquet not found: {path}")

    df = pl.read_parquet(path)
    if "beatmap_id" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"Expected beatmap_id and embedding columns in {path}")

    beatmap_ids = df["beatmap_id"].to_numpy().astype(np.int64)
    embeddings = np.asarray(df["embedding"].to_list(), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    id_to_index = {int(beatmap_id): idx for idx, beatmap_id in enumerate(beatmap_ids)}
    return beatmap_ids, embeddings, id_to_index


def load_metadata(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()

    wanted = [
        "id",
        "beatmapset_id",
        "artist",
        "title",
        "creator",
        "version",
        "status",
        "ranked",
        "difficulty_rating",
        "bpm",
        "total_length",
        "hit_length",
    ]
    try:
        return pl.read_parquet(path, columns=wanted)
    except Exception:
        return pl.read_parquet(path)


def metadata_by_id(metadata_df: pl.DataFrame) -> dict[int, dict]:
    if metadata_df.is_empty() or "id" not in metadata_df.columns:
        return {}
    return {
        int(row["id"]): row
        for row in metadata_df.unique("id", maintain_order=True).iter_rows(named=True)
    }


def clean_value(value, default="?"):
    if value is None:
        return default
    if isinstance(value, float) and isnan(value):
        return default
    return value


def get_query_set_id(
    beatmap_id: int,
    raw_input: str,
    metadata_lookup: dict[int, dict],
) -> int | None:
    from_url = extract_beatmapset_id(raw_input)
    if from_url is not None:
        return from_url

    row = metadata_lookup.get(beatmap_id)
    if row:
        value = clean_value(row.get("beatmapset_id"), None)
        if value is not None:
            return int(value)
    return None


def sharded_osu_path(beatmap_id: int, beatmaps_dir: Path) -> Path:
    return Path(get_sharded_path(str(beatmap_id), str(beatmaps_dir)))


def ensure_osu_file(beatmap_id: int, beatmaps_dir: Path, allow_download: bool) -> Path:
    path = sharded_osu_path(beatmap_id, beatmaps_dir)
    if path.exists():
        content = path.read_bytes()
        if is_valid_osu_file(content):
            return path

    if not allow_download:
        raise FileNotFoundError(f"No valid .osu file found at {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for tier in API_TIERS:
        time.sleep(float(tier.get("delay", 0.0)))
        url = tier["url"].format(id=beatmap_id)
        try:
            response = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=15)
            if response.status_code == 200 and is_valid_osu_file(response.content):
                path.write_bytes(response.content)
                return path
            errors.append(f"{tier['name']}: HTTP {response.status_code}")
        except requests.exceptions.RequestException as exc:
            errors.append(f"{tier['name']}: {exc}")

    raise RuntimeError(f"Failed to download {beatmap_id}: {'; '.join(errors)}")


def beatmap_vectors_from_osu(path: Path, max_seq_len: int):
    vectors, _map_features = beatmap_inputs_from_osu(path, max_seq_len)
    return vectors


def beatmap_inputs_from_osu(path: Path, max_seq_len: int):
    from core.data.feature import build_feature_tensors
    from core.data.feature import calculate_drain_times
    from core.data.parser import parse_osu_file
    from scripts.tasks.data.dataset import (
        extract_beatmap_record,
        extract_hitobject_records,
        validate_beatmap,
    )

    raw_beatmap = parse_osu_file(str(path))
    if not validate_beatmap(raw_beatmap):
        raise ValueError(f"Could not parse a valid beatmap from {path}")

    beatmaps_df = pl.DataFrame([extract_beatmap_record(raw_beatmap)])
    hitobjects_df = pl.DataFrame(extract_hitobject_records(raw_beatmap))
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
        raise ValueError(f"Could not engineer hitobject features for {path}")

    expanded_count = vectors[0].shape[0]
    truncate_len = min(expanded_count, max_seq_len or expanded_count)
    row = beatmaps_df.row(0, named=True)
    map_features = {
        name: float(row.get(name, 0.0) or 0.0) for name in MAP_FEATURE_ATTRIBUTES
    }
    return vectors[0][:truncate_len], map_features


class LazyEmbedder:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path | None,
        device: str | None = None,
        pretrain: bool = False,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device_name = device
        self.pretrain = pretrain
        self.config = None
        self.model = None
        self.normalizer = None
        self.device = None

    def load(self):
        import torch
        from omegaconf import OmegaConf

        from core.data.normalizer import BeatmapNormalizer
        from scripts.tasks.embed.bobert import (
            find_checkpoint,
            load_alignment_model,
            load_pretraining_model,
        )

        if self.model is not None:
            return

        self.config = OmegaConf.load(self.config_path)
        self.device = torch.device(
            self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint_dir = (
            self.config.pretraining.checkpoint_dir
            if self.pretrain
            else self.config.alignment.checkpoint_dir
        )
        ckpt_path = find_checkpoint(self.checkpoint_path, checkpoint_dir)
        loader = load_pretraining_model if self.pretrain else load_alignment_model
        self.model, checkpoint = loader(self.config, ckpt_path, self.device)
        self.normalizer = BeatmapNormalizer(
            vector_stats=checkpoint["vector_stats"],
            attribute_stats=checkpoint.get("attribute_stats", {}),
        )

    def embed_osu(self, path: Path) -> np.ndarray:
        import torch

        from core.data.batch import pack_batch

        self.load()
        vectors, raw_map_features = beatmap_inputs_from_osu(
            path, self.config.data.max_seq_len
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

        packed = packed.to(self.device)
        cu_seqlens = cu_seqlens.to(self.device)
        map_features = map_features.to(self.device)
        amp_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=amp_dtype,
                enabled=self.device.type == "cuda",
            ):
                if self.pretrain:
                    embedding = self.model.embed_packed(
                        packed, cu_seqlens, max_seqlen
                    )
                else:
                    embedding = self.model.embed_packed(
                        packed, cu_seqlens, max_seqlen, map_features
                    )

        embedding = embedding.float().cpu().numpy()[0]
        norm = np.linalg.norm(embedding)
        return (embedding / max(norm, 1e-12)).astype(np.float32)


def format_number(value, decimals: int = 2):
    value = clean_value(value, None)
    if value is None:
        return "?"
    number = float(value)
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.{decimals}f}"


def format_length(value):
    value = clean_value(value, None)
    if value is None:
        return "?"
    seconds = int(round(float(value)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def format_bpm(value):
    value = clean_value(value, None)
    if value is None:
        return "?"
    return str(int(float(value)))


def beatmap_table_values(beatmap_id: int, row: dict | None):
    if row is None:
        return str(beatmap_id), "?", "?", "?", "?", "?", "?"

    title = clean_value(row.get("title"))
    creator = clean_value(row.get("creator"))
    version = clean_value(row.get("version"))
    stars = format_number(row.get("difficulty_rating"), 2)
    bpm = format_bpm(row.get("bpm"))
    length = format_length(clean_value(row.get("total_length"), row.get("hit_length")))
    return str(beatmap_id), title, creator, version, stars, bpm, length


def beatmap_table_values_missing(row: dict | None) -> bool:
    if row is None:
        return True
    return any(
        clean_value(value, None) is None
        for value in (
            row.get("title"),
            row.get("creator"),
            row.get("version"),
            row.get("difficulty_rating"),
            row.get("bpm"),
            clean_value(row.get("total_length"), row.get("hit_length")),
        )
    )


def beatmap_map_style(row: dict | None) -> str:
    status = str(clean_value((row or {}).get("status"), "")).lower()
    ranked = str(clean_value((row or {}).get("ranked"), "")).lower()
    values = {status, ranked}
    if "ranked" in values or "1" in values:
        return "yellow"
    if "loved" in values or "4" in values:
        return "bright_magenta"
    return "grey70"
