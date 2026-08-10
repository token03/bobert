from __future__ import annotations

import re
import time
from math import isnan
from pathlib import Path

import numpy as np
import polars as pl
import requests
from core.features import normalize
from scripts.common.osu import (
    API_TIERS,
    DOWNLOAD_HEADERS,
    get_sharded_path,
    is_valid_osu_file,
)
from scripts.common.paths import PROJECT_ROOT


DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "beatmaps.parquet"
DEFAULT_BEATMAPS_DIR = PROJECT_ROOT / "data" / "beatmaps"


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


def load_embeddings(
    path: Path,
    *,
    dtype: np.dtype | type | None = np.float32,
    normalize: bool = True,
):
    if not path.exists():
        raise FileNotFoundError(f"Embeddings parquet not found: {path}")

    df = pl.read_parquet(path)
    if "beatmap_id" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"Expected beatmap_id and embedding columns in {path}")

    beatmap_ids = df["beatmap_id"].to_numpy().astype(np.int64)
    embeddings = df["embedding"].to_numpy()
    if embeddings.dtype == object:
        embeddings = np.stack(embeddings)
    if normalize:
        embeddings = embeddings.astype(np.float32, copy=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
    if dtype is not None:
        embeddings = embeddings.astype(dtype, copy=False)
    id_to_index = {int(beatmap_id): idx for idx, beatmap_id in enumerate(beatmap_ids)}
    return beatmap_ids, embeddings, id_to_index


def load_metadata(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()

    wanted = [
        "id",
        "beatmapset_id",
        "user_id",
        "owners",
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
    from core.features import build_feature_tensors
    from core.osu import (
        extract_beatmap_record,
        extract_hitobject_records,
        parse_osu_file,
    )
    from scripts.dataset.build import (
        validate_beatmap,
    )

    raw_beatmap = parse_osu_file(str(path))
    if not validate_beatmap(raw_beatmap):
        raise ValueError(f"Could not parse a valid beatmap from {path}")

    beatmaps_df = pl.DataFrame([extract_beatmap_record(raw_beatmap)])
    hitobjects_df = pl.DataFrame(extract_hitobject_records(raw_beatmap))
    vectors, _ids = build_feature_tensors(
        beatmaps_df,
        hitobjects_df,
        max_seq_len=max_seq_len,
    )
    if not vectors:
        raise ValueError(f"Could not engineer hitobject features for {path}")
    return vectors[0][:max_seq_len]


class LazyEmbedder:
    def __init__(
        self,
        model_path: Path,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.device_name = device
        self.model = None
        self.vector_stats = None
        self.device = None

    def load(self):
        import torch

        from scripts.model.embed import find_model, load_model

        if self.model is not None:
            return

        self.device = torch.device(
            self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.vector_stats = load_model(
            find_model(self.model_path), self.device
        )

    def embed_osu(self, path: Path) -> np.ndarray:
        import torch

        self.load()
        vectors = beatmap_vectors_from_osu(path, self.model.max_seq_len)
        vectors = normalize(vectors, self.vector_stats)
        packed = vectors[: self.model.max_seq_len].contiguous()
        max_seqlen = packed.shape[0]
        cu_seqlens = torch.tensor([0, max_seqlen], dtype=torch.int32)
        packed = packed.to(self.device)
        cu_seqlens = cu_seqlens.to(self.device)
        amp_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=amp_dtype,
                enabled=self.device.type == "cuda",
            ):
                embedding = self.model.embed_packed(packed, cu_seqlens, max_seqlen)

        embedding = embedding.float().cpu().numpy()[0]
        return embedding.astype(np.float32)


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
