from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scripts.common.osu import (
    API_TIERS,
    DOWNLOAD_HEADERS,
    get_sharded_path,
    is_valid_osu_file,
)
from scripts.common.paths import PROJECT_ROOT


DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "bobert_alignment_embeddings.parquet"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "beatmaps.parquet"
DEFAULT_BEATMAPS_DIR = PROJECT_ROOT / "data" / "beatmaps"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def extract_beatmap_id(raw_input: str) -> int:
    text = raw_input.strip()
    if not text:
        raise ValueError("empty input")

    if "osu.ppy.sh" in text:
        hash_match = re.search(r"beatmapsets/\d+#(?:osu|taiko|fruits|mania)/(\d+)", text)
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

    df = pd.read_parquet(path)
    if "beatmap_id" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"Expected beatmap_id and embedding columns in {path}")

    beatmap_ids = df["beatmap_id"].astype(np.int64).to_numpy()
    embeddings = np.asarray(df["embedding"].tolist(), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    id_to_index = {int(beatmap_id): idx for idx, beatmap_id in enumerate(beatmap_ids)}
    return beatmap_ids, embeddings, id_to_index


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    wanted = [
        "id",
        "beatmapset_id",
        "artist",
        "title",
        "version",
        "difficulty_rating",
        "bpm",
    ]
    try:
        return pd.read_parquet(path, columns=wanted)
    except Exception:
        return pd.read_parquet(path)


def metadata_by_id(metadata_df: pd.DataFrame) -> dict[int, dict]:
    if metadata_df.empty or "id" not in metadata_df.columns:
        return {}
    return {
        int(row["id"]): row.to_dict()
        for _, row in metadata_df.drop_duplicates("id").iterrows()
    }


def clean_value(value, default="?"):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except ValueError:
        pass
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
    from core.data.features import engineer_features_vectorized
    from core.data.parser import parse_osu_file
    from scripts.tasks.data.dataset import (
        extract_beatmap_record,
        extract_hitobject_records,
        validate_beatmap,
    )

    raw_beatmap = parse_osu_file(str(path))
    if not validate_beatmap(raw_beatmap):
        raise ValueError(f"Could not parse a valid beatmap from {path}")

    beatmaps_df = pd.DataFrame([extract_beatmap_record(raw_beatmap)])
    hitobjects_df = pd.DataFrame(extract_hitobject_records(raw_beatmap))
    vectors, ids, original_counts = engineer_features_vectorized(beatmaps_df, hitobjects_df)
    if not vectors:
        raise ValueError(f"Could not engineer hitobject features for {path}")

    beatmap_id = int(ids[0])
    original_count = original_counts.get(beatmap_id, vectors[0].shape[0])
    truncate_len = min(original_count, max_seq_len)
    return vectors[0][:truncate_len]


class LazyEmbedder:
    def __init__(self, config_path: Path, checkpoint_path: Path | None):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.config = None
        self.model = None
        self.normalizer = None
        self.device = None

    def load(self):
        import torch
        from omegaconf import OmegaConf

        from core.data.transforms import BeatmapNormalizer
        from scripts.tasks.embed.bobert import find_checkpoint, load_alignment_model

        if self.model is not None:
            return

        self.config = OmegaConf.load(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = find_checkpoint(self.checkpoint_path)
        self.model, checkpoint = load_alignment_model(self.config, ckpt_path, self.device)
        self.normalizer = BeatmapNormalizer(
            vector_stats=checkpoint["vector_stats"],
            attribute_stats=checkpoint.get("attribute_stats", {}),
        )

    def embed_osu(self, path: Path) -> np.ndarray:
        import torch

        from core.data.datamodule import _pad_batch

        self.load()
        vectors = beatmap_vectors_from_osu(path, self.config.data.max_seq_len)
        vectors = self.normalizer.normalize_vectors(vectors)
        vector_dim = vectors.shape[1]
        padded, mask, cu_seqlens = _pad_batch([vectors], self.config.data.max_seq_len, vector_dim)

        padded = padded.to(self.device)
        mask = mask.to(self.device)
        cu_seqlens = cu_seqlens.to(self.device)
        amp_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=amp_dtype,
                enabled=self.device.type == "cuda",
            ):
                embedding = self.model(padded, mask, cu_seqlens)["embedding"]

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


def format_beatmap_line(beatmap_id: int, row: dict | None):
    url = f"https://osu.ppy.sh/b/{beatmap_id}"
    if row is None:
        return f"{url}\n    ? - ? [?]\n    ?★ · ? BPM"

    artist = clean_value(row.get("artist"))
    title = clean_value(row.get("title"))
    version = clean_value(row.get("version"))
    stars = format_number(row.get("difficulty_rating"), 2)
    bpm = format_number(row.get("bpm"), 1)
    return f"{url}\n    {artist} - {title} [{version}]\n    {stars}★ · {bpm} BPM"


def load_beatmap_titles(path, columns=("id", "beatmapset_id", "title")) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns))


def unique_beatmapset_results(df: pd.DataFrame, limit: int) -> list[pd.Series]:
    seen = set()
    rows = []
    for _, row in df.iterrows():
        if row["beatmapset_id"] in seen:
            continue
        seen.add(row["beatmapset_id"])
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def print_similarity_table(rows: list[pd.Series], *, title: str, max_width: int = 100) -> None:
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    sim_col_w = 6
    id_col_w = 10
    max_title_len = terminal_width - sim_col_w - id_col_w - 2

    print(f"\n{title}:\n")
    print(f"{'Sim':<{sim_col_w}} {'ID':<{id_col_w}} {'Name'}")
    print("-" * min(terminal_width, max_width))

    for row in rows:
        name = str(row["title"])
        if len(name) > max_title_len:
            name = name[: max_title_len - 3] + "..."
        print(f"{row['similarity']:<{sim_col_w}.3f} {int(row['beatmap_id']):<{id_col_w}} {name}")
