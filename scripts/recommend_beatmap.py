import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.datamodule import _pad_batch
from core.data.features import engineer_features_vectorized
from core.data.parser import parse_osu_file
from core.data.transforms import BeatmapNormalizer
from scripts.create_dataset import (
    extract_beatmap_record,
    extract_hitobject_records,
    validate_beatmap,
)
from scripts.export_bobert_embeddings import find_checkpoint, load_alignment_model
from scripts.fetch_osu import API_TIERS, DOWNLOAD_HEADERS, is_valid_osu_file
from scripts.shard_beatmaps import get_sharded_path

DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "bobert_alignment_embeddings.parquet"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "beatmaps.parquet"
DEFAULT_BEATMAPS_DIR = PROJECT_ROOT / "data" / "beatmaps"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def resolve_path(path: str | Path) -> Path:
    path = Path(str(path).strip()).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


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


def beatmap_vectors_from_osu(path: Path, max_seq_len: int) -> torch.Tensor:
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


def format_result(rank: int, similarity: float, beatmap_id: int, row: dict | None):
    url = f"https://osu.ppy.sh/b/{beatmap_id}"
    if row is None:
        return f"{rank:>2}. {similarity:.3f}  {url}\n    ? - ? [?]\n    ?★ · ? BPM"

    artist = clean_value(row.get("artist"))
    title = clean_value(row.get("title"))
    version = clean_value(row.get("version"))
    stars = format_number(row.get("difficulty_rating"), 2)
    bpm = format_number(row.get("bpm"), 1)
    return (
        f"{rank:>2}. {similarity:.3f}  {url}\n"
        f"    {artist} - {title} [{version}]\n"
        f"    {stars}★ · {bpm} BPM"
    )


def recommend(
    raw_input: str,
    beatmap_ids: np.ndarray,
    embeddings: np.ndarray,
    id_to_index: dict[int, int],
    metadata_lookup: dict[int, dict],
    embedder: LazyEmbedder,
    beatmaps_dir: Path,
    top_k: int,
    include_same_set: bool,
    allow_download: bool,
):
    beatmap_id = extract_beatmap_id(raw_input)
    query_set_id = get_query_set_id(beatmap_id, raw_input, metadata_lookup)

    if beatmap_id in id_to_index:
        query_embedding = embeddings[id_to_index[beatmap_id]]
        source = "stored embedding"
    else:
        osu_path = ensure_osu_file(beatmap_id, beatmaps_dir, allow_download)
        query_embedding = embedder.embed_osu(osu_path)
        source = f"embedded {osu_path}"

    similarities = embeddings @ query_embedding
    order = np.argsort(-similarities)
    results = []

    for idx in order:
        candidate_id = int(beatmap_ids[idx])
        if candidate_id == beatmap_id:
            continue

        row = metadata_lookup.get(candidate_id)
        candidate_set_id = None
        if row:
            value = clean_value(row.get("beatmapset_id"), None)
            if value is not None:
                candidate_set_id = int(value)

        if not include_same_set and query_set_id is not None and candidate_set_id == query_set_id:
            continue

        results.append((candidate_id, float(similarities[idx]), row))
        if len(results) >= top_k:
            break

    print(f"\nQuery: https://osu.ppy.sh/b/{beatmap_id} ({source})")
    if query_set_id is not None and not include_same_set:
        print(f"Excluding same beatmapset: {query_set_id}")
    print()
    for rank, (candidate_id, similarity, row) in enumerate(results, 1):
        print(format_result(rank, similarity, candidate_id, row))
    print()


def run_interactive(args, loaded):
    print("Paste a beatmap id or osu! URL. Press Ctrl+C/Ctrl+D, q, quit, or empty input to exit.")
    while True:
        try:
            raw_input = input("beatmap> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if raw_input.lower() in {"", "q", "quit", "exit"}:
            return

        try:
            recommend(raw_input, *loaded)
        except Exception as exc:
            print(f"Error: {exc}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend nearest Bobert embedding neighbors for an osu! beatmap id or URL"
    )
    parser.add_argument("beatmap", nargs="?", help="Beatmap id or osu! URL. Omit for interactive mode.")
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--beatmaps-dir", default=str(DEFAULT_BEATMAPS_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", default=None, help="Defaults to newest experiments/**/checkpoints/last.ckpt")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--include-same-set", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings_path = resolve_path(args.embeddings)
    metadata_path = resolve_path(args.metadata)
    beatmaps_dir = resolve_path(args.beatmaps_dir)
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else None

    beatmap_ids, embeddings, id_to_index = load_embeddings(embeddings_path)
    metadata_df = load_metadata(metadata_path)
    lookup = metadata_by_id(metadata_df)
    embedder = LazyEmbedder(config_path, checkpoint_path)

    loaded = (
        beatmap_ids,
        embeddings,
        id_to_index,
        lookup,
        embedder,
        beatmaps_dir,
        args.top_k,
        args.include_same_set,
        not args.no_download,
    )

    print(f"Loaded {len(beatmap_ids):,} embeddings from {embeddings_path}")
    if args.beatmap:
        recommend(args.beatmap, *loaded)
    else:
        run_interactive(args, loaded)


if __name__ == "__main__":
    main()
