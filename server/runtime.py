from __future__ import annotations

import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isnan
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from core.model import BobertEncoder, EmbeddingTransform

DATA_DIR = Path(os.getenv("BOBERT_DATA_DIR", "/app/data"))
RUN_DIR = Path(os.getenv("BOBERT_RUN_DIR", "/app/runs/current"))
CACHE_DB = Path(os.getenv("BOBERT_CACHE_DB", "/app/cache/runtime.sqlite"))
MODEL_PATH = Path(os.getenv("BOBERT_MODEL_PATH", RUN_DIR / "bobert.pt"))
EMBEDDINGS_PATH = Path(
    os.getenv("BOBERT_EMBEDDINGS_PATH", RUN_DIR / "embeddings.parquet")
)
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"
BEATMAPSETS_PATH = DATA_DIR / "beatmapsets.parquet"
SEARCH_CANDIDATE_FACTORS = (10, 25, 100)
DEFAULT_COUNTS = (15, 40, 30, 15)
RANKED_STATUS_VALUES = {"1", "2", "3", "4", "ranked", "approved", "qualified", "loved"}
STATUS_GROUPS = {
    "ranked": {"1", "2", "3", "ranked", "approved", "qualified"},
    "loved": {"4", "loved"},
    "unranked": {"-2", "-1", "0", "graveyard", "wip", "pending"},
}
SEARCH_COLUMNS = [
    "id",
    "beatmapset_id",
    "user_id",
    "artist",
    "title",
    "creator",
    "version",
    "mode",
    "status",
    "ranked",
    "difficulty_rating",
    "ar",
    "cs",
    "accuracy",
    "drain",
    "bpm",
    "total_length",
    "hit_length",
    "url",
    "last_updated",
    "ranked_date",
    "submitted_date",
    "favourite_count",
    "play_count",
]


@dataclass(frozen=True, slots=True)
class CachedEmbedding:
    beatmap_id: int
    embedding: np.ndarray
    metadata: dict[str, Any]


class SQLiteCache:
    def __init__(self, path: Path, embedding_dim: int, run_id: str):
        self.path = path
        self.embedding_dim = embedding_dim
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (beatmap_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL, metadata_json TEXT NOT NULL, created_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS unavailable_beatmaps (beatmap_id INTEGER PRIMARY KEY, reason TEXT NOT NULL, created_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = conn.execute(
                "SELECT value FROM cache_metadata WHERE key = 'run_id'"
            ).fetchone()
            if row is not None and row[0] != run_id:
                conn.execute("DELETE FROM embeddings")
                conn.execute("DELETE FROM unavailable_beatmaps")
            conn.execute(
                "INSERT OR REPLACE INTO cache_metadata (key, value) VALUES ('run_id', ?)",
                (run_id,),
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def get(self, beatmap_id: int) -> CachedEmbedding | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT embedding, metadata_json FROM embeddings WHERE beatmap_id = ?",
                (int(beatmap_id),),
            ).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row[0], dtype=np.float32).copy()
        if vector.shape != (self.embedding_dim,):
            return None
        return CachedEmbedding(int(beatmap_id), vector, json.loads(row[1] or "{}"))

    def upsert(
        self, beatmap_id: int, embedding: np.ndarray, metadata: dict[str, Any]
    ) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (self.embedding_dim,):
            raise ValueError(f"invalid embedding shape: {vector.shape}")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (beatmap_id, embedding, metadata_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    int(beatmap_id),
                    vector.tobytes(),
                    json.dumps(metadata, separators=(",", ":")),
                    int(time.time()),
                ),
            )

    def delete(self, beatmap_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM embeddings WHERE beatmap_id = ?", (beatmap_id,))

    def is_unavailable(self, beatmap_id: int) -> bool:
        with self._connect() as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM unavailable_beatmaps WHERE beatmap_id = ?",
                    (beatmap_id,),
                ).fetchone()
                is not None
            )

    def mark_unavailable(self, beatmap_id: int, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO unavailable_beatmaps (beatmap_id, reason, created_at) VALUES (?, ?, ?)",
                (beatmap_id, reason, int(time.time())),
            )


class Runtime:
    def __init__(self) -> None:
        missing = [
            str(path)
            for path in (BEATMAPS_PATH, BEATMAPSETS_PATH, EMBEDDINGS_PATH, MODEL_PATH)
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(f"missing required data files: {missing}")

        frame = pl.read_parquet(
            EMBEDDINGS_PATH, columns=["beatmap_id", "embedding", "density"]
        )
        self.static_ids = [int(value) for value in frame["beatmap_id"].to_list()]
        self.static_embeddings = np.array(
            frame["embedding"].to_numpy(),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        self.static_densities = (
            frame["density"].to_numpy().astype(np.float32, copy=True)
        )
        del frame
        if (
            self.static_embeddings.ndim != 2
            or not self.static_embeddings.flags.c_contiguous
            or self.static_embeddings.shape[0] != len(self.static_ids)
        ):
            raise RuntimeError("invalid embeddings.parquet shape")
        sidecar = json.loads(
            EMBEDDINGS_PATH.with_suffix(".json").read_text(encoding="utf-8")
        )
        retrieval = sidecar.get("retrieval", {})
        if retrieval.get("method") != "csls":
            raise RuntimeError("embeddings do not contain a CSLS retrieval index")
        self.retrieval_density_k = int(retrieval["density_k"])
        self.retrieval_lambda = float(retrieval["lambda"])
        if (
            self.static_densities.shape != (len(self.static_ids),)
            or not np.isfinite(self.static_densities).all()
            or self.retrieval_density_k <= 0
            or self.retrieval_density_k >= len(self.static_ids)
            or not np.isfinite(self.retrieval_lambda)
            or self.retrieval_lambda < 0
        ):
            raise RuntimeError("invalid CSLS retrieval index")
        self.transform = EmbeddingTransform(
            np.asarray(sidecar["layer_means"], dtype=np.float32)
        )
        self.static_embeddings /= np.maximum(
            np.linalg.norm(self.static_embeddings, axis=1, keepdims=True), 1e-12
        )
        self.static_embeddings.flags.writeable = False
        self.embedding_dim = self.static_embeddings.shape[1]
        self.static_index = {
            beatmap_id: index for index, beatmap_id in enumerate(self.static_ids)
        }

        beatmaps = pl.read_parquet(BEATMAPS_PATH, columns=SEARCH_COLUMNS)
        defaults = (
            beatmaps.filter(pl.col("status").is_in(["1", "4"]), pl.col("mode") == "osu")
            .sort("difficulty_rating", descending=True)
            .unique("beatmapset_id", keep="first", maintain_order=True)
            .join(pl.DataFrame({"id": self.static_ids}), on="id", how="semi")
            .filter(
                pl.col("difficulty_rating").is_between(5, 9, closed="left"),
                pl.col("play_count") > 50000,
                pl.col("favourite_count")
                >= pl.min_horizontal(pl.col("play_count") / 250, pl.lit(1000)),
            )
            .sort("id")
        )
        self.metadata_by_id = {
            int(row["id"]): row
            for row in beatmaps.unique("id", maintain_order=True).iter_rows(named=True)
        }
        self.default_pools = [
            [
                public_summary(int(beatmap_id), self.metadata_by_id[int(beatmap_id)])
                for beatmap_id in defaults.filter(
                    pl.col("difficulty_rating").floor() == star
                )["id"]
            ]
            for star in range(5, 9)
        ]
        del beatmaps
        self.dynamic_ids: list[int] = []
        self.dynamic_embeddings: list[np.ndarray] = []
        self.dynamic_densities: list[float] = []
        self.dynamic_index: dict[int, int] = {}
        self.dynamic_metadata: dict[int, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.inference_semaphore = threading.Semaphore(1)
        self.cache = SQLiteCache(CACHE_DB, self.embedding_dim, RUN_DIR.resolve().name)

        device = torch.device("cpu")
        self.model, self.vector_stats = BobertEncoder.from_pretrained(
            MODEL_PATH, device
        )
        self.model.to(device).float().eval()
        if self.model.d_model != self.embedding_dim:
            raise RuntimeError(
                f"model dimension {self.model.d_model} does not match embeddings dimension {self.embedding_dim}"
            )

    def default_summaries(self, seed: int | None = None) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        results = [
            item
            for pool, count in zip(self.default_pools, DEFAULT_COUNTS)
            for item in rng.sample(pool, count)
        ]
        rng.shuffle(results)
        return results

    def memory_embedding(
        self, beatmap_id: int
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        static_index = self.static_index.get(beatmap_id)
        if static_index is not None:
            return self.static_embeddings[static_index], self.metadata_by_id.get(
                beatmap_id, {}
            )
        with self.lock:
            dynamic_index = self.dynamic_index.get(beatmap_id)
            if dynamic_index is None:
                return None
            return self.dynamic_embeddings[dynamic_index], self.dynamic_metadata[
                beatmap_id
            ]

    def cached_embedding(
        self, beatmap_id: int
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        cached = self.cache.get(beatmap_id)
        if cached is None:
            return None
        if not metadata_complete(cached.metadata):
            self.cache.delete(beatmap_id)
            return None
        return (
            self._append(
                beatmap_id,
                cached.embedding,
                self._density(cached.embedding),
                cached.metadata,
            ),
            cached.metadata,
        )

    def infer_and_store(
        self, beatmap_id: int, content: bytes, metadata: dict[str, Any]
    ) -> np.ndarray:
        with self.inference_semaphore:
            raw = self.model.embed_osu_bytes(
                content, self.vector_stats, beatmap_id=beatmap_id
            )
        transformed = self._transform(raw)
        self.cache.upsert(beatmap_id, transformed, metadata)
        return self._append(
            beatmap_id, transformed, self._density(transformed), metadata
        )

    def _density(self, vector: np.ndarray) -> float:
        scores = self.static_embeddings @ vector
        neighbors = np.partition(scores, -self.retrieval_density_k)[
            -self.retrieval_density_k :
        ]
        return float(neighbors.mean())

    def _transform(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != self.transform.means.shape:
            raise ValueError(f"invalid embedding shape: {vector.shape}")
        transformed = self.transform.apply(vector)
        if (
            transformed.shape != (self.embedding_dim,)
            or not np.isfinite(transformed).all()
        ):
            raise ValueError("embedding transform returned an invalid vector")
        return transformed

    def _append(
        self,
        beatmap_id: int,
        vector: np.ndarray,
        density: float,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        with self.lock:
            index = self.dynamic_index.get(beatmap_id)
            if index is not None:
                return self.dynamic_embeddings[index]
            self.dynamic_index[beatmap_id] = len(self.dynamic_ids)
            self.dynamic_ids.append(beatmap_id)
            vector.flags.writeable = False
            self.dynamic_embeddings.append(vector)
            self.dynamic_densities.append(density)
            self.dynamic_metadata[beatmap_id] = metadata
            return vector

    def summary(self, beatmap_id: int) -> dict[str, Any]:
        metadata = self.metadata_by_id.get(beatmap_id)
        if metadata is None:
            with self.lock:
                metadata = self.dynamic_metadata.get(beatmap_id, {})
        return public_summary(beatmap_id, metadata)

    def search(
        self,
        query_beatmap_id: int,
        query_embedding: np.ndarray,
        query_metadata: dict[str, Any],
        top_k: int,
        filters: Any,
    ) -> list[dict[str, Any]]:
        static_scores = self.static_embeddings @ query_embedding
        with self.lock:
            dynamic_ids = list(self.dynamic_ids)
            dynamic_vectors = list(self.dynamic_embeddings)
            dynamic_densities = list(self.dynamic_densities)
            dynamic_metadata = dict(self.dynamic_metadata)
        if dynamic_vectors:
            dynamic_scores = (
                np.asarray(dynamic_vectors, dtype=np.float32) @ query_embedding
            )
            similarities = np.concatenate((static_scores, dynamic_scores))
            densities = np.concatenate(
                (self.static_densities, np.asarray(dynamic_densities, dtype=np.float32))
            )
        else:
            similarities = static_scores
            densities = self.static_densities
        scores = similarities - self.retrieval_lambda * 0.5 * densities
        static_count = len(self.static_ids)
        query_set_id = metadata_set_id(query_metadata)
        date_cutoff = date_window_cutoff(filters.date_window)
        seen_set_ids: set[int] = set()
        results: list[dict[str, Any]] = []
        evaluated: set[int] = set()
        for indices in candidate_index_batches(scores, top_k):
            for raw_index in indices:
                index = int(raw_index)
                if index in evaluated:
                    continue
                evaluated.add(index)
                beatmap_id = (
                    self.static_ids[index]
                    if index < static_count
                    else dynamic_ids[index - static_count]
                )
                if beatmap_id == query_beatmap_id:
                    continue
                metadata = self.metadata_by_id.get(
                    beatmap_id, dynamic_metadata.get(beatmap_id, {})
                )
                candidate_set_id = metadata_set_id(metadata)
                if not passes_filters(metadata, filters, date_cutoff):
                    continue
                if (
                    filters.exclude_same_set
                    and query_set_id is not None
                    and candidate_set_id == query_set_id
                ):
                    continue
                if candidate_set_id is not None and candidate_set_id in seen_set_ids:
                    continue
                result = public_summary(beatmap_id, metadata)
                result["score"] = float(similarities[index])
                results.append(result)
                if candidate_set_id is not None:
                    seen_set_ids.add(candidate_set_id)
                if len(results) == top_k:
                    return results
        return results

    def catalog_detail(self, beatmap_id: int) -> dict[str, Any] | None:
        beatmap_rows = (
            pl.scan_parquet(BEATMAPS_PATH)
            .filter(pl.col("id") == beatmap_id)
            .limit(1)
            .collect()
        )
        beatmap = beatmap_rows.row(0, named=True) if beatmap_rows.height else None
        if beatmap is None:
            with self.lock:
                dynamic = self.dynamic_metadata.get(beatmap_id)
            if dynamic is None:
                return None
            beatmap = {"id": beatmap_id, **dynamic}
        beatmapset_id = metadata_set_id(beatmap)
        beatmapsets = pl.scan_parquet(BEATMAPSETS_PATH)
        set_rows = (
            beatmapsets.filter(pl.col("beatmap_id") == beatmap_id).limit(1).collect()
        )
        if not set_rows.height and beatmapset_id is not None:
            set_rows = (
                beatmapsets.filter(pl.col("beatmapset_id") == beatmapset_id)
                .limit(1)
                .collect()
            )
        beatmapset = set_rows.row(0, named=True) if set_rows.height else None
        return {"beatmap": json_value(beatmap), "beatmapset": json_value(beatmapset)}


def candidate_index_batches(scores: np.ndarray, top_k: int):
    count = len(scores)
    previous = 0
    for factor in SEARCH_CANDIDATE_FACTORS:
        candidate_count = min(count, max(top_k * factor, top_k))
        if candidate_count <= previous:
            continue
        previous = candidate_count
        if candidate_count == count:
            yield np.argsort(-scores)
            return
        indices = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
        yield indices[np.argsort(-scores[indices])]
    yield np.argsort(-scores)


def public_summary(beatmap_id: int, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "beatmap_id": beatmap_id,
        "beatmapset_id": json_value(metadata.get("beatmapset_id")),
        "artist": json_value(metadata.get("artist")),
        "title": json_value(metadata.get("title")),
        "creator": json_value(metadata.get("creator")),
        "user_id": json_value(metadata.get("user_id")),
        "version": json_value(metadata.get("version")),
        "status": json_value(metadata.get("status")),
        "stars": json_value(metadata.get("difficulty_rating", metadata.get("stars"))),
        "ar": json_value(metadata.get("ar")),
        "cs": json_value(metadata.get("cs")),
        "accuracy": json_value(metadata.get("accuracy")),
        "drain": json_value(metadata.get("drain")),
        "bpm": json_value(metadata.get("bpm")),
        "total_length": json_value(metadata.get("total_length")),
        "last_updated": json_value(metadata.get("last_updated")),
        "ranked_date": json_value(metadata.get("ranked_date")),
        "submitted_date": json_value(metadata.get("submitted_date")),
        "release_date": json_value(metadata_release_date(metadata)),
        "url": json_value(metadata.get("url")) or f"https://osu.ppy.sh/b/{beatmap_id}",
    }


def metadata_complete(metadata: dict[str, Any] | None) -> bool:
    if not metadata:
        return False
    required = (
        metadata.get("title"),
        metadata.get("creator"),
        metadata.get("version"),
        metadata.get("difficulty_rating", metadata.get("stars")),
        metadata.get("bpm"),
        metadata.get("total_length", metadata.get("hit_length")),
    )
    return all(json_value(value) is not None for value in required) and (
        json_value(metadata.get("status")) is not None
        or json_value(metadata.get("ranked")) is not None
    )


def passes_filters(
    metadata: dict[str, Any], filters: Any, date_cutoff: datetime | None
) -> bool:
    stars = metadata.get("difficulty_rating", metadata.get("stars"))
    if not passes_range(stars, filters.min_sr, filters.max_sr):
        return False
    for key, minimum, maximum in (
        ("ar", filters.min_ar, filters.max_ar),
        ("cs", filters.min_cs, filters.max_cs),
        ("accuracy", filters.min_accuracy, filters.max_accuracy),
        ("drain", filters.min_drain, filters.max_drain),
        ("bpm", filters.min_bpm, filters.max_bpm),
    ):
        if not passes_range(metadata.get(key), minimum, maximum):
            return False
    if not passes_range(
        metadata.get("total_length", metadata.get("hit_length")),
        filters.min_length,
        filters.max_length,
    ):
        return False
    if filters.status:
        statuses = {
            str(metadata.get("status", "")).lower(),
            str(metadata.get("ranked", "")).lower(),
        }
        accepted = STATUS_GROUPS.get(filters.status.lower(), {filters.status.lower()})
        if statuses.isdisjoint(accepted):
            return False
    if date_cutoff is not None:
        release_date = parse_metadata_datetime(metadata_release_date(metadata))
        if release_date is None or release_date < date_cutoff:
            return False
    return True


def passes_range(value: Any, minimum: float | None, maximum: float | None) -> bool:
    if minimum is None and maximum is None:
        return True
    if value is None:
        return False
    number = float(value)
    return (
        not isnan(number)
        and (minimum is None or number >= minimum)
        and (maximum is None or number <= maximum)
    )


def metadata_set_id(metadata: dict[str, Any] | None) -> int | None:
    value = (metadata or {}).get("beatmapset_id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def metadata_release_date(metadata: dict[str, Any]) -> Any:
    if any(
        str(json_value(metadata.get(key))).lower() in RANKED_STATUS_VALUES
        for key in ("status", "ranked")
    ):
        ranked_date = json_value(metadata.get("ranked_date"))
        if ranked_date is not None:
            return ranked_date
    return json_value(metadata.get("submitted_date"))


def date_window_cutoff(window: Any) -> datetime | None:
    if window is None or str(window.value) == "all_time":
        return None
    now = datetime.now(UTC)
    months = {
        "last_month": -1,
        "last_3_months": -3,
        "last_6_months": -6,
        "last_year": -12,
        "last_2_years": -24,
        "last_5_years": -60,
    }
    if window.value == "last_week":
        return now - timedelta(days=7)
    month_index = now.month - 1 + months[window.value]
    year = now.year + month_index // 12
    month = month_index % 12 + 1
    days = (
        29
        if month == 2 and (year % 400 == 0 or (year % 4 == 0 and year % 100))
        else 28
        if month == 2
        else 30
        if month in {4, 6, 9, 11}
        else 31
    )
    return now.replace(year=year, month=month, day=min(now.day, days))


def parse_metadata_datetime(value: Any) -> datetime | None:
    value = json_value(value)
    if value is None:
        return None
    try:
        parsed = (
            value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        )
    except ValueError:
        return None
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and isnan(value):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return value
