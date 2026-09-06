from __future__ import annotations

import json
import os
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
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
STRAINS_PATH = DATA_DIR / "strains.parquet"
FILTERED_SCORE_THRESHOLD = 0.1
FILTERED_SCORE_BATCH_SIZE = 8192
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


def mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("embeddings must be a non-empty matrix")
    combined = vectors.mean(axis=0)
    norm = float(np.linalg.norm(combined))
    if not np.isfinite(combined).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("combined embedding is invalid")
    return np.asarray(combined / norm, dtype=np.float32)


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
            for path in (
                BEATMAPS_PATH,
                BEATMAPSETS_PATH,
                STRAINS_PATH,
                EMBEDDINGS_PATH,
                MODEL_PATH,
            )
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
        self.static_penalty = np.asarray(
            -0.5 * self.retrieval_lambda * self.static_densities, dtype=np.float32
        )
        self.static_penalty.flags.writeable = False
        del self.static_densities
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

        strains = (
            pl.read_parquet(
                STRAINS_PATH,
                columns=["beatmap_id", "seq_len", "actual_stars"],
            )
            .filter(pl.col("actual_stars").is_not_null())
            .sort("seq_len", descending=True)
            .unique("beatmap_id", keep="first")
            .sort("beatmap_id")
        )
        self.strain_ids = strains["beatmap_id"].to_numpy()
        self.strain_stars = strains["actual_stars"].to_numpy()
        beatmaps = (
            pl.read_parquet(BEATMAPS_PATH, columns=SEARCH_COLUMNS)
            .join(
                strains.select(
                    pl.col("beatmap_id").alias("id"),
                    pl.col("actual_stars"),
                ),
                on="id",
                how="left",
            )
            .with_columns(
                pl.coalesce("actual_stars", "difficulty_rating").alias(
                    "difficulty_rating"
                )
            )
            .drop("actual_stars")
        )
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
        static_order = pl.DataFrame(
            {"id": self.static_ids, "_static_index": range(len(self.static_ids))}
        )
        self.static_metadata = (
            static_order.join(
                beatmaps.unique("id", maintain_order=True), on="id", how="left"
            )
            .sort("_static_index")
            .drop("_static_index")
        )
        self.static_id_values = np.asarray(self.static_ids, dtype=np.int64)
        self.static_set_ids = (
            self.static_metadata["beatmapset_id"]
            .fill_null(-1)
            .to_numpy()
            .astype(np.int64, copy=False)
        )
        missing_set_ids = self.static_set_ids == -1
        self.static_set_groups = np.empty(len(self.static_ids), dtype=np.int32)
        if missing_set_ids.all():
            first_missing_group = 0
        else:
            _, self.static_set_groups[~missing_set_ids] = np.unique(
                self.static_set_ids[~missing_set_ids], return_inverse=True
            )
            first_missing_group = (
                int(self.static_set_groups[~missing_set_ids].max()) + 1
            )
        if missing_set_ids.any():
            self.static_set_groups[missing_set_ids] = np.arange(
                first_missing_group,
                first_missing_group + int(missing_set_ids.sum()),
                dtype=np.int32,
            )
        self.static_set_group_count = int(self.static_set_groups.max()) + 1
        self.static_set_order = np.argsort(
            self.static_set_groups, kind="stable"
        ).astype(np.int32)
        ordered_set_groups = self.static_set_groups[self.static_set_order]
        self.static_set_starts = np.flatnonzero(
            np.r_[True, ordered_set_groups[1:] != ordered_set_groups[:-1]]
        ).astype(np.int32)
        self.filter_values = {
            column: self.static_metadata[column]
            .cast(pl.Float64, strict=False)
            .fill_null(float("nan"))
            .to_numpy()
            for column in (
                "difficulty_rating",
                "ar",
                "cs",
                "accuracy",
                "drain",
                "bpm",
                "total_length",
            )
        }
        status_values = (
            self.static_metadata["status"]
            .cast(pl.String, strict=False)
            .fill_null("none")
            .str.to_lowercase()
            .to_numpy()
        )
        ranked_values = (
            self.static_metadata["ranked"]
            .cast(pl.String, strict=False)
            .fill_null("none")
            .str.to_lowercase()
            .to_numpy()
        )
        self.static_status_values = (status_values, ranked_values)
        self.static_status_groups = {
            group: np.isin(status_values, list(accepted))
            | np.isin(ranked_values, list(accepted))
            for group, accepted in STATUS_GROUPS.items()
        }
        ranked_status = (
            pl.col("status")
            .cast(pl.String, strict=False)
            .fill_null("")
            .str.to_lowercase()
            .is_in(RANKED_STATUS_VALUES)
            | pl.col("ranked")
            .cast(pl.String, strict=False)
            .fill_null("")
            .str.to_lowercase()
            .is_in(RANKED_STATUS_VALUES)
        )
        release_date = (
            pl.when(ranked_status & pl.col("ranked_date").is_not_null())
            .then(pl.col("ranked_date"))
            .otherwise(pl.col("submitted_date"))
        )
        self.static_release_dates = (
            self.static_metadata.select(
                release_date
                .cast(pl.String, strict=False)
                .str.to_datetime(strict=False, time_zone="UTC")
                .dt.epoch("us")
                .fill_null(np.iinfo(np.int64).min)
                .alias("release_date")
            )["release_date"]
            .to_numpy()
            .astype(np.int64, copy=False)
        )
        self.default_pools = [
            [
                self.public_summary(int(row["id"]), row)
                for row in defaults.filter(
                    pl.col("difficulty_rating").floor() == star
                ).iter_rows(named=True)
            ]
            for star in range(5, 9)
        ]
        del beatmaps
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
            return self.static_embeddings[static_index], self.static_metadata.row(
                static_index, named=True
            )
        return None

    def cached_embedding(
        self, beatmap_id: int
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        cached = self.cache.get(beatmap_id)
        if cached is None:
            return None
        if not metadata_complete(cached.metadata):
            self.cache.delete(beatmap_id)
            return None
        return cached.embedding, cached.metadata

    def infer_and_store(
        self, beatmap_id: int, content: bytes, metadata: dict[str, Any]
    ) -> np.ndarray:
        with self.inference_semaphore:
            raw = self.model.embed_osu_bytes(
                content, self.vector_stats, beatmap_id=beatmap_id
            )
        transformed = self._transform(raw)
        self.cache.upsert(beatmap_id, transformed, metadata)
        return transformed

    def _transform(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != self.transform.means.shape:
            raise ValueError(f"invalid embedding shape: {vector.shape}")
        transformed = self.model.transform(vector, self.transform)
        if (
            transformed.shape != (self.embedding_dim,)
            or not np.isfinite(transformed).all()
        ):
            raise ValueError("embedding transform returned an invalid vector")
        return transformed

    def summary(self, beatmap_id: int) -> dict[str, Any]:
        static_index = self.static_index.get(beatmap_id)
        if static_index is not None:
            metadata = self.static_metadata.row(static_index, named=True)
        else:
            cached = self.cache.get(beatmap_id)
            metadata = cached.metadata if cached is not None else {}
        return self.public_summary(beatmap_id, metadata)

    def star_rating(self, beatmap_id: int) -> float | None:
        index = int(np.searchsorted(self.strain_ids, beatmap_id))
        if index >= len(self.strain_ids) or self.strain_ids[index] != beatmap_id:
            return None
        return float(self.strain_stars[index])

    def public_summary(
        self, beatmap_id: int, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        summary = public_summary(beatmap_id, metadata)
        stars = self.star_rating(beatmap_id)
        if stars is not None:
            summary["stars"] = stars
        return summary

    def search(
        self,
        source_ids: list[int],
        query_embedding: np.ndarray,
        source_metadata: list[dict[str, Any]],
        top_k: int,
        filters: Any,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        source_set_ids = [
            beatmapset_id
            for metadata in source_metadata
            if (beatmapset_id := metadata_set_id(metadata)) is not None
        ]
        min_month = month_start_us(filters.min_date)
        max_month = month_end_exclusive_us(filters.max_date)
        eligible = ~np.isin(self.static_id_values, source_ids)
        if filters.exclude_same_set and source_set_ids:
            eligible &= ~np.isin(self.static_set_ids, source_set_ids)
        for column, minimum, maximum in (
            ("difficulty_rating", filters.min_sr, filters.max_sr),
            ("ar", filters.min_ar, filters.max_ar),
            ("cs", filters.min_cs, filters.max_cs),
            ("accuracy", filters.min_accuracy, filters.max_accuracy),
            ("drain", filters.min_drain, filters.max_drain),
            ("bpm", filters.min_bpm, filters.max_bpm),
            ("total_length", filters.min_length, filters.max_length),
        ):
            values = self.filter_values[column]
            if minimum is not None:
                eligible &= values >= minimum
            if maximum is not None:
                eligible &= values <= maximum
        if filters.status:
            status = filters.status.lower()
            status_mask = self.static_status_groups.get(status)
            if status_mask is None:
                accepted = list(STATUS_GROUPS.get(status, {status}))
                status_mask = np.isin(self.static_status_values[0], accepted) | np.isin(
                    self.static_status_values[1], accepted
                )
            eligible &= status_mask
        if min_month is not None:
            eligible &= self.static_release_dates >= min_month
        if max_month is not None:
            eligible &= self.static_release_dates < max_month

        eligible_indices = np.flatnonzero(eligible)
        if not len(eligible_indices):
            return []
        score_filtered = (
            len(eligible_indices) <= len(self.static_ids) * FILTERED_SCORE_THRESHOLD
        )
        if score_filtered:
            scores = np.empty(len(eligible_indices), dtype=np.float32)
            for start in range(0, len(eligible_indices), FILTERED_SCORE_BATCH_SIZE):
                stop = min(start + FILTERED_SCORE_BATCH_SIZE, len(eligible_indices))
                batch_indices = eligible_indices[start:stop]
                scores[start:stop] = (
                    self.static_embeddings[batch_indices] @ query_embedding
                )
            scores += self.static_penalty[eligible_indices]
            groups = self.static_set_groups[eligible_indices]
            best_scores = np.full(
                self.static_set_group_count, -np.inf, dtype=np.float32
            )
            np.maximum.at(best_scores, groups, scores)
            winning = scores == best_scores[groups]
            best_positions = np.full(
                self.static_set_group_count, -1, dtype=np.int32
            )
            np.maximum.at(best_positions, groups[winning], np.flatnonzero(winning))
            available_groups = np.flatnonzero(best_positions >= 0)
        else:
            ranking_scores = self.static_embeddings @ query_embedding
            ranking_scores += self.static_penalty
            ranking_scores[~eligible] = -np.inf
            best_scores = np.maximum.reduceat(
                ranking_scores[self.static_set_order], self.static_set_starts
            )
            available_groups = np.flatnonzero(np.isfinite(best_scores))
        result_count = min(top_k, len(available_groups))
        if result_count < len(available_groups):
            selected = np.argpartition(
                best_scores[available_groups], -result_count
            )[-result_count:]
            selected_groups = available_groups[selected]
        else:
            selected_groups = available_groups
        selected_groups = selected_groups[
            np.argsort(best_scores[selected_groups])[::-1]
        ]
        if score_filtered:
            selected_positions = best_positions[selected_groups]
            selected_indices = eligible_indices[selected_positions]
            selected_scores = scores[selected_positions]
        else:
            selected_indices = []
            for group in selected_groups:
                start = self.static_set_starts[group]
                stop = (
                    self.static_set_starts[group + 1]
                    if group + 1 < self.static_set_group_count
                    else len(self.static_set_order)
                )
                group_indices = self.static_set_order[start:stop]
                group_scores = ranking_scores[group_indices]
                tied = np.flatnonzero(group_scores == best_scores[group])
                selected_indices.append(group_indices[tied[-1]])
            selected_scores = ranking_scores[selected_indices]

        results = []
        for index, score in zip(selected_indices, selected_scores):
            beatmap_id = self.static_ids[index]
            result = self.public_summary(
                beatmap_id, self.static_metadata.row(index, named=True)
            )
            result["score"] = float(score - self.static_penalty[index])
            results.append(result)
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
            cached = self.cache.get(beatmap_id)
            if cached is None:
                return None
            beatmap = {"id": beatmap_id, **cached.metadata}
        stars = self.star_rating(beatmap_id)
        if stars is not None:
            beatmap["difficulty_rating"] = stars
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


def month_start_us(value: str | None) -> int | None:
    if not value:
        return None
    parsed = datetime.strptime(value, "%Y-%m").replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1_000_000)


def month_end_exclusive_us(value: str | None) -> int | None:
    if not value:
        return None
    parsed = datetime.strptime(value, "%Y-%m").replace(tzinfo=UTC)
    year = parsed.year + (parsed.month // 12)
    month = parsed.month % 12 + 1
    return int(datetime(year, month, 1, tzinfo=UTC).timestamp() * 1_000_000)


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
