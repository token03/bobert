from __future__ import annotations

import ctypes
import gc
import json
import os
import random
import sqlite3
import sys
import threading
import time
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isnan
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import torch

from core.artifacts import RunArtifacts, checksum, read_labels, validate_run
from core.index import (
    DENSITY_WEIGHT,
    PackedEmbeddings,
    Scorer,
    from_bfloat16,
    pack_embeddings,
    read_index,
    to_bfloat16,
    unpack_embeddings,
)
from core.model import BobertEncoder
from core.modes import MODES

RUN_DIR = Path(os.getenv("BOBERT_RUN_DIR", "/app/runs/current"))
DATA_DIR = Path(os.getenv("BOBERT_DATA_DIR", "/app/data"))
CACHE_DB = Path(os.getenv("BOBERT_CACHE_DB", "/app/cache/runtime.sqlite"))
BEATMAPS_PATH = DATA_DIR / "beatmaps.parquet"
STRAINS_PATH = DATA_DIR / "strains.parquet"
FILTERED_SCORE_THRESHOLD = 0.1
COPY_SIMILARITY = 0.95
COPY_CANDIDATES = 50
DEFAULT_COUNTS = (15, 40, 30, 15)
DEFAULT_ERA_STARTS = (2018, 2021, 2023)
DEFAULT_FAVOURITE_PERCENTILE = 0.75
DEFAULT_MIN_FAVOURITES = 150
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


FILTER_RANGES = {
    "difficulty_rating": ("min_sr", "max_sr"),
    "ar": ("min_ar", "max_ar"),
    "cs": ("min_cs", "max_cs"),
    "accuracy": ("min_accuracy", "max_accuracy"),
    "drain": ("min_drain", "max_drain"),
    "bpm": ("min_bpm", "max_bpm"),
    "total_length": ("min_length", "max_length"),
}


def _release_memory() -> None:
    gc.collect()
    pa.default_memory_pool().release_unused()
    if sys.platform == "linux":
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def _status_match_expr(accepted: Collection[str]) -> pl.Expr:
    values = list(accepted)
    status = (
        pl.col("status").cast(pl.String, strict=False).fill_null("").str.to_lowercase()
    )
    ranked = (
        pl.col("ranked").cast(pl.String, strict=False).fill_null("").str.to_lowercase()
    )
    return status.is_in(values) | ranked.is_in(values)


def _release_date_expr() -> pl.Expr:
    ranked = _status_match_expr(RANKED_STATUS_VALUES)
    return (
        pl.when(ranked & pl.col("ranked_date").is_not_null())
        .then(pl.col("ranked_date"))
        .otherwise(pl.col("submitted_date"))
        .cast(pl.String, strict=False)
        .str.to_datetime(strict=False, time_zone="UTC")
        .dt.epoch("us")
        .fill_null(np.iinfo(np.int64).min)
    )


@dataclass(frozen=True, slots=True)
class CachedEmbedding:
    beatmap_id: int
    embedding: np.ndarray
    density: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Candidates:
    ids: np.ndarray
    embeddings: PackedEmbeddings
    penalties: np.ndarray
    scorer: Scorer
    metadata: pl.DataFrame
    set_ids: np.ndarray
    set_groups: np.ndarray
    filter_values: dict[str, np.ndarray]
    status_groups: dict[str, np.ndarray]
    release_dates: np.ndarray

    @classmethod
    def build(
        cls,
        embeddings: PackedEmbeddings,
        penalties: np.ndarray,
        metadata: pl.DataFrame,
        scorer: Scorer,
    ) -> Candidates:
        set_ids = metadata["beatmapset_id"].fill_null(-1).to_numpy().astype(np.int64)
        columns = metadata.select(
            *(
                pl.col(column).cast(pl.Float32, strict=False).fill_null(float("nan"))
                for column in FILTER_RANGES
            ),
            *(
                _status_match_expr(accepted).alias(group)
                for group, accepted in STATUS_GROUPS.items()
            ),
            _release_date_expr().alias("release_date"),
        )
        return cls(
            ids=metadata["id"].to_numpy(),
            embeddings=embeddings,
            penalties=-penalties.astype(np.float32),
            scorer=scorer,
            metadata=metadata,
            set_ids=set_ids,
            set_groups=np.unique(set_ids, return_inverse=True)[1],
            filter_values={
                column: columns[column].to_numpy() for column in FILTER_RANGES
            },
            status_groups={group: columns[group].to_numpy() for group in STATUS_GROUPS},
            release_dates=columns["release_date"].to_numpy(),
        )

    def vector(self, row: int) -> np.ndarray:
        return unpack_embeddings(self.embeddings, np.array([row]))[0]

    def status_mask(self, status: str) -> np.ndarray:
        if status in self.status_groups:
            return self.status_groups[status]
        return self.metadata.select(_status_match_expr({status}))[:, 0].to_numpy()

    def eligible(
        self, filters: Any, source_ids: list[int], source_set_ids: list[int]
    ) -> np.ndarray:
        eligible = (self.set_ids != -1) & ~np.isin(self.ids, source_ids)
        if filters.exclude_same_set:
            eligible &= ~np.isin(self.set_ids, source_set_ids)
        for column, (minimum_field, maximum_field) in FILTER_RANGES.items():
            minimum = getattr(filters, minimum_field)
            maximum = getattr(filters, maximum_field)
            if minimum is not None:
                eligible &= self.filter_values[column] >= minimum
            if maximum is not None:
                eligible &= self.filter_values[column] <= maximum
        if filters.status:
            eligible &= self.status_mask(filters.status.lower())
        if filters.min_date:
            eligible &= self.release_dates >= month_start_us(filters.min_date)
        if filters.max_date:
            eligible &= self.release_dates < month_end_exclusive_us(filters.max_date)
        return eligible

    def best_per_set(
        self, query: np.ndarray, eligible: np.ndarray, count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.flatnonzero(eligible)
        if len(rows) > len(self.ids) * FILTERED_SCORE_THRESHOLD:
            scores = self.scorer(self.embeddings, query)[rows]
        else:
            scores = self.scorer(self.embeddings, query, rows)
        scores += self.penalties[rows]
        groups = self.set_groups[rows]
        best = np.full(self.set_groups.max(initial=0) + 1, -np.inf, dtype=np.float32)
        np.maximum.at(best, groups, scores)
        available = np.flatnonzero(np.isfinite(best))
        if count < len(available):
            available = available[np.argpartition(best[available], -count)[-count:]]
        top = available[np.argsort(best[available])[::-1]]
        winners = np.full(len(best), -1, dtype=np.int64)
        winning = np.isin(groups, top) & (scores == best[groups])
        np.maximum.at(winners, groups[winning], rows[winning])
        return winners[top], best[top]


def mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("embeddings must be a non-empty matrix")
    combined = vectors.mean(axis=0)
    norm = float(np.linalg.norm(combined))
    if not np.isfinite(combined).all() or not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("combined embedding is invalid")
    return np.asarray(combined / norm, dtype=np.float32)


def _cached_embedding(
    beatmap_id: int, blob: bytes, density: float, metadata_json: str
) -> CachedEmbedding:
    return CachedEmbedding(
        int(beatmap_id),
        from_bfloat16(np.frombuffer(blob, dtype=np.uint16)),
        float(density),
        json.loads(metadata_json),
    )


class SQLiteCache:
    def __init__(self, path: Path, run_id: str):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = conn.execute(
                "SELECT value FROM cache_metadata WHERE key = 'run_id'"
            ).fetchone()
            if row is not None and row[0] != run_id:
                conn.execute("DROP TABLE IF EXISTS embeddings")
                conn.execute("DROP TABLE IF EXISTS unavailable_beatmaps")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (beatmap_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL, density REAL NOT NULL, metadata_json TEXT NOT NULL, created_at INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS unavailable_beatmaps (beatmap_id INTEGER PRIMARY KEY, reason TEXT NOT NULL, created_at INTEGER NOT NULL)"
            )
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
                "SELECT embedding, density, metadata_json FROM embeddings WHERE beatmap_id = ?",
                (int(beatmap_id),),
            ).fetchone()
        if row is None:
            return None
        return _cached_embedding(beatmap_id, *row)

    def list_all(self) -> list[CachedEmbedding]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT beatmap_id, embedding, density, metadata_json FROM embeddings"
            ).fetchall()
        return [_cached_embedding(*row) for row in rows]

    def upsert(self, entry: CachedEmbedding) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (beatmap_id, embedding, density, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    entry.beatmap_id,
                    to_bfloat16(entry.embedding).tobytes(),
                    entry.density,
                    json.dumps(entry.metadata, separators=(",", ":")),
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


class Catalog:
    def __init__(self):
        strains = (
            pl.scan_parquet(STRAINS_PATH)
            .select("beatmap_id", "actual_stars")
            .filter(pl.col("actual_stars").is_not_null())
            .sort("beatmap_id")
            .collect()
        )
        self.strain_ids = strains["beatmap_id"].to_numpy()
        self.strain_stars = strains["actual_stars"].to_numpy()
        self._strains_join = strains.select(
            pl.col("beatmap_id").alias("id"), "actual_stars"
        ).lazy()

    def source_modes(self) -> pl.DataFrame:
        return pl.scan_parquet(BEATMAPS_PATH).select("id", "mode").collect()

    def mode_beatmaps(self, mode_label: str) -> pl.DataFrame:
        return (
            pl.scan_parquet(BEATMAPS_PATH)
            .select(SEARCH_COLUMNS)
            .filter(pl.col("mode") == mode_label)
            .join(self._strains_join, on="id", how="left")
            .with_columns(
                pl.coalesce("actual_stars", "difficulty_rating").alias(
                    "difficulty_rating"
                )
            )
            .drop("actual_stars")
            .collect()
        )


class Runtime:
    def __init__(self, artifacts: RunArtifacts, catalog: Catalog):
        self.mode = artifacts.mode
        self.index = read_index(artifacts)
        self.labels = read_labels(artifacts)
        self.static_ids = [int(value) for value in self.index.ids]
        self.static_index = {
            beatmap_id: row for row, beatmap_id in enumerate(self.static_ids)
        }

        self.strain_ids = catalog.strain_ids
        self.strain_stars = catalog.strain_stars
        beatmaps = catalog.mode_beatmaps(MODES[self.mode].label)
        defaults = (
            beatmaps.filter(
                pl.col("status").is_in(["1", "4"]),
                pl.col("mode") == MODES[self.mode].label,
            )
            .sort("difficulty_rating", descending=True)
            .unique("beatmapset_id", keep="first", maintain_order=True)
            .join(pl.DataFrame({"id": self.static_ids}), on="id", how="semi")
            .filter(pl.col("difficulty_rating").is_between(5, 9, closed="left"))
            .with_columns(
                pl.coalesce("ranked_date", "submitted_date")
                .cast(pl.String)
                .str.slice(0, 4)
                .cast(pl.Int32, strict=False)
                .alias("_year")
            )
            .filter(
                pl.col("favourite_count").rank("average").over("_year")
                / pl.len().over("_year")
                >= DEFAULT_FAVOURITE_PERCENTILE,
                pl.col("favourite_count") >= DEFAULT_MIN_FAVOURITES,
            )
            .with_columns(
                pl.sum_horizontal(
                    pl.col("_year") >= start for start in DEFAULT_ERA_STARTS
                ).alias("_era")
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
            .with_columns(pl.lit(MODES[self.mode].label).alias("mode"))
        )
        self.static = Candidates.build(
            self.index.packed,
            self.index.penalties,
            self.static_metadata,
            self.index.scorer,
        )
        self.fallback_defaults = [
            self.public_summary(int(row["id"]), row)
            for row in beatmaps.join(
                pl.DataFrame({"id": self.static_ids}), on="id", how="semi"
            )
            .filter(pl.col("status").is_in(["1", "4", "ranked", "loved"]))
            .sort("favourite_count", descending=True, nulls_last=True)
            .unique("beatmapset_id", keep="first", maintain_order=True)
            .head(500)
            .iter_rows(named=True)
        ]
        self.default_pools = [
            [
                [
                    self.public_summary(int(row["id"]), row)
                    for row in defaults.filter(
                        pl.col("difficulty_rating").floor() == star,
                        pl.col("_era") == era,
                    ).iter_rows(named=True)
                ]
                for era in range(len(DEFAULT_ERA_STARTS) + 1)
            ]
            for star in range(5, 9)
        ]
        del beatmaps
        self.inference_semaphore = threading.Semaphore(1)
        cache_path = CACHE_DB.with_name(f"{CACHE_DB.stem}-{self.mode}{CACHE_DB.suffix}")
        self.cache = SQLiteCache(cache_path, checksum(artifacts.model))
        self._dynamic_lock = threading.Lock()
        self._dynamic_entries: dict[int, CachedEmbedding] = {}
        self.dynamic: Candidates | None = None
        self._hydrate_dynamic()

        self.artifacts = artifacts
        self._model = None

    def _hydrate_dynamic(self) -> None:
        entries = []
        for entry in self.cache.list_all():
            if metadata_complete(entry.metadata):
                entries.append(entry)
            else:
                self.cache.delete(entry.beatmap_id)
        self._add_dynamic(entries)

    def _add_dynamic(self, entries: list[CachedEmbedding]) -> None:
        with self._dynamic_lock:
            for entry in entries:
                if entry.beatmap_id not in self.static_index:
                    self._dynamic_entries[entry.beatmap_id] = entry
            if not self._dynamic_entries:
                return
            cached = list(self._dynamic_entries.values())
            metadata = pl.DataFrame(
                [{**entry.metadata, "id": entry.beatmap_id} for entry in cached],
                schema=self.static.metadata.schema,
                strict=False,
            ).with_columns(pl.lit(MODES[self.mode].label).alias("mode"))
            self.dynamic = Candidates.build(
                pack_embeddings(
                    np.stack([entry.embedding for entry in cached]), self.index.format
                ),
                DENSITY_WEIGHT
                * np.array([entry.density for entry in cached], dtype=np.float32),
                metadata,
                self.index.scorer,
            )

    @property
    def model(self) -> BobertEncoder:
        if self._model is None:
            device = torch.device("cpu")
            self._model = BobertEncoder.from_pretrained(
                self.artifacts.directory, device
            )
        return self._model

    def validate_source(self, beatmap_id: int, metadata: dict[str, Any]) -> None:
        if metadata_mode(metadata) != self.mode:
            raise ValueError(
                f"beatmap {beatmap_id} belongs to {metadata_mode(metadata)}, not {self.mode}"
            )

    def default_summaries(self, seed: int | None = None) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        results = []
        for eras, count in zip(self.default_pools, DEFAULT_COUNTS):
            queues = [rng.sample(pool, len(pool)) for pool in eras]
            rng.shuffle(queues)
            picked = 0
            while picked < count and any(queues):
                for queue in queues:
                    if queue and picked < count:
                        results.append(queue.pop())
                        picked += 1
        if len(results) < sum(DEFAULT_COUNTS):
            used = {row["beatmapset_id"] for row in results}
            remaining = [
                row
                for row in self.fallback_defaults
                if row["beatmapset_id"] not in used
            ]
            results.extend(
                rng.sample(
                    remaining, min(len(remaining), sum(DEFAULT_COUNTS) - len(results))
                )
            )
        rng.shuffle(results)
        return results

    def memory_embedding(
        self, beatmap_id: int
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        static_index = self.static_index.get(beatmap_id)
        if static_index is not None:
            return self.static.vector(static_index), self.static_metadata.row(
                static_index, named=True
            )
        return None

    def unindexed(self, beatmap_ids: list[int]) -> list[int]:
        return [
            beatmap_id
            for beatmap_id in beatmap_ids
            if beatmap_id not in self.static_index
            and self.cache.get(beatmap_id) is None
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
        return cached.embedding, cached.metadata

    def infer_and_store(
        self, beatmap_id: int, content: bytes, metadata: dict[str, Any]
    ) -> np.ndarray:
        self.validate_source(beatmap_id, metadata)
        with self.inference_semaphore:
            raw = self.model.embed_osu_bytes(content, beatmap_id=beatmap_id)
        if not np.isfinite(raw).all():
            raise ValueError(f"beatmap {beatmap_id} produced an invalid embedding")
        embedding = from_bfloat16(to_bfloat16(raw))
        entry = CachedEmbedding(
            beatmap_id,
            embedding,
            self.index.density(embedding),
            metadata,
        )
        self.cache.upsert(entry)
        self._add_dynamic([entry])
        return embedding

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
        distinct: bool,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        wanted = offset + top_k
        source_set_ids = [
            beatmapset_id
            for metadata in source_metadata
            if (beatmapset_id := metadata_set_id(metadata)) is not None
        ]
        count = wanted + COPY_CANDIDATES if distinct else wanted
        best_by_set: dict[int, tuple[float, Candidates, int]] = {}
        for candidates in (self.static, self.dynamic):
            if candidates is None:
                continue
            eligible = candidates.eligible(filters, source_ids, source_set_ids)
            rows, scores = candidates.best_per_set(query_embedding, eligible, count)
            for row, score in zip(rows.tolist(), scores.tolist()):
                set_id = int(candidates.set_ids[row])
                if set_id not in best_by_set or score > best_by_set[set_id][0]:
                    best_by_set[set_id] = (score, candidates, row)
        hits = sorted(best_by_set.values(), key=lambda hit: hit[0], reverse=True)
        hits = hits[:count]
        if distinct and hits:
            vectors = np.stack([candidates.vector(row) for _, candidates, row in hits])
            ranked = np.array(
                [candidates.status_groups["ranked"][row] for _, candidates, row in hits]
            )
            hits = [
                hits[i]
                for i in distinct_positions(query_embedding, vectors, ranked, wanted)
            ]

        results = []
        for score, candidates, row in hits[offset:wanted]:
            result = self.public_summary(
                int(candidates.ids[row]), candidates.metadata.row(row, named=True)
            )
            result["score"] = float(score - candidates.penalties[row])
            results.append(result)
        return results


def distinct_positions(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    ranked: np.ndarray,
    top_k: int,
) -> list[int]:
    copies_query = embeddings @ query_embedding >= COPY_SIMILARITY
    kept: list[int] = []
    for position in sorted(range(len(embeddings)), key=lambda i: not ranked[i]):
        if copies_query[position]:
            continue
        if kept and (embeddings[kept] @ embeddings[position]).max() >= COPY_SIMILARITY:
            continue
        kept.append(position)
    return sorted(kept)[:top_k]


class RuntimeRegistry:
    def __init__(self):
        selected = validate_run(RUN_DIR)
        catalog = Catalog()
        self.source_modes = catalog.source_modes()
        self.runtimes = {}
        inference_semaphore = threading.Semaphore(1)
        for mode, artifacts in selected.items():
            self.runtimes[mode] = Runtime(artifacts, catalog)
            self.runtimes[mode].inference_semaphore = inference_semaphore
            _release_memory()
        del catalog
        _release_memory()

    def validate_sources(self, mode: str, ids: list[int]) -> None:
        known = self.source_modes.filter(pl.col("id").is_in(ids))
        wrong = known.filter(pl.col("mode") != MODES[mode].label)
        if wrong.height:
            row = wrong.row(0, named=True)
            raise ValueError(
                f"beatmap {row['id']} belongs to {metadata_mode(row)}, not {mode}"
            )

    def source_mode(self, beatmap_id: int) -> str | None:
        rows = self.source_modes.filter(pl.col("id") == beatmap_id)
        if rows.height:
            return metadata_mode(rows.row(0, named=True))
        for mode, runtime in self.runtimes.items():
            if beatmap_id in runtime.static_index:
                return mode
            cached = runtime.cache.get(beatmap_id)
            if cached is not None:
                return metadata_mode(cached.metadata)
        return None

    def get(self, mode: str) -> Runtime:
        if mode not in self.runtimes:
            raise ValueError(f"mode {mode} is unavailable")
        return self.runtimes[mode]


def metadata_mode(metadata: dict[str, Any]) -> str:
    value = metadata.get("mode")
    for name, mode in MODES.items():
        if value in (name, mode.label, mode.mode_int):
            return name
    raise ValueError("beatmap metadata has no supported mode")


def public_summary(beatmap_id: int, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "beatmap_id": beatmap_id,
        "mode": metadata_mode(metadata),
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
        "play_count": json_value(metadata.get("play_count")),
        "favourite_count": json_value(metadata.get("favourite_count")),
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
