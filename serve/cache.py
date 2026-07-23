from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CachedEmbedding:
    beatmap_id: int
    embedding: np.ndarray
    metadata: dict[str, Any]


class RuntimeCache:
    def __init__(self, path: Path, embedding_dim: int):
        self.path = path
        self.embedding_dim = int(embedding_dim)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    beatmap_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS unavailable_beatmaps (
                    beatmap_id INTEGER PRIMARY KEY,
                    reason TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )

    def load_all(self) -> list[CachedEmbedding]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT beatmap_id, embedding, metadata_json FROM embeddings"
            ).fetchall()

        cached = []
        for beatmap_id, blob, metadata_json in rows:
            embedding = np.frombuffer(blob, dtype=np.float32).copy()
            if embedding.shape != (self.embedding_dim,):
                continue
            cached.append(
                CachedEmbedding(
                    beatmap_id=int(beatmap_id),
                    embedding=embedding,
                    metadata=json.loads(metadata_json or "{}"),
                )
            )
        return cached

    def get(self, beatmap_id: int) -> CachedEmbedding | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT embedding, metadata_json
                FROM embeddings
                WHERE beatmap_id = ?
                """,
                (int(beatmap_id),),
            ).fetchone()
        if row is None:
            return None

        embedding = np.frombuffer(row[0], dtype=np.float32).copy()
        if embedding.shape != (self.embedding_dim,):
            return None
        return CachedEmbedding(
            beatmap_id=int(beatmap_id),
            embedding=embedding,
            metadata=json.loads(row[1] or "{}"),
        )

    def upsert(
        self, beatmap_id: int, embedding: np.ndarray, metadata: dict[str, Any]
    ) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (self.embedding_dim,):
            raise ValueError(
                f"embedding shape {vector.shape} does not match ({self.embedding_dim},)"
            )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                    (beatmap_id, embedding, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    int(beatmap_id),
                    vector.tobytes(),
                    json.dumps(metadata, separators=(",", ":")),
                    int(time.time()),
                ),
            )

    def delete(self, beatmap_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM embeddings WHERE beatmap_id = ?",
                (int(beatmap_id),),
            )

    def is_unavailable(self, beatmap_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM unavailable_beatmaps WHERE beatmap_id = ?",
                (int(beatmap_id),),
            ).fetchone()
        return row is not None

    def mark_unavailable(self, beatmap_id: int, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO unavailable_beatmaps
                    (beatmap_id, reason, created_at)
                VALUES (?, ?, ?)
                """,
                (int(beatmap_id), str(reason), int(time.time())),
            )
