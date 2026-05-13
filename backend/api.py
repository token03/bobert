from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from math import isnan
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl
import torch
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.cache import RuntimeCache
from backend.inference import CpuInferencer
from backend.osu import fetch_osu_file, parse_osu_metadata


DATA_DIR = Path(os.getenv("BOBERT_DATA_DIR", "/app/data"))
CACHE_DB = Path(os.getenv("BOBERT_CACHE_DB", "/app/cache/runtime.sqlite"))
MODEL_PATH = Path(os.getenv("BOBERT_MODEL_PATH", DATA_DIR / "bobert.pt"))
CONFIG_PATH = Path(os.getenv("BOBERT_CONFIG_PATH", "/app/config.api.yaml"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "")
API_SHARED_SECRET = os.getenv("API_SHARED_SECRET", "")

DEFAULT_RATE_LIMITS = {
    "global_recommend_per_hour": 600,
    "server_recommend_per_hour": 300,
    "ip_recommend_per_hour": 10,
}

torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))


class RecommendFilters(BaseModel):
    min_sr: float | None = Field(default=None, ge=0)
    max_sr: float | None = Field(default=None, ge=0)
    min_ar: float | None = Field(default=None, ge=0)
    max_ar: float | None = Field(default=None, ge=0)
    min_cs: float | None = Field(default=None, ge=0)
    max_cs: float | None = Field(default=None, ge=0)
    min_accuracy: float | None = Field(default=None, ge=0)
    max_accuracy: float | None = Field(default=None, ge=0)
    min_drain: float | None = Field(default=None, ge=0)
    max_drain: float | None = Field(default=None, ge=0)
    status: str | None = Field(default=None, max_length=32)
    exclude_same_set: bool = True


class RecommendRequest(BaseModel):
    beatmap_id: int = Field(gt=0)
    top_k: int = Field(default=20, ge=1, le=50)
    filters: RecommendFilters = Field(default_factory=RecommendFilters)


@dataclass
class Runtime:
    embedding_ids: list[int]
    embeddings: np.ndarray
    id_to_index: dict[int, int]
    metadata_by_id: dict[int, dict[str, Any]]
    cache: RuntimeCache
    inferencer: CpuInferencer
    lock: threading.Lock


_RUNTIME: Runtime | None = None
_RUNTIME_LOCK = threading.Lock()
_RATE_LIMITS: dict[str, tuple[int, int]] = {}


def load_rate_limit_config() -> dict[str, int]:
    if not CONFIG_PATH.exists():
        return DEFAULT_RATE_LIMITS.copy()

    with CONFIG_PATH.open() as f:
        config = yaml.safe_load(f) or {}

    configured = config.get("api", {}).get("rate_limits", {}) or {}
    return {
        key: int(configured.get(key, default))
        for key, default in DEFAULT_RATE_LIMITS.items()
    }


RATE_LIMIT_CONFIG = load_rate_limit_config()


app = FastAPI(title="bobert-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Turnstile-Token", "X-API-Key"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost",
        "127.0.0.1",
        "bobert.jessiezhong.com",
        "*.trycloudflare.com",
    ],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"ok": "true"}


@app.post("/recommend")
async def recommend(
    payload: RecommendRequest,
    request: Request,
    x_turnstile_token: str | None = Header(default=None, alias="X-Turnstile-Token"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    ip = client_ip(request)
    trusted_server = bool(API_SHARED_SECRET) and x_api_key == API_SHARED_SECRET

    rate_limit(
        "global:recommend",
        limit=RATE_LIMIT_CONFIG["global_recommend_per_hour"],
        window_seconds=3600,
    )
    if trusted_server:
        rate_limit(
            "server:recommend",
            limit=RATE_LIMIT_CONFIG["server_recommend_per_hour"],
            window_seconds=3600,
        )
    else:
        rate_limit(
            f"ip:{ip}:recommend",
            limit=RATE_LIMIT_CONFIG["ip_recommend_per_hour"],
            window_seconds=3600,
        )
        await verify_turnstile(x_turnstile_token, ip)

    rt = get_runtime()
    query_embedding, cache_status, query_metadata = await get_query_embedding(
        rt, payload.beatmap_id
    )
    results = await run_in_threadpool(
        search,
        rt,
        payload.beatmap_id,
        query_embedding,
        query_metadata,
        payload.top_k,
        payload.filters,
    )
    return {
        "query": {
            "beatmap_id": payload.beatmap_id,
            "cache": cache_status,
            "metadata": public_metadata(payload.beatmap_id, query_metadata),
        },
        "count": len(results),
        "results": results,
    }


def get_runtime() -> Runtime:
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME

    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME

        beatmaps_path = DATA_DIR / "beatmaps.parquet"
        beatmapsets_path = DATA_DIR / "beatmapsets.parquet"
        embeddings_path = DATA_DIR / "embeddings.parquet"
        missing = [
            str(path)
            for path in [beatmaps_path, beatmapsets_path, embeddings_path, MODEL_PATH]
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(f"missing required data files: {missing}")

        beatmaps = pl.read_parquet(beatmaps_path)
        embeddings_df = pl.read_parquet(embeddings_path)
        if "beatmap_id" not in embeddings_df.columns or "embedding" not in embeddings_df.columns:
            raise RuntimeError("embeddings.parquet must contain beatmap_id and embedding")

        embedding_ids = [int(x) for x in embeddings_df["beatmap_id"].to_list()]
        embeddings = normalize_rows(
            np.asarray(embeddings_df["embedding"].to_list(), dtype=np.float32)
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(embedding_ids):
            raise RuntimeError("invalid embeddings.parquet shape")

        id_to_index = {beatmap_id: idx for idx, beatmap_id in enumerate(embedding_ids)}
        metadata_by_id = load_metadata_lookup(beatmaps)
        cache = RuntimeCache(CACHE_DB, embedding_dim=embeddings.shape[1])

        cached = [item for item in cache.load_all() if item.beatmap_id not in id_to_index]
        if cached:
            start = len(embedding_ids)
            embedding_ids.extend(item.beatmap_id for item in cached)
            embeddings = np.vstack([embeddings, [item.embedding for item in cached]]).astype(
                np.float32
            )
            id_to_index.update(
                {item.beatmap_id: start + idx for idx, item in enumerate(cached)}
            )
            metadata_by_id.update({item.beatmap_id: item.metadata for item in cached})

        _RUNTIME = Runtime(
            embedding_ids=embedding_ids,
            embeddings=embeddings,
            id_to_index=id_to_index,
            metadata_by_id=metadata_by_id,
            cache=cache,
            inferencer=CpuInferencer(CONFIG_PATH, MODEL_PATH),
            lock=threading.Lock(),
        )
        return _RUNTIME


async def get_query_embedding(
    rt: Runtime, beatmap_id: int
) -> tuple[np.ndarray, str, dict[str, Any]]:
    with rt.lock:
        idx = rt.id_to_index.get(int(beatmap_id))
        if idx is not None:
            return rt.embeddings[idx], "hit", rt.metadata_by_id.get(int(beatmap_id), {})

    cached = rt.cache.get(beatmap_id)
    if cached is not None:
        with rt.lock:
            append_embedding(rt, cached.beatmap_id, cached.embedding, cached.metadata)
        return cached.embedding, "hit", cached.metadata

    osu_bytes = await fetch_osu_file(beatmap_id)
    metadata = parse_osu_metadata(osu_bytes, beatmap_id)
    embedding = await run_in_threadpool(rt.inferencer.embed_osu_bytes, osu_bytes)

    rt.cache.upsert(beatmap_id, embedding, metadata)
    with rt.lock:
        append_embedding(rt, beatmap_id, embedding, metadata)
    return embedding, "miss", metadata


def append_embedding(
    rt: Runtime, beatmap_id: int, embedding: np.ndarray, metadata: dict[str, Any]
) -> None:
    if beatmap_id in rt.id_to_index:
        rt.metadata_by_id.setdefault(beatmap_id, metadata)
        return
    rt.id_to_index[beatmap_id] = len(rt.embedding_ids)
    rt.embedding_ids.append(beatmap_id)
    rt.embeddings = np.vstack([rt.embeddings, normalize_rows(embedding[None, :])]).astype(
        np.float32
    )
    rt.metadata_by_id[beatmap_id] = metadata


def search(
    rt: Runtime,
    query_beatmap_id: int,
    query_embedding: np.ndarray,
    query_metadata: dict[str, Any],
    top_k: int,
    filters: RecommendFilters,
) -> list[dict[str, Any]]:
    with rt.lock:
        embedding_ids = list(rt.embedding_ids)
        embeddings = rt.embeddings.copy()
        metadata_by_id = dict(rt.metadata_by_id)

    scores = embeddings @ query_embedding
    query_set_id = metadata_set_id(query_metadata)
    seen_set_ids: set[int] = set()
    results = []

    for idx in np.argsort(-scores):
        beatmap_id = int(embedding_ids[int(idx)])
        if beatmap_id == query_beatmap_id:
            continue

        metadata = metadata_by_id.get(beatmap_id, {})
        candidate_set_id = metadata_set_id(metadata)
        if not passes_filters(metadata, filters):
            continue
        if (
            filters.exclude_same_set
            and query_set_id is not None
            and candidate_set_id == query_set_id
        ):
            continue
        if candidate_set_id is not None and candidate_set_id in seen_set_ids:
            continue

        result = public_metadata(beatmap_id, metadata)
        result["score"] = float(scores[int(idx)])
        results.append(result)
        if candidate_set_id is not None:
            seen_set_ids.add(candidate_set_id)
        if len(results) >= top_k:
            break

    return results


def load_metadata_lookup(beatmaps: pl.DataFrame) -> dict[int, dict[str, Any]]:
    id_col = "id" if "id" in beatmaps.columns else "beatmap_id"
    if id_col not in beatmaps.columns:
        return {}
    return {
        int(row[id_col]): row
        for row in beatmaps.unique(id_col, maintain_order=True).iter_rows(named=True)
    }


def public_metadata(beatmap_id: int, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "beatmap_id": int(beatmap_id),
        "beatmapset_id": json_value(metadata.get("beatmapset_id")),
        "artist": json_value(metadata.get("artist")),
        "title": json_value(metadata.get("title")),
        "creator": json_value(metadata.get("creator")),
        "version": json_value(metadata.get("version")),
        "status": json_value(metadata.get("status")),
        "stars": json_value(metadata.get("difficulty_rating", metadata.get("stars"))),
        "ar": json_value(metadata.get("ar")),
        "cs": json_value(metadata.get("cs")),
        "accuracy": json_value(metadata.get("accuracy")),
        "drain": json_value(metadata.get("drain")),
        "bpm": json_value(metadata.get("bpm")),
        "total_length": json_value(metadata.get("total_length")),
        "url": json_value(metadata.get("url")) or f"https://osu.ppy.sh/b/{int(beatmap_id)}",
    }


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and isnan(value):
        return None
    return value


def passes_filters(metadata: dict[str, Any], filters: RecommendFilters) -> bool:
    stars = metadata.get("difficulty_rating", metadata.get("stars"))
    if filters.min_sr is not None and (stars is None or float(stars) < filters.min_sr):
        return False
    if filters.max_sr is not None and (stars is None or float(stars) > filters.max_sr):
        return False
    if not passes_range(metadata.get("ar"), filters.min_ar, filters.max_ar):
        return False
    if not passes_range(metadata.get("cs"), filters.min_cs, filters.max_cs):
        return False
    if not passes_range(
        metadata.get("accuracy"), filters.min_accuracy, filters.max_accuracy
    ):
        return False
    if not passes_range(metadata.get("drain"), filters.min_drain, filters.max_drain):
        return False
    if filters.status:
        status_values = {
            str(metadata.get("status", "")).lower(),
            str(metadata.get("ranked", "")).lower(),
        }
        if filters.status.lower() not in status_values:
            return False
    return True


def passes_range(value: Any, minimum: float | None, maximum: float | None) -> bool:
    if minimum is None and maximum is None:
        return True
    if value is None:
        return False
    value = float(value)
    if isnan(value):
        return False
    if minimum is not None and value < minimum:
        return False
    if maximum is not None and value > maximum:
        return False
    return True


def metadata_set_id(metadata: dict[str, Any] | None) -> int | None:
    value = (metadata or {}).get("beatmapset_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-9, None)


def client_ip(request: Request) -> str:
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(key: str, limit: int, window_seconds: int) -> None:
    now = int(time.time())
    bucket = now // window_seconds
    current_bucket, count = _RATE_LIMITS.get(key, (bucket, 0))
    if current_bucket != bucket:
        current_bucket, count = bucket, 0
    count += 1
    _RATE_LIMITS[key] = (current_bucket, count)

    if len(_RATE_LIMITS) > 10000:
        stale = [k for k, (b, _) in _RATE_LIMITS.items() if b != bucket]
        for stale_key in stale[:1000]:
            _RATE_LIMITS.pop(stale_key, None)

    if count > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


async def verify_turnstile(token: str | None, remote_ip: str) -> None:
    if not TURNSTILE_SECRET_KEY:
        return
    if not token:
        raise HTTPException(status_code=401, detail="Missing Turnstile token")

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data={
                "secret": TURNSTILE_SECRET_KEY,
                "response": token,
                "remoteip": remote_ip,
            },
        )

    data = response.json()
    if not data.get("success"):
        raise HTTPException(status_code=401, detail="Turnstile verification failed")
