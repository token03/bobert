from __future__ import annotations

import multiprocessing
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from math import isnan
from pathlib import Path
from typing import Any

CPU_COUNT = multiprocessing.cpu_count() or 1
DEFAULT_THREAD_COUNT = min(2, CPU_COUNT)
THREAD_COUNT = int(os.getenv("TORCH_NUM_THREADS", str(DEFAULT_THREAD_COUNT)))
os.environ.setdefault("OMP_NUM_THREADS", str(THREAD_COUNT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREAD_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(THREAD_COUNT))
os.environ.setdefault("POLARS_MAX_THREADS", str(THREAD_COUNT))

import httpx
import numpy as np
from ossapi import Ossapi
import polars as pl
import torch
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.cache import CachedEmbedding, RuntimeCache
from backend.inference import CpuInferencer
from backend.logging import (
    RequestLoggingMiddleware,
    async_timed,
    configure_logging,
    get_logger,
    request_client_ip,
    timed,
    timed_call,
)
from backend.osu import close_osu_http_client, fetch_osu_file, open_osu_http_client


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
DEFAULT_MAX_RECOMMEND_TOP_K = 200
DEFAULT_RECOMMEND_BEATMAP_IDS: list[int] = []
OSU_TOKEN_REFRESH_SKEW_SECONDS = 300
RANKED_STATUS_VALUES = {"1", "2", "3", "4", "ranked", "approved", "qualified", "loved"}
BEATMAP_METADATA_COLUMNS = [
    "id",
    "beatmapset_id",
    "user_id",
    "artist",
    "title",
    "creator",
    "version",
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
]


class DateWindow(str, Enum):
    last_week = "last_week"
    last_month = "last_month"
    last_3_months = "last_3_months"
    last_6_months = "last_6_months"
    last_year = "last_year"
    last_2_years = "last_2_years"
    last_5_years = "last_5_years"
    all_time = "all_time"


def load_api_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}

    with CONFIG_PATH.open() as f:
        config = yaml.safe_load(f) or {}

    return config.get("api", {}) or {}


API_CONFIG = load_api_config()
MAX_RECOMMEND_TOP_K = int(
    API_CONFIG.get("max_recommend_top_k", DEFAULT_MAX_RECOMMEND_TOP_K)
)
DEFAULT_RECOMMEND_IDS = [
    int(beatmap_id)
    for beatmap_id in API_CONFIG.get(
        "default_recommend_beatmap_ids", DEFAULT_RECOMMEND_BEATMAP_IDS
    )
]
OSU_API_VERSION = str(API_CONFIG.get("osu_api_version", "20241024"))

torch.set_num_threads(THREAD_COUNT)
configure_logging()
log = get_logger("bobert.api")


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
    min_bpm: float | None = Field(default=None, ge=0)
    max_bpm: float | None = Field(default=None, ge=0)
    min_length: float | None = Field(default=None, ge=0)
    max_length: float | None = Field(default=None, ge=0)
    status: str | None = Field(default=None, max_length=32)
    date_window: DateWindow | None = None
    exclude_same_set: bool = True


class RecommendRequest(BaseModel):
    beatmap_id: int = Field(gt=0)
    top_k: int = Field(default=20, ge=1, le=MAX_RECOMMEND_TOP_K)
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
    dynamic_embeddings: list[np.ndarray] = field(default_factory=list)
    dynamic_embedding_ids: list[int] = field(default_factory=list)


_RUNTIME: Runtime | None = None
_RUNTIME_LOCK = threading.Lock()
_RATE_LIMITS: dict[str, tuple[int, int]] = {}
_OSU_API: Ossapi | None = None
_OSU_API_LOCK = threading.Lock()
_HTTP_CLIENT: httpx.AsyncClient | None = None
_INFERENCE_SEMAPHORE = threading.Semaphore(1)

SEARCH_CANDIDATE_FACTORS = (10, 25, 100)


class BeatmapUnavailableError(ValueError):
    pass


def load_rate_limit_config() -> dict[str, int]:
    configured = API_CONFIG.get("rate_limits", {}) or {}
    return {
        key: int(configured.get(key, default))
        for key, default in DEFAULT_RATE_LIMITS.items()
    }


RATE_LIMIT_CONFIG = load_rate_limit_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _HTTP_CLIENT
    _HTTP_CLIENT = httpx.AsyncClient(
        timeout=5.0,
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
    )
    open_osu_http_client()
    try:
        rt = await run_in_threadpool(get_runtime)
        await run_in_threadpool(rt.inferencer.load)
        await run_in_threadpool(get_osu_api)
        yield
    finally:
        if _HTTP_CLIENT is not None:
            await _HTTP_CLIENT.aclose()
            _HTTP_CLIENT = None
        await close_osu_http_client()


app = FastAPI(title="bobert-api", lifespan=lifespan)

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Turnstile-Token", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
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


@app.get("/recommend")
def default_recommend() -> dict[str, Any]:
    rt = get_runtime()
    results = [
        public_metadata(beatmap_id, rt.metadata_by_id.get(beatmap_id, {}))
        for beatmap_id in DEFAULT_RECOMMEND_IDS
    ]
    return {"count": len(results), "results": results}


@app.post("/recommend")
async def recommend(
    payload: RecommendRequest,
    request: Request,
    x_turnstile_token: str | None = Header(default=None, alias="X-Turnstile-Token"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    ip = request_client_ip(request)
    trusted_server = bool(API_SHARED_SECRET) and x_api_key == API_SHARED_SECRET

    with timed(
        "recommend.rate_limit",
        beatmap_id=payload.beatmap_id,
        trusted_server=trusted_server,
    ):
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
    if not trusted_server:
        async with async_timed("recommend.turnstile", beatmap_id=payload.beatmap_id):
            await verify_turnstile(x_turnstile_token, ip)

    with timed("recommend.runtime", beatmap_id=payload.beatmap_id):
        rt = get_runtime()
    try:
        async with async_timed("recommend.query_embedding", beatmap_id=payload.beatmap_id):
            query_embedding, cache_status, query_metadata = await get_query_embedding(
                rt, payload.beatmap_id
            )
    except BeatmapUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    async with async_timed(
        "recommend.search",
        beatmap_id=payload.beatmap_id,
        top_k=payload.top_k,
        cache=cache_status,
    ):
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

        with timed("startup.read_beatmaps"):
            beatmaps = pl.read_parquet(beatmaps_path, columns=BEATMAP_METADATA_COLUMNS)
        with timed("startup.read_embeddings"):
            embeddings_df = pl.read_parquet(
                embeddings_path, columns=["beatmap_id", "embedding"]
            )
        if "beatmap_id" not in embeddings_df.columns or "embedding" not in embeddings_df.columns:
            raise RuntimeError("embeddings.parquet must contain beatmap_id and embedding")

        with timed("startup.prepare_embeddings", rows=embeddings_df.height):
            embedding_ids = [int(x) for x in embeddings_df["beatmap_id"].to_list()]
            embeddings = np.array(
                embeddings_df["embedding"].to_numpy(), dtype=np.float32, copy=True
            )
            normalize_rows_inplace(embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(embedding_ids):
            raise RuntimeError("invalid embeddings.parquet shape")

        with timed("startup.build_metadata", rows=beatmaps.height):
            id_to_index = {beatmap_id: idx for idx, beatmap_id in enumerate(embedding_ids)}
            metadata_by_id = load_metadata_lookup(beatmaps)
            cache = RuntimeCache(CACHE_DB, embedding_dim=embeddings.shape[1])

        with timed("startup.load_cache"):
            cached = []
            for item in cache.load_all():
                if item.beatmap_id in id_to_index:
                    continue
                if not metadata_complete(item.metadata):
                    cache.delete(item.beatmap_id)
                    continue
                cached.append(item)
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
    memory = lookup_memory_embedding(rt, beatmap_id)
    if memory is not None:
        embedding, metadata = memory
        log.info("embedding.cache_hit", beatmap_id=beatmap_id, cache="memory")
        return embedding, "hit", metadata

    cached = lookup_cached_embedding(rt, beatmap_id)
    if cached is not None:
        if not metadata_complete(cached.metadata):
            delete_stale_cached_embedding(rt, beatmap_id)
        else:
            append_query_embedding(
                rt, cached.beatmap_id, cached.embedding, cached.metadata, "sqlite"
            )
            log.info("embedding.cache_hit", beatmap_id=beatmap_id, cache="sqlite")
            return cached.embedding, "hit", cached.metadata

    if is_unavailable_embedding(rt, beatmap_id):
        log.info("embedding.unavailable", beatmap_id=beatmap_id)
        raise BeatmapUnavailableError(f"beatmap {beatmap_id} is unavailable")

    try:
        metadata = await fetch_query_metadata(beatmap_id)
    except BeatmapUnavailableError:
        mark_unavailable_embedding(rt, beatmap_id, "osu api metadata unavailable")
        raise
    if not metadata_complete(metadata):
        mark_unavailable_embedding(rt, beatmap_id, "incomplete osu api metadata")
        raise BeatmapUnavailableError(f"beatmap {beatmap_id} is unavailable")

    try:
        osu_bytes = await download_query_osu(beatmap_id)
    except ValueError as exc:
        mark_unavailable_embedding(rt, beatmap_id, str(exc))
        raise BeatmapUnavailableError(f"beatmap {beatmap_id} is unavailable") from exc
    embedding = await infer_query_embedding(rt, beatmap_id, osu_bytes)

    upsert_query_embedding(rt, beatmap_id, embedding, metadata)
    append_query_embedding(rt, beatmap_id, embedding, metadata, "miss")
    return embedding, "miss", metadata


@timed_call("embedding.memory_lookup", fields=("beatmap_id",))
def lookup_memory_embedding(
    rt: Runtime, beatmap_id: int
) -> tuple[np.ndarray, dict[str, Any]] | None:
    with rt.lock:
        idx = rt.id_to_index.get(int(beatmap_id))
        if idx is None:
            return None
        if idx < len(rt.embedding_ids):
            embedding = rt.embeddings[idx]
        else:
            embedding = rt.dynamic_embeddings[idx - len(rt.embedding_ids)]
        return embedding, rt.metadata_by_id.get(int(beatmap_id), {})


@timed_call("embedding.sqlite_lookup", fields=("beatmap_id",))
def lookup_cached_embedding(rt: Runtime, beatmap_id: int) -> CachedEmbedding | None:
    return rt.cache.get(beatmap_id)


@timed_call("embedding.sqlite_delete_stale", fields=("beatmap_id",))
def delete_stale_cached_embedding(rt: Runtime, beatmap_id: int) -> None:
    rt.cache.delete(beatmap_id)


@timed_call("embedding.unavailable_lookup", fields=("beatmap_id",))
def is_unavailable_embedding(rt: Runtime, beatmap_id: int) -> bool:
    return rt.cache.is_unavailable(beatmap_id)


@timed_call("embedding.mark_unavailable", fields=("beatmap_id",))
def mark_unavailable_embedding(rt: Runtime, beatmap_id: int, reason: str) -> None:
    rt.cache.mark_unavailable(beatmap_id, reason)


@timed_call("embedding.osu_metadata", fields=("beatmap_id",))
async def fetch_query_metadata(beatmap_id: int) -> dict[str, Any]:
    return await fetch_full_beatmap_metadata(beatmap_id)


@timed_call("embedding.osu_download", fields=("beatmap_id",))
async def download_query_osu(beatmap_id: int) -> bytes:
    return await fetch_osu_file(beatmap_id)


@timed_call("embedding.cpu_inference", fields=("beatmap_id",))
async def infer_query_embedding(
    rt: Runtime, beatmap_id: int, osu_bytes: bytes
) -> np.ndarray:
    return await run_in_threadpool(embed_osu_bytes_serialized, rt, osu_bytes)


@timed_call("embedding.cache_upsert", fields=("beatmap_id",))
def upsert_query_embedding(
    rt: Runtime, beatmap_id: int, embedding: np.ndarray, metadata: dict[str, Any]
) -> None:
    rt.cache.upsert(beatmap_id, embedding, metadata)


@timed_call("embedding.runtime_append", fields=("beatmap_id", "cache"))
def append_query_embedding(
    rt: Runtime,
    beatmap_id: int,
    embedding: np.ndarray,
    metadata: dict[str, Any],
    cache: str,
) -> None:
    with rt.lock:
        append_embedding(rt, beatmap_id, embedding, metadata)


def append_embedding(
    rt: Runtime, beatmap_id: int, embedding: np.ndarray, metadata: dict[str, Any]
) -> None:
    if beatmap_id in rt.id_to_index:
        rt.metadata_by_id.setdefault(beatmap_id, metadata)
        return
    rt.id_to_index[beatmap_id] = len(rt.embedding_ids) + len(rt.dynamic_embedding_ids)
    rt.dynamic_embedding_ids.append(beatmap_id)
    rt.dynamic_embeddings.append(normalize_rows(embedding[None, :]).astype(np.float32)[0])
    rt.metadata_by_id[beatmap_id] = metadata


def embed_osu_bytes_serialized(rt: Runtime, osu_bytes: bytes) -> np.ndarray:
    with _INFERENCE_SEMAPHORE:
        return rt.inferencer.embed_osu_bytes(osu_bytes)


def get_osu_api() -> Ossapi:
    global _OSU_API
    if _OSU_API is not None:
        return _OSU_API

    client_id = os.getenv("OSU_CLIENT_ID") or os.getenv("client_id")
    client_secret = os.getenv("OSU_CLIENT_SECRET") or os.getenv("client_secret")
    if not client_id or not client_secret:
        raise RuntimeError("osu API credentials are required for cache-miss metadata")

    with _OSU_API_LOCK:
        if _OSU_API is None:
            _OSU_API = Ossapi(int(client_id), client_secret)
        return _OSU_API


def get_osu_api_token(force_refresh: bool = False) -> str:
    api = get_osu_api()

    with _OSU_API_LOCK:
        token = api.session.token
        expires_at = float(token.get("expires_at") or 0)
        if force_refresh or expires_at <= time.time() + OSU_TOKEN_REFRESH_SKEW_SECONDS:
            api.session = api._new_client_grant(api.client_id, api.client_secret)
            token = api.session.token
            log.info(
                "osu.token_refreshed",
                expires_at=token.get("expires_at"),
                expires_in=token.get("expires_in"),
            )

        access_token = token.get("access_token")

    if not access_token:
        raise RuntimeError("osu API token is missing an access token")
    return str(access_token)


async def fetch_full_beatmap_metadata(beatmap_id: int) -> dict[str, Any]:
    response = await request_full_beatmap_metadata(beatmap_id)

    if response.status_code == 401:
        response = await request_full_beatmap_metadata(beatmap_id, force_refresh=True)

    response.raise_for_status()

    data = response.json()
    if data == {"authentication": "basic"}:
        response = await request_full_beatmap_metadata(beatmap_id, force_refresh=True)
        response.raise_for_status()
        data = response.json()

    beatmaps = data.get("beatmaps", []) if isinstance(data, dict) else []
    if not beatmaps:
        raise BeatmapUnavailableError(f"beatmap {beatmap_id} is unavailable")

    metadata = beatmap_metadata(beatmaps[0])
    if isinstance(metadata.get("ranked"), int):
        metadata["status"] = metadata["ranked"]
    if metadata.get("deleted_at"):
        raise BeatmapUnavailableError(f"beatmap {beatmap_id} is unavailable")
    return metadata


async def request_full_beatmap_metadata(
    beatmap_id: int, force_refresh: bool = False
) -> httpx.Response:
    response = await get_http_client().get(
        "https://osu.ppy.sh/api/v2/beatmaps",
        params=[("ids[]", int(beatmap_id))],
        headers={
            "Authorization": f"Bearer {get_osu_api_token(force_refresh)}",
            "x-api-version": OSU_API_VERSION,
        },
    )
    return response


def beatmap_metadata(bm: Any) -> dict[str, Any]:
    metadata = ossapi_model_data(bm)
    bs = metadata.pop("beatmapset", None) or ossapi_model_data(
        getattr(bm, "_beatmapset", None) or getattr(bm, "beatmapset", None)
    )
    if bs is not None:
        metadata.update(
            {
                key: bs.get(key)
                for key in (
                    "artist",
                    "artist_unicode",
                    "title",
                    "title_unicode",
                    "creator",
                    "source",
                    "tags",
                    "nsfw",
                    "video",
                    "storyboard",
                    "favourite_count",
                    "play_count",
                    "ranked_date",
                    "submitted_date",
                )
            }
        )

    ranked = getattr(bm, "ranked", None)
    owners = getattr(bm, "owners", None)
    if ranked is not None:
        metadata["ranked"] = getattr(ranked, "value", ranked)
    if owners:
        metadata["owners"] = " ".join(str(owner.id) for owner in owners)
    return metadata


def ossapi_model_data(value: Any) -> Any:
    if value is None:
        return None
    data = getattr(value, "_ossapi_data", None)
    if isinstance(data, dict):
        return {
            key: ossapi_model_data(item)
            for key, item in data.items()
            if not key.startswith("_")
        }
    if isinstance(value, dict):
        return {
            key: ossapi_model_data(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [ossapi_model_data(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return str(value)
    return value


def metadata_complete(metadata: dict[str, Any] | None) -> bool:
    if not metadata:
        return False
    required = [
        metadata.get("title"),
        metadata.get("creator"),
        metadata.get("version"),
        metadata.get("difficulty_rating", metadata.get("stars")),
        metadata.get("bpm"),
        metadata.get("total_length", metadata.get("hit_length")),
    ]
    if any(json_value(value) is None for value in required):
        return False
    return json_value(metadata.get("status")) is not None or json_value(
        metadata.get("ranked")
    ) is not None


def search(
    rt: Runtime,
    query_beatmap_id: int,
    query_embedding: np.ndarray,
    query_metadata: dict[str, Any],
    top_k: int,
    filters: RecommendFilters,
) -> list[dict[str, Any]]:
    with rt.lock:
        embedding_ids = list(rt.embedding_ids) + list(rt.dynamic_embedding_ids)
        embeddings = rt.embeddings
        dynamic_embeddings = list(rt.dynamic_embeddings)
        metadata_by_id = rt.metadata_by_id

    scores = embeddings @ query_embedding
    if dynamic_embeddings:
        dynamic_scores = np.asarray(dynamic_embeddings, dtype=np.float32) @ query_embedding
        scores = np.concatenate([scores, dynamic_scores])
    query_set_id = metadata_set_id(query_metadata)
    date_cutoff = date_window_cutoff(filters.date_window)
    seen_set_ids: set[int] = set()
    results = []

    evaluated: set[int] = set()
    for candidate_indices in candidate_index_batches(scores, top_k):
        for idx in candidate_indices:
            idx = int(idx)
            if idx in evaluated:
                continue
            evaluated.add(idx)

            beatmap_id = int(embedding_ids[idx])
            if beatmap_id == query_beatmap_id:
                continue

            metadata = metadata_by_id.get(beatmap_id, {})
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

            result = public_metadata(beatmap_id, metadata)
            result["score"] = float(scores[idx])
            results.append(result)
            if candidate_set_id is not None:
                seen_set_ids.add(candidate_set_id)
            if len(results) >= top_k:
                break
        if len(results) >= top_k:
            break

    return results


def candidate_index_batches(scores: np.ndarray, top_k: int):
    count = len(scores)
    last_candidate_count = 0
    for factor in SEARCH_CANDIDATE_FACTORS:
        candidate_count = min(count, max(top_k * factor, top_k))
        if candidate_count <= last_candidate_count:
            continue
        last_candidate_count = candidate_count
        if candidate_count >= count:
            yield np.argsort(-scores)
            return
        partition_indices = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
        yield partition_indices[np.argsort(-scores[partition_indices])]

    yield np.argsort(-scores)


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


def passes_filters(
    metadata: dict[str, Any], filters: RecommendFilters, date_cutoff: datetime | None
) -> bool:
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
    if not passes_range(metadata.get("bpm"), filters.min_bpm, filters.max_bpm):
        return False
    if not passes_range(
        metadata.get("total_length", metadata.get("hit_length")),
        filters.min_length,
        filters.max_length,
    ):
        return False
    if filters.status:
        status_values = {
            str(metadata.get("status", "")).lower(),
            str(metadata.get("ranked", "")).lower(),
        }
        if filters.status.lower() not in status_values:
            return False
    if date_cutoff is not None:
        release_date = parse_metadata_datetime(metadata_release_date(metadata))
        if release_date is None or release_date < date_cutoff:
            return False
    return True


def metadata_release_date(metadata: dict[str, Any]) -> Any:
    if is_ranked_status(metadata):
        ranked_date = json_value(metadata.get("ranked_date"))
        if ranked_date is not None:
            return ranked_date
    return json_value(metadata.get("submitted_date"))


def is_ranked_status(metadata: dict[str, Any]) -> bool:
    values = [metadata.get("status"), metadata.get("ranked")]
    return any(str(json_value(value)).lower() in RANKED_STATUS_VALUES for value in values)


def date_window_cutoff(window: DateWindow | None) -> datetime | None:
    if window is None:
        return None
    now = datetime.now(timezone.utc)
    if window == DateWindow.last_week:
        return now - timedelta(days=7)
    if window == DateWindow.last_month:
        return shift_months(now, -1)
    if window == DateWindow.last_3_months:
        return shift_months(now, -3)
    if window == DateWindow.last_6_months:
        return shift_months(now, -6)
    if window == DateWindow.last_year:
        return shift_months(now, -12)
    if window == DateWindow.last_2_years:
        return shift_months(now, -24)
    if window == DateWindow.last_5_years:
        return shift_months(now, -60)
    return None


def shift_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, days_in_month(year, month))
    return value.replace(year=year, month=month, day=day)


def days_in_month(year: int, month: int) -> int:
    if month == 2:
        if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
            return 29
        return 28
    return 30 if month in {4, 6, 9, 11} else 31


def parse_metadata_datetime(value: Any) -> datetime | None:
    value = json_value(value)
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


def normalize_rows_inplace(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x /= np.clip(norm, 1e-9, None)
    return x


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

    client = get_http_client()
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


def get_http_client() -> httpx.AsyncClient:
    if _HTTP_CLIENT is None:
        raise RuntimeError("HTTP client is not initialized")
    return _HTTP_CLIENT
