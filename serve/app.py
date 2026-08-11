from __future__ import annotations

import logging
import logging.handlers
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any

THREAD_COUNT = min(
    2, os.cpu_count() or 1, max(1, int(os.getenv("TORCH_NUM_THREADS", "2")))
)
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "TORCH_NUM_THREADS",
):
    os.environ[_variable] = str(THREAD_COUNT)

import httpx
import torch

torch.set_num_threads(THREAD_COUNT)

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from serve.osu import BeatmapUnavailableError, OsuClient
from serve.runtime import Runtime, metadata_complete, public_summary

MAX_RECOMMEND_TOP_K = 1000
DEFAULT_RECOMMEND_IDS = [
    1872396,
    658127,
    1351114,
    2535968,
    2809623,
    4881796,
    3592622,
    724015,
    1031991,
    2872154,
    2736518,
    3333745,
    2250670,
    1380717,
    2719326,
    555797,
    1988753,
    2096523,
    1419243,
    1787848,
]
RATE_LIMITS = {"global": 1200, "server": 600, "ip": 120}
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "")
API_SHARED_SECRET = os.getenv("API_SHARED_SECRET", "")


logging.getLogger("uvicorn.access").disabled = True
for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)
log = logging.getLogger("bobert.api")


def configure_logging() -> None:
    Path("/app/logs").mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        "/app/logs/access.log", maxBytes=2 * 1024 * 1024, backupCount=2
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


class DateWindow(str, Enum):
    last_week = "last_week"
    last_month = "last_month"
    last_3_months = "last_3_months"
    last_6_months = "last_6_months"
    last_year = "last_year"
    last_2_years = "last_2_years"
    last_5_years = "last_5_years"
    all_time = "all_time"


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


class RequestLoggingMiddleware:
    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        request_id = headers.get(b"x-request-id", str(uuid.uuid4()).encode()).decode(
            errors="replace"
        )
        client_ip = request_client_ip(scope, headers)
        start = time.perf_counter()
        status = 500
        logged = False

        async def send_logged(message: dict[str, Any]) -> None:
            nonlocal status, logged
            if message["type"] == "http.response.start":
                status = message["status"]
                elapsed = (time.perf_counter() - start) * 1000
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode()))
                response_headers.append(
                    (b"x-process-time-ms", f"{elapsed:.2f}".encode())
                )
                message["headers"] = response_headers
            await send(message)
            if (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
                and not logged
            ):
                logged = True
                elapsed = (time.perf_counter() - start) * 1000
                log.info(
                    "request_id=%s client_ip=%s method=%s path=%s status=%d elapsed_ms=%.2f",
                    request_id,
                    client_ip,
                    scope["method"],
                    scope["path"],
                    status,
                    elapsed,
                )

        try:
            await self.app(scope, receive, send_logged)
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000
            log.exception(
                "uncaught request exception request_id=%s client_ip=%s method=%s path=%s status=500 elapsed_ms=%.2f",
                request_id,
                client_ip,
                scope["method"],
                scope["path"],
                elapsed,
            )
            raise


_runtime: Runtime | None = None
_osu: OsuClient | None = None
_http_client: httpx.AsyncClient | None = None
_rate_buckets: dict[str, tuple[int, int]] = {}
_rate_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _http_client, _osu, _runtime
    configure_logging()
    _http_client = httpx.AsyncClient(
        timeout=20.0,
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
    )
    try:
        _runtime = await run_in_threadpool(Runtime)
        _osu = OsuClient(_http_client)
        await _osu.start()
        yield
    finally:
        await _http_client.aclose()
        _http_client = None
        _osu = None
        _runtime = None


app = FastAPI(title="bobert-api", lifespan=lifespan)
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
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/recommend")
def default_recommend() -> dict[str, Any]:
    runtime = get_runtime()
    results = [runtime.summary(beatmap_id) for beatmap_id in DEFAULT_RECOMMEND_IDS]
    return {"count": len(results), "results": results}


@app.post("/recommend")
async def recommend(
    payload: RecommendRequest,
    request: Request,
    x_turnstile_token: str | None = Header(default=None, alias="X-Turnstile-Token"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    ip = request_client_ip(request.scope, request.headers.raw)
    trusted_server = bool(API_SHARED_SECRET) and x_api_key == API_SHARED_SECRET
    rate_limit("global:recommend", RATE_LIMITS["global"])
    if trusted_server:
        rate_limit("server:recommend", RATE_LIMITS["server"])
    else:
        rate_limit(f"ip:{ip}:recommend", RATE_LIMITS["ip"])
        await verify_turnstile(x_turnstile_token, ip)

    runtime = get_runtime()
    memory = runtime.memory_embedding(payload.beatmap_id)
    cache_status = "hit"
    if memory is None:
        memory = await run_in_threadpool(runtime.cached_embedding, payload.beatmap_id)
    if memory is None:
        if await run_in_threadpool(runtime.cache.is_unavailable, payload.beatmap_id):
            raise HTTPException(
                status_code=404, detail=f"beatmap {payload.beatmap_id} is unavailable"
            )
        try:
            metadata = await get_osu().metadata(payload.beatmap_id)
            if not metadata_complete(metadata):
                raise BeatmapUnavailableError(
                    f"beatmap {payload.beatmap_id} is unavailable"
                )
            content = await get_osu().download(payload.beatmap_id)
            embedding = await run_in_threadpool(
                runtime.infer_and_store,
                payload.beatmap_id,
                content,
                metadata,
            )
        except BeatmapUnavailableError as exc:
            await run_in_threadpool(
                runtime.cache.mark_unavailable, payload.beatmap_id, str(exc)
            )
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            await run_in_threadpool(
                runtime.cache.mark_unavailable, payload.beatmap_id, str(exc)
            )
            raise HTTPException(
                status_code=404,
                detail=f"beatmap {payload.beatmap_id} is unavailable",
            ) from exc
        memory = embedding, metadata
        cache_status = "miss"
    embedding, metadata = memory
    results = await run_in_threadpool(
        runtime.search,
        payload.beatmap_id,
        embedding,
        metadata,
        payload.top_k,
        payload.filters,
    )
    return {
        "query": {
            "beatmap_id": payload.beatmap_id,
            "cache": cache_status,
            "metadata": public_summary(payload.beatmap_id, metadata),
        },
        "count": len(results),
        "results": results,
    }


@app.get("/beatmaps/{beatmap_id}")
async def beatmap_detail(beatmap_id: int) -> dict[str, Any]:
    detail = await run_in_threadpool(get_runtime().catalog_detail, beatmap_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"beatmap {beatmap_id} not found")
    return detail


def get_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("runtime is not initialized")
    return _runtime


def get_osu() -> OsuClient:
    if _osu is None:
        raise RuntimeError("osu client is not initialized")
    return _osu


def get_http_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError("HTTP client is not initialized")
    return _http_client


def request_client_ip(scope: dict[str, Any], raw_headers: Any) -> str:
    headers = (
        raw_headers
        if isinstance(raw_headers, dict)
        else {key.lower(): value for key, value in raw_headers}
    )
    for name in (b"cf-connecting-ip", b"x-forwarded-for"):
        value = headers.get(name)
        if value:
            return value.decode(errors="replace").split(",", 1)[0].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"


def rate_limit(key: str, limit: int) -> None:
    bucket = int(time.time()) // 3600
    with _rate_lock:
        current_bucket, count = _rate_buckets.get(key, (bucket, 0))
        count = count + 1 if current_bucket == bucket else 1
        _rate_buckets[key] = bucket, count
        if len(_rate_buckets) > 10000:
            stale = [
                name
                for name, (item_bucket, _) in _rate_buckets.items()
                if item_bucket != bucket
            ]
            for name in stale[:1000]:
                _rate_buckets.pop(name, None)
    if count > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


async def verify_turnstile(token: str | None, remote_ip: str) -> None:
    if not TURNSTILE_SECRET_KEY:
        return
    if not token:
        raise HTTPException(status_code=401, detail="Missing Turnstile token")
    response = await get_http_client().post(
        "https://challenges.cloudflare.com/turnstile/v0/siteverify",
        data={
            "secret": TURNSTILE_SECRET_KEY,
            "response": token,
            "remoteip": remote_ip,
        },
    )
    if not response.json().get("success"):
        raise HTTPException(status_code=401, detail="Turnstile verification failed")
