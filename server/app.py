from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

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
import numpy as np
import torch
import uvicorn

torch.set_num_threads(THREAD_COUNT)

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from server.osu import BeatmapUnavailableError, OsuClient
from server.runtime import Runtime, mean_embedding, metadata_complete

MAX_RECOMMEND_TOP_K = 1000
MAX_RECOMMEND_SOURCES = 10


log = logging.getLogger("uvicorn.error")


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
    min_date: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    max_date: str | None = Field(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    exclude_same_set: bool = True


class RecommendRequest(BaseModel):
    beatmap_ids: list[Annotated[int, Field(gt=0)]] = Field(
        min_length=1, max_length=MAX_RECOMMEND_SOURCES
    )
    top_k: int = Field(default=20, ge=1, le=MAX_RECOMMEND_TOP_K)
    filters: RecommendFilters = Field(default_factory=RecommendFilters)


class BeatmapSummary(BaseModel):
    beatmap_id: int
    beatmapset_id: int | None
    artist: str | None
    title: str | None
    creator: str | None
    user_id: int | None
    version: str | None
    status: str | int | None
    stars: float | None
    ar: float | None
    cs: float | None
    accuracy: float | None
    drain: float | None
    bpm: float | None
    total_length: float | None
    last_updated: str | None
    ranked_date: str | None
    submitted_date: str | None
    release_date: str | None
    url: str


class ScoredBeatmapSummary(BeatmapSummary):
    score: float


class DefaultRecommendResponse(BaseModel):
    count: int
    results: list[BeatmapSummary]


class RecommendSource(BaseModel):
    beatmap_id: int
    cache: Literal["hit", "miss"]
    metadata: BeatmapSummary


class RecommendResponse(BaseModel):
    sources: list[RecommendSource]
    count: int
    results: list[ScoredBeatmapSummary]


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
                if not (scope["path"] == "/health" and status < 400):
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _http_client, _osu, _runtime
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


app = FastAPI(
    title="bobert-api",
    version="0.1.0",
    lifespan=lifespan,
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    swagger_ui_oauth2_redirect_url="/api/docs/oauth2-redirect",
)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/recommend", response_model=DefaultRecommendResponse)
def default_recommend(response: Response, seed: int | None = None) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    results = get_runtime().default_summaries(seed)
    return {"count": len(results), "results": results}


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest) -> dict[str, Any]:
    runtime = get_runtime()
    beatmap_ids = list(dict.fromkeys(payload.beatmap_ids))

    async def load_source(
        beatmap_id: int,
    ) -> tuple[int, np.ndarray, dict[str, Any], Literal["hit", "miss"]]:
        memory = runtime.memory_embedding(beatmap_id)
        cache_status: Literal["hit", "miss"] = "hit"
        if memory is None:
            memory = await run_in_threadpool(runtime.cached_embedding, beatmap_id)
        if memory is None:
            if await run_in_threadpool(runtime.cache.is_unavailable, beatmap_id):
                raise HTTPException(
                    status_code=404, detail=f"beatmap {beatmap_id} is unavailable"
                )
            try:
                metadata = await get_osu().metadata(beatmap_id)
                if not metadata_complete(metadata):
                    raise BeatmapUnavailableError(
                        f"beatmap {beatmap_id} is unavailable"
                    )
                content = await get_osu().download(beatmap_id)
                embedding = await run_in_threadpool(
                    runtime.infer_and_store,
                    beatmap_id,
                    content,
                    metadata,
                )
            except BeatmapUnavailableError as exc:
                await run_in_threadpool(
                    runtime.cache.mark_unavailable, beatmap_id, str(exc)
                )
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                await run_in_threadpool(
                    runtime.cache.mark_unavailable, beatmap_id, str(exc)
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"beatmap {beatmap_id} is unavailable",
                ) from exc
            memory = embedding, metadata
            cache_status = "miss"
        embedding, metadata = memory
        return beatmap_id, embedding, metadata, cache_status

    sources = await asyncio.gather(
        *(load_source(beatmap_id) for beatmap_id in beatmap_ids)
    )
    embedding = mean_embedding([source[1] for source in sources])
    results = await run_in_threadpool(
        runtime.search,
        beatmap_ids,
        embedding,
        [source[2] for source in sources],
        payload.top_k,
        payload.filters,
    )
    return {
        "sources": [
            {
                "beatmap_id": beatmap_id,
                "cache": cache_status,
                "metadata": runtime.public_summary(beatmap_id, metadata),
            }
            for beatmap_id, _, metadata, cache_status in sources
        ],
        "count": len(results),
        "results": results,
    }


@app.get("/api/beatmaps/{beatmap_id}")
async def beatmap_detail(beatmap_id: int) -> dict[str, Any]:
    detail = await run_in_threadpool(get_runtime().catalog_detail, beatmap_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"beatmap {beatmap_id} not found")
    return detail


@app.get("/api/beatmaps/{beatmap_id}/summary", response_model=BeatmapSummary)
async def beatmap_summary(
    beatmap_id: int, response: Response
) -> dict[str, Any]:
    runtime = get_runtime()
    memory = runtime.memory_embedding(beatmap_id)
    if memory is None:
        memory = await run_in_threadpool(runtime.cached_embedding, beatmap_id)
    if memory is not None:
        metadata = memory[1]
    else:
        if await run_in_threadpool(runtime.cache.is_unavailable, beatmap_id):
            raise HTTPException(
                status_code=404, detail=f"beatmap {beatmap_id} is unavailable"
            )
        try:
            metadata = await get_osu().metadata(beatmap_id)
            if not metadata_complete(metadata):
                raise BeatmapUnavailableError(
                    f"beatmap {beatmap_id} is unavailable"
                )
        except (BeatmapUnavailableError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    response.headers["Cache-Control"] = "public, max-age=300"
    return runtime.public_summary(beatmap_id, metadata)


def get_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("runtime is not initialized")
    return _runtime


def get_osu() -> OsuClient:
    if _osu is None:
        raise RuntimeError("osu client is not initialized")
    return _osu


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
if __name__ == "__main__":
    sockets = []
    for family, host in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        sock = socket.socket(family)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind((host, 8000))
        sock.setblocking(False)
        sockets.append(sock)
    config = uvicorn.Config(app, workers=1, proxy_headers=True, access_log=False)
    asyncio.run(uvicorn.Server(config).serve(sockets=sockets))
