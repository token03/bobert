from __future__ import annotations

import asyncio
import hmac
import logging
import os
import secrets
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

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core.modes import MODES, ModeName
from server import auth
from server.osu import BeatmapUnavailableError, OsuClient
from server.plays import group_plays
from server.runtime import (
    Runtime,
    RuntimeRegistry,
    mean_embedding,
    metadata_complete,
    metadata_mode,
)

MAX_RECOMMEND_TOP_K = 1000
MAX_RECOMMEND_SOURCES = 100
MAX_PLAYS = 100
MAX_UNINDEXED_SOURCES = 5
MODE_BY_LABEL = {mode.label: name for name, mode in MODES.items()}


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
    mode: ModeName | None = None
    beatmap_ids: list[Annotated[int, Field(gt=0)]] = Field(
        min_length=1, max_length=MAX_RECOMMEND_SOURCES
    )
    top_k: int = Field(default=20, ge=1, le=MAX_RECOMMEND_TOP_K)
    filters: RecommendFilters = Field(default_factory=RecommendFilters)
    distinct: bool = False


class BeatmapSummary(BaseModel):
    mode: ModeName
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
    play_count: int | None
    favourite_count: int | None
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


class PlayerProfile(BaseModel):
    user_id: int
    username: str
    avatar_url: str | None
    mode: ModeName


class Session(BaseModel):
    user_id: int
    username: str


class Play(BaseModel):
    beatmap_id: int
    pp: float | None
    accuracy: float
    rank: str
    mods: list[str]
    ended_at: str


class PlaysResponse(BaseModel):
    plays: list[Play]


class GroupPlay(BaseModel):
    beatmap_id: int = Field(gt=0)
    mods: list[Annotated[str, Field(max_length=4)]] = Field(max_length=16)


class GroupRequest(BaseModel):
    mode: ModeName
    plays: list[GroupPlay] = Field(min_length=1, max_length=MAX_PLAYS)


class PlayGroup(BaseModel):
    label: str
    beatmap_ids: list[int]


class GroupResponse(BaseModel):
    groups: list[PlayGroup]
    sources: list[BeatmapSummary]
    missing: list[int]


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
        cookie = headers.get(b"cookie")
        session = auth.cookie_session(
            cookie.decode(errors="replace") if cookie else None
        )
        user = f"{session['user_id']}:{session['username']}" if session else "-"
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
                if session:
                    response_headers.append((b"x-bobert-user", user.encode()))
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
                        "request_id=%s client_ip=%s user=%s method=%s path=%s status=%d elapsed_ms=%.2f",
                        request_id,
                        client_ip,
                        user,
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
                "uncaught request exception request_id=%s client_ip=%s user=%s method=%s path=%s status=500 elapsed_ms=%.2f",
                request_id,
                client_ip,
                user,
                scope["method"],
                scope["path"],
                elapsed,
            )
            raise


_runtime: RuntimeRegistry | None = None
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
        _runtime = await run_in_threadpool(RuntimeRegistry)
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


@app.get("/api/modes")
def available_modes() -> dict[str, Any]:
    return {"modes": list(_runtime.runtimes) if _runtime else []}


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/api/auth/login")
def login(
    next: Annotated[str, Query(pattern=r"^/([^/\\].*)?$")] = "/",
) -> RedirectResponse:
    state = secrets.token_urlsafe(16)
    response = RedirectResponse(get_osu().authorize_url(state), status_code=302)
    response.set_cookie(
        auth.STATE_COOKIE,
        auth.sign({"state": state, "next": next}, auth.STATE_SECONDS),
        max_age=auth.STATE_SECONDS,
        path="/api/auth",
        secure=True,
        httponly=True,
        samesite="lax",
    )
    return response


@app.get("/api/auth/callback")
async def login_callback(request: Request, code: str, state: str) -> RedirectResponse:
    pending = auth.verify(request.cookies.get(auth.STATE_COOKIE))
    if pending is None or not hmac.compare_digest(pending["state"], state):
        raise HTTPException(status_code=400, detail="login expired; try again")
    user = await get_osu().identify(code)
    session = {"user_id": user["user_id"], "username": user["username"]}
    log.info("login user=%s:%s", session["user_id"], session["username"])
    response = RedirectResponse(pending["next"], status_code=302)
    response.delete_cookie(auth.STATE_COOKIE, path="/api/auth")
    response.set_cookie(
        auth.SESSION_COOKIE,
        auth.sign(session, auth.SESSION_SECONDS),
        max_age=auth.SESSION_SECONDS,
        secure=True,
        httponly=True,
        samesite="lax",
    )
    return response


@app.post("/api/auth/logout", status_code=204)
def logout(response: Response) -> None:
    response.delete_cookie(auth.SESSION_COOKIE)


def require_session(request: Request, response: Response) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    session = auth.verify(request.cookies.get(auth.SESSION_COOKIE))
    if session is None:
        raise HTTPException(status_code=401, detail="not logged in")
    return session


@app.get("/api/me", response_model=Session)
def me(session: Annotated[dict[str, Any], Depends(require_session)]) -> dict[str, Any]:
    return session


@app.get("/api/me/groups", response_model=GroupResponse)
async def my_groups(
    session: Annotated[dict[str, Any], Depends(require_session)],
    mode: ModeName = "std",
) -> dict[str, Any]:
    plays = await get_osu().top_plays(session["user_id"], MODES[mode].label, MAX_PLAYS)
    return await run_in_threadpool(
        play_groups, mode, [(play["beatmap_id"], play["mods"]) for play in plays]
    )


@app.get("/api/search/{mode}")
def search_catalog(mode: ModeName) -> FileResponse:
    return FileResponse(
        get_runtime(mode).artifacts.search,
        media_type="application/octet-stream",
        headers={"Content-Encoding": "br", "Cache-Control": "no-cache"},
    )


@app.get("/api/recommend", response_model=DefaultRecommendResponse)
def default_recommend(
    response: Response, seed: int | None = None, mode: ModeName = "std"
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    results = get_runtime(mode).default_summaries(seed)
    return {"count": len(results), "results": results}


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest) -> dict[str, Any]:
    beatmap_ids = list(dict.fromkeys(payload.beatmap_ids))
    inferred_metadata = {}
    mode = payload.mode
    if mode is None:
        mode = await run_in_threadpool(_runtime.source_mode, beatmap_ids[0])
        if mode is None:
            try:
                metadata = await get_osu().metadata(beatmap_ids[0])
                if not metadata_complete(metadata):
                    raise BeatmapUnavailableError(
                        f"beatmap {beatmap_ids[0]} is unavailable"
                    )
                mode = metadata_mode(metadata)
                inferred_metadata[beatmap_ids[0]] = metadata
            except BeatmapUnavailableError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
    runtime = get_runtime(mode)
    try:
        _runtime.validate_sources(mode, beatmap_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    unindexed = await run_in_threadpool(runtime.unindexed, beatmap_ids)
    if len(unindexed) > MAX_UNINDEXED_SOURCES:
        raise HTTPException(
            status_code=422,
            detail=f"at most {MAX_UNINDEXED_SOURCES} unindexed beatmaps per request",
        )

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
                metadata = inferred_metadata.get(
                    beatmap_id
                ) or await get_osu().metadata(beatmap_id)
                if not metadata_complete(metadata):
                    raise BeatmapUnavailableError(
                        f"beatmap {beatmap_id} is unavailable"
                    )
                runtime.validate_source(beatmap_id, metadata)
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
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            memory = embedding, metadata
            cache_status = "miss"
        embedding, metadata = memory
        try:
            runtime.validate_source(beatmap_id, metadata)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
        payload.distinct,
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


@app.get("/api/players/{username}", response_model=PlayerProfile)
async def player_profile(
    username: Annotated[str, Path(pattern=r"^[\w \[\]-]{2,15}$")],
) -> dict[str, Any]:
    try:
        user = await get_osu().player(username)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=404, detail=f"user {username} not found"
            ) from exc
        raise
    return {**user, "mode": MODE_BY_LABEL[user["playmode"]]}


@app.get("/api/users/{user_id}/top", response_model=PlaysResponse)
async def top_plays(
    user_id: Annotated[int, Path(gt=0)],
    mode: ModeName = "std",
    limit: Annotated[int, Query(ge=1, le=MAX_PLAYS)] = MAX_PLAYS,
) -> dict[str, Any]:
    try:
        plays = await get_osu().top_plays(user_id, MODES[mode].label, limit)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise HTTPException(
                status_code=404, detail=f"user {user_id} not found"
            ) from exc
        raise
    return {"plays": plays}


@app.post("/api/beatmaps/groups", response_model=GroupResponse)
def group_beatmaps(payload: GroupRequest) -> dict[str, Any]:
    return play_groups(
        payload.mode, [(play.beatmap_id, play.mods) for play in payload.plays]
    )


def play_groups(mode: str, plays: list[tuple[int, list[str]]]) -> dict[str, Any]:
    runtime = get_runtime(mode)
    found: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    missing: list[int] = []
    for beatmap_id in dict.fromkeys(beatmap_id for beatmap_id, _ in plays):
        memory = runtime.memory_embedding(beatmap_id) or runtime.cached_embedding(
            beatmap_id
        )
        if memory is None:
            missing.append(beatmap_id)
        else:
            found[beatmap_id] = memory
    groups = group_plays(
        [(beatmap_id, mods) for beatmap_id, mods in plays if beatmap_id in found],
        {beatmap_id: embedding for beatmap_id, (embedding, _) in found.items()},
        runtime.labels,
    )
    return {
        "groups": [{"label": label, "beatmap_ids": ids} for label, ids in groups],
        "sources": [
            runtime.public_summary(beatmap_id, metadata)
            for beatmap_id, (_, metadata) in found.items()
        ],
        "missing": missing,
    }


def get_runtime(mode: str = "std") -> Runtime:
    if _runtime is None:
        raise RuntimeError("runtime is not initialized")
    try:
        return _runtime.get(mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
    uvicorn.run(app, host="0.0.0.0", port=8000, proxy_headers=True, access_log=False)
