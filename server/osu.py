from __future__ import annotations

import asyncio
import os
import time
from enum import Enum
from typing import Any

import httpx

OSU_API_VERSION = "20241024"
TOKEN_REFRESH_SKEW_SECONDS = 300


class BeatmapUnavailableError(ValueError):
    pass


class OsuClient:
    def __init__(self, client: httpx.AsyncClient):
        client_id = os.getenv("OSU_CLIENT_ID") or os.getenv("client_id")
        client_secret = os.getenv("OSU_CLIENT_SECRET") or os.getenv("client_secret")
        if not client_id or not client_secret:
            raise RuntimeError("osu API credentials are required")
        self.client = client
        self.client_id = int(client_id)
        self.client_secret = client_secret
        self._token: str | None = None
        self._expires_at = 0.0
        self._token_lock = asyncio.Lock()
        self._metadata: dict[int, dict[str, Any]] = {}
        self._metadata_tasks: dict[int, asyncio.Task[dict[str, Any]]] = {}

    async def start(self) -> None:
        await self._access_token()

    async def _access_token(self, force: bool = False) -> str:
        if (
            not force
            and self._token
            and self._expires_at > time.time() + TOKEN_REFRESH_SKEW_SECONDS
        ):
            return self._token
        async with self._token_lock:
            if (
                not force
                and self._token
                and self._expires_at > time.time() + TOKEN_REFRESH_SKEW_SECONDS
            ):
                return self._token
            response = await self.client.post(
                "https://osu.ppy.sh/oauth/token",
                json={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials",
                    "scope": "public",
                },
            )
            response.raise_for_status()
            data = response.json()
            self._token = str(data["access_token"])
            self._expires_at = time.time() + int(data.get("expires_in", 0))
            return self._token

    async def metadata(self, beatmap_id: int) -> dict[str, Any]:
        cached = self._metadata.get(beatmap_id)
        if cached is not None:
            return cached
        task = self._metadata_tasks.get(beatmap_id)
        if task is None:
            task = asyncio.create_task(self._fetch_metadata(beatmap_id))
            self._metadata_tasks[beatmap_id] = task
        try:
            metadata = await asyncio.shield(task)
            self._metadata[beatmap_id] = metadata
            if len(self._metadata) > 256:
                self._metadata.pop(next(iter(self._metadata)))
            return metadata
        finally:
            if task.done() and self._metadata_tasks.get(beatmap_id) is task:
                self._metadata_tasks.pop(beatmap_id, None)

    async def _fetch_metadata(self, beatmap_id: int) -> dict[str, Any]:
        data = await self._get("beatmaps", [("ids[]", int(beatmap_id))])
        beatmaps = data.get("beatmaps", []) if isinstance(data, dict) else []
        if not beatmaps:
            raise BeatmapUnavailableError(f"beatmap {beatmap_id} is unavailable")
        metadata = parse_beatmap_metadata(beatmaps[0])
        if metadata.get("deleted_at"):
            raise BeatmapUnavailableError(f"beatmap {beatmap_id} is unavailable")
        return metadata

    async def user_plays(
        self, user_id: int, kind: str, ruleset: str, limit: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        user, scores = await asyncio.gather(
            self._get(f"users/{int(user_id)}/{ruleset}", {"key": "id"}),
            self._get(
                f"users/{int(user_id)}/scores/{kind}",
                {"mode": ruleset, "limit": limit},
            ),
        )
        return parse_player(user), [parse_play(score) for score in scores]

    async def _get(self, path: str, params: Any) -> Any:
        response = await self._request(path, params)
        if response.status_code == 401:
            response = await self._request(path, params, force=True)
        response.raise_for_status()
        return response.json()

    async def _request(
        self, path: str, params: Any, force: bool = False
    ) -> httpx.Response:
        return await self.client.get(
            f"https://osu.ppy.sh/api/v2/{path}",
            params=params,
            headers={
                "Authorization": f"Bearer {await self._access_token(force)}",
                "x-api-version": OSU_API_VERSION,
            },
        )

    async def download(self, beatmap_id: int) -> bytes:
        response = await self.client.get(
            f"https://osu.ppy.sh/osu/{int(beatmap_id)}",
            headers={
                "User-Agent": "bobert-api/0.1 (+https://osu.ppy.sh)",
                "Accept": "text/plain,*/*;q=0.8",
            },
        )
        if response.status_code != 200 or not is_valid_osu_file(response.content):
            raise BeatmapUnavailableError(f"beatmap {beatmap_id} is unavailable")
        return response.content


def parse_beatmap_metadata(beatmap: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: json_data(value) for key, value in beatmap.items() if key != "beatmapset"
    }
    beatmapset = beatmap.get("beatmapset") or {}
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
    ):
        metadata[key] = json_data(beatmapset.get(key))
    if isinstance(metadata.get("ranked"), int):
        metadata["status"] = metadata["ranked"]
    owners = metadata.get("owners")
    if owners:
        metadata["owners"] = " ".join(str(owner["id"]) for owner in owners)
    return metadata


def parse_player(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": user["id"],
        "username": user["username"],
        "avatar_url": user.get("avatar_url"),
    }


def parse_play(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "beatmap_id": score["beatmap_id"],
        "pp": score.get("pp"),
        "accuracy": score["accuracy"],
        "rank": score["rank"],
        "mods": [
            mod["acronym"] for mod in score.get("mods", []) if mod["acronym"] != "CL"
        ],
        "ended_at": score["ended_at"],
    }


def json_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_data(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def is_valid_osu_file(content: bytes) -> bool:
    if len(content) < 100:
        return False
    first_line = content.decode("utf-8", errors="ignore").split("\n", 1)[0]
    return first_line.lstrip("\ufeff").strip().startswith("osu file format v")
