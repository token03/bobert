from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from collections import deque
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

from server.mods import MOD_ORDER, mods_name, normalize_mods

IRC_HOST = "irc.ppy.sh"
IRC_PORT = 6667
PAGE_SIZE = 5
POOL_SIZE = 25
MESSAGE_BYTES = 450
SEND_LIMIT = 8
SEND_PERIOD = 6.0
QUEUE_SIZE = 32
READ_TIMEOUT = 300.0
MIN_BACKOFF = 5.0
MAX_BACKOFF = 300.0
HELP = (
    "/np: maps like this one | !r: ranked maps like your top plays, "
    "add mods for modded plays (!r dt, !r hdhr). [https://bobert-web.pages.dev/ bobert]"
)
DIFFICULTY_MODS = frozenset(MOD_ORDER) - {"HD"}
BEATMAP_LINK = re.compile(
    r"osu\.ppy\.sh/(?:b|beatmaps)/(\d+)|osu\.ppy\.sh/beatmapsets/\d+#\w+/(\d+)"
)

log = logging.getLogger("bobert.bot")


class AuthenticationError(RuntimeError):
    pass


class SendWindow:
    def __init__(self, limit: int, period: float):
        self.limit = limit
        self.period = period
        self.sent: deque[float] = deque()

    async def acquire(self) -> None:
        while len(self.sent) >= self.limit:
            wait = self.sent[0] + self.period - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self.sent.popleft()
        self.sent.append(time.monotonic())


def beatmap_id(text: str) -> int | None:
    match = BEATMAP_LINK.search(text)
    if match is None:
        return None
    return int(match.group(1) or match.group(2))


def parse_mods(words: list[str]) -> frozenset[str] | None:
    text = "".join(words).upper().replace("+", "")
    if text in ("", "NM", "NOMOD"):
        return frozenset()
    pairs = [text[start : start + 2] for start in range(0, len(text), 2)]
    mods = normalize_mods(pairs)
    if len(text) % 2 or not mods <= set(MOD_ORDER):
        return None
    return mods


def plays_with(mods: frozenset[str], play_mods: list[str]) -> bool:
    played = normalize_mods(play_mods)
    if mods:
        return mods <= played
    return not played & DIFFICULTY_MODS


def fit_line(results: list[dict[str, Any]], prefix: str = "") -> str:
    for width in range(80, 8, -4):
        line = prefix + " | ".join(link(result, width) for result in results)
        if len(line.encode()) <= MESSAGE_BYTES:
            return line
    raise ValueError("recommendations do not fit in one message")


def link(result: dict[str, Any], width: int) -> str:
    name = f"{result['title']} [{result['version']}]"
    if len(name) > width:
        name = name[: width - 1] + "…"
    return (
        f"[https://osu.ppy.sh/b/{result['beatmap_id']} {name}] {result['stars']:.2f}*"
    )


class Bot:
    def __init__(self, username: str, password: str, api: httpx.AsyncClient):
        self.username = username
        self.nick = username.replace(" ", "_")
        self.password = password
        self.api = api
        self.outbox: asyncio.Queue[tuple[str, str]] = asyncio.Queue(QUEUE_SIZE)
        self.window = SendWindow(SEND_LIMIT, SEND_PERIOD)
        self.busy: set[str] = set()
        self.tasks: set[asyncio.Task[None]] = set()
        self.backoff = MIN_BACKOFF

    async def run(self) -> None:
        while True:
            try:
                await self.connect()
            except (OSError, TimeoutError) as exc:
                log.warning("disconnected: %r", exc)
            log.info("reconnecting in %.0fs", self.backoff)
            await asyncio.sleep(self.backoff)
            self.backoff = min(self.backoff * 2, MAX_BACKOFF)

    async def connect(self) -> None:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(IRC_HOST, IRC_PORT), 30
        )
        sender = asyncio.create_task(self.send_loop(writer))
        try:
            self.write(writer, f"PASS {self.password}")
            self.write(writer, f"NICK {self.nick}")
            self.write(writer, f"USER {self.nick} 0 * :{self.nick}")
            await writer.drain()
            while line := await asyncio.wait_for(reader.readline(), READ_TIMEOUT):
                await self.handle(writer, line.decode(errors="replace").rstrip("\r\n"))
        finally:
            sender.cancel()
            writer.close()

    def write(self, writer: asyncio.StreamWriter, line: str) -> None:
        writer.write(line.replace("\r", " ").replace("\n", " ").encode() + b"\r\n")

    async def send_loop(self, writer: asyncio.StreamWriter) -> None:
        while True:
            target, text = await self.outbox.get()
            await self.window.acquire()
            self.write(writer, f"PRIVMSG {target} :{text}")
            await writer.drain()
            log.info("-> %s: %s", target, text)

    async def handle(self, writer: asyncio.StreamWriter, line: str) -> None:
        prefix, _, rest = (
            line.partition(" ") if line.startswith(":") else ("", "", line)
        )
        command, _, params = rest.partition(" ")
        if command == "PING":
            self.write(writer, f"PONG {params}")
            await writer.drain()
        elif command == "001":
            log.info("logged in as %s", self.nick)
            self.backoff = MIN_BACKOFF
        elif command == "464" or "required to authenticate" in params:
            raise AuthenticationError(
                "Bancho rejected the IRC password, get it from "
                "https://osu.ppy.sh/home/account/edit#legacy-api"
            )
        elif command == "PRIVMSG":
            target, _, text = params.partition(" :")
            sender = prefix[1:].partition("!")[0]
            if target.lower() == self.nick.lower() and sender.lower() not in (
                self.nick.lower(),
                "banchobot",
            ):
                self.dispatch(sender, text)

    def dispatch(self, sender: str, text: str) -> None:
        if sender in self.busy:
            return
        self.busy.add(sender)
        task = asyncio.create_task(self.respond(sender, text))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def respond(self, sender: str, text: str) -> None:
        try:
            reply = await self.reply(sender, text)
            if reply is not None:
                self.queue(sender, reply)
        except Exception:
            log.exception("failed to answer %s: %s", sender, text)
            self.queue(sender, "Something went wrong, try again later.")
        finally:
            self.busy.discard(sender)

    def queue(self, sender: str, text: str) -> None:
        try:
            self.outbox.put_nowait((sender, text))
        except asyncio.QueueFull:
            log.warning("outbox full, dropped reply to %s", sender)

    async def reply(self, sender: str, text: str) -> str | None:
        words = text.split()
        command = words[0].lower() if words else ""
        found = beatmap_id(text) if command == "\x01action" else None
        if found is None and not command.startswith("!"):
            return None
        log.info("<- %s: %s", sender, text)
        if found is not None:
            return await self.similar(found)
        if command != "!r":
            return HELP
        mods = parse_mods(words[1:])
        if mods is None:
            return "Unknown mods. Try !r, !r dt or !r hdhr."
        return await self.recommend(sender, mods)

    async def similar(self, beatmap_id: int) -> str:
        response = await self.api.post(
            "/api/recommend",
            json={"beatmap_ids": [beatmap_id], "top_k": PAGE_SIZE, "distinct": True},
        )
        if response.status_code in (404, 422):
            return f"Can't recommend from that map: {response.json()['detail']}"
        response.raise_for_status()
        return fit_line(response.json()["results"])

    async def recommend(self, sender: str, mods: frozenset[str]) -> str:
        player = await self.player(sender)
        if player is None:
            return "Couldn't find your osu! profile."
        top = await self.top_plays(player)
        plays = [play for play in top if plays_with(mods, play["mods"])]
        if not plays:
            return f"No top plays with {mods_name(mods) or 'no mods'}."
        response = await self.api.post(
            "/api/beatmaps/groups",
            json={
                "mode": player["mode"],
                "plays": [
                    {"beatmap_id": play["beatmap_id"], "mods": play["mods"]}
                    for play in plays
                ],
            },
        )
        response.raise_for_status()
        grouping = response.json()
        groups = grouping["groups"]
        if not groups:
            return "Those top plays aren't in the recommendation index yet."
        group = random.choices(
            groups, weights=[len(group["beatmap_ids"]) for group in groups]
        )[0]
        played_sets = {source["beatmapset_id"] for source in grouping["sources"]}
        played_maps = {play["beatmap_id"] for play in top}
        response = await self.api.post(
            "/api/recommend",
            json={
                "mode": player["mode"],
                "beatmap_ids": group["beatmap_ids"],
                "top_k": POOL_SIZE + len(played_sets),
                "filters": {"status": "ranked"},
                "distinct": True,
            },
        )
        response.raise_for_status()
        pool = [
            result
            for result in response.json()["results"]
            if result["beatmapset_id"] not in played_sets
            and result["beatmap_id"] not in played_maps
        ][:POOL_SIZE]
        if not pool:
            return "No new maps found for your top plays."
        picks = random.sample(pool, min(PAGE_SIZE, len(pool)))
        prefix = f"+{mods_name(mods)}: " if mods else ""
        return fit_line(sorted(picks, key=lambda result: -result["score"]), prefix)

    async def player(self, sender: str) -> dict[str, Any] | None:
        response = await self.api.get(f"/api/players/{quote(sender)}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def top_plays(self, player: dict[str, Any]) -> list[dict[str, Any]]:
        response = await self.api.get(
            f"/api/users/{player['user_id']}/top", params={"mode": player["mode"]}
        )
        response.raise_for_status()
        return response.json()["plays"]


async def main_async() -> None:
    load_dotenv()
    username = os.environ["OSU_IRC_USERNAME"]
    password = os.environ["OSU_IRC_PASSWORD"]
    base_url = os.getenv("BOBERT_API_URL", "http://127.0.0.1:8008")
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as api:
        await Bot(username, password, api).run()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
