from __future__ import annotations

import re


TOKEN_ALIASES = {
    "alternating": "alt",
    "alternate": "alt",
    "bursts": "burst",
    "doubletime": "dt",
    "easy": "ez",
    "favorite": "fav",
    "favorites": "fav",
    "favourites": "fav",
    "fingercontrol": "finger_control",
    "hardrock": "hr",
    "hidden": "hd",
    "jumps": "jump",
    "nomod": "nm",
    "prac": "practice",
    "streaming": "stream",
    "streams": "stream",
    "tourneys": "tournament",
    "tournaments": "tournament",
    "tourney": "tournament",
    "trained": "train",
    "training": "train",
}

STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "beatmap",
    "beatmaps",
    "bms",
    "by",
    "collection",
    "collections",
    "for",
    "from",
    "good",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "lov",
    "map",
    "maps",
    "new",
    "of",
    "old",
    "on",
    "or",
    "osu",
    "pack",
    "packs",
    "part",
    "play",
    "plays",
    "ranked",
    "song",
    "songs",
    "std",
    "stuff",
    "the",
    "this",
    "those",
    "to",
    "with",
    "without",
    "you",
    "your",
}


def normalize_token(token: str) -> str:
    token = token.lower().strip("'")
    token = TOKEN_ALIASES.get(token, token)
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(title: object) -> list[str]:
    text = str(title or "").lower()
    text = re.sub(r"[_/\\-]+", " ", text)
    tokens = []
    for token in re.findall(r"[a-z0-9][a-z0-9']*", text):
        token = normalize_token(token)
        if len(token) < 2 or token in STOPWORDS or token.isdigit():
            continue
        tokens.append(token)
    return tokens
