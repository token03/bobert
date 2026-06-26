from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.common.paths import COLLECTIONS_DIR, resolve_path


VERTEX_PATH = COLLECTIONS_DIR / "vertices.parquet"
EDGE_PATH = COLLECTIONS_DIR / "edges.parquet"
OUTPUT_PATH = COLLECTIONS_DIR / "ngrams.txt"

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


def collect_ngrams(titles: list[object]) -> tuple[Counter[str], Counter[str]]:
    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    for title in titles:
        tokens = tokenize(title)
        unigrams.update(set(tokens))
        bigrams.update({" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)})
    return unigrams, bigrams


def load_titles(vertex_path: Path, edge_path: Path) -> list[object]:
    vertices = pd.read_parquet(vertex_path, columns=["collection_id", "source", "name"])
    edges = pd.read_parquet(edge_path, columns=["collection_id", "source"]).drop_duplicates()
    return vertices.merge(edges, on=["collection_id", "source"], how="inner")["name"].tolist()


def write_ngrams(
    output_path: Path,
    unigrams: Counter[str],
    bigrams: Counter[str],
    min_unigram_df: int,
    min_bigram_df: int,
) -> None:
    filtered_unigrams = sorted(
        (ngram for ngram, count in unigrams.items() if count >= min_unigram_df),
        key=lambda ngram: (-unigrams[ngram], ngram),
    )
    filtered_bigrams = sorted(
        (ngram for ngram, count in bigrams.items() if count >= min_bigram_df),
        key=lambda ngram: (-bigrams[ngram], ngram),
    )
    lines = [
        f"# unigrams min_doc_freq={min_unigram_df}",
        *filtered_unigrams,
        "",
        f"# bigrams min_doc_freq={min_bigram_df}",
        *filtered_bigrams,
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate title n-grams for collection labeling")
    parser.add_argument("--vertices", type=Path, default=VERTEX_PATH)
    parser.add_argument("--edges", type=Path, default=EDGE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--min-unigram-df", type=int, default=50)
    parser.add_argument("--min-bigram-df", type=int, default=20)
    args = parser.parse_args()

    titles = load_titles(resolve_path(args.vertices), resolve_path(args.edges))
    unigrams, bigrams = collect_ngrams(titles)
    write_ngrams(
        resolve_path(args.output),
        unigrams,
        bigrams,
        args.min_unigram_df,
        args.min_bigram_df,
    )
    print(
        f"Wrote {sum(count >= args.min_unigram_df for count in unigrams.values())} unigrams "
        f"and {sum(count >= args.min_bigram_df for count in bigrams.values())} bigrams "
        f"to {resolve_path(args.output)}"
    )


if __name__ == "__main__":
    main()
