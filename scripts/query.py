from scripts.common.cli import dispatch


COMMANDS = {
    "embeddings": "scripts.tasks.query_emb",
    "recommend": "scripts.tasks.recommend_beatmap",
    "topics": "scripts.tasks.query_topics",
}

ALIASES = {
    "emb": "embeddings",
    "rec": "recommend",
}


def main() -> int:
    return dispatch("scripts.query", COMMANDS, ALIASES)


if __name__ == "__main__":
    raise SystemExit(main())
