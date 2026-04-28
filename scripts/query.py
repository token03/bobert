from scripts.common.cli import dispatch


COMMANDS = {
    "embeddings": "scripts.tasks.query.embeddings",
    "recommend": "scripts.tasks.query.recommend",
    "topics": "scripts.tasks.query.topics",
}

ALIASES = {
    "emb": "embeddings",
    "rec": "recommend",
}


def main() -> int:
    return dispatch("scripts.query", COMMANDS, ALIASES)


if __name__ == "__main__":
    raise SystemExit(main())
