from scripts.common.cli import dispatch


COMMANDS = {
    "compare": "scripts.tasks.query.recommend",
    "recommend": "scripts.tasks.query.recommend",
}

ALIASES = {
    "cmp": "compare",
    "rec": "recommend",
}


def main() -> int:
    return dispatch("scripts.query", COMMANDS, ALIASES)


if __name__ == "__main__":
    raise SystemExit(main())
