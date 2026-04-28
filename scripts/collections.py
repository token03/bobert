from scripts.common.cli import dispatch


COMMANDS = {
    "edges": "scripts.tasks.collections.edges",
    "vertices": "scripts.tasks.collections.vertices",
}

ALIASES = {
    "edge": "edges",
    "vertex": "vertices",
}


def main() -> int:
    return dispatch("scripts.collections", COMMANDS, ALIASES)


if __name__ == "__main__":
    raise SystemExit(main())
