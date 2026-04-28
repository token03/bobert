from scripts.common.cli import dispatch


COMMANDS = {
    "edges": "scripts.tasks.fetch_collection_edge",
    "vertices": "scripts.tasks.fetch_collection_vertex",
}

ALIASES = {
    "edge": "edges",
    "vertex": "vertices",
}


def main() -> int:
    return dispatch("scripts.collections", COMMANDS, ALIASES)


if __name__ == "__main__":
    raise SystemExit(main())
