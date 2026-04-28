from scripts.common.cli import dispatch


COMMANDS = {
    "alignment-cache": "scripts.tasks.data.alignment",
    "dataset": "scripts.tasks.data.dataset",
    "ratings": "scripts.tasks.data.ratings",
    "shard": "scripts.tasks.data.shard",
}


def main() -> int:
    return dispatch("scripts.data", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
