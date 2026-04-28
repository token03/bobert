from scripts.common.cli import dispatch


COMMANDS = {
    "alignment-cache": "scripts.tasks.build_alignment_cache",
    "dataset": "scripts.tasks.create_dataset",
    "ratings": "scripts.tasks.create_ratings",
    "shard": "scripts.tasks.shard_beatmaps",
}


def main() -> int:
    return dispatch("scripts.data", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
