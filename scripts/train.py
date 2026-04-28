from scripts.common.cli import dispatch


COMMANDS = {
    "beatmap2vec": "scripts.tasks.train.cooccurrence",
    "lightgcn": "scripts.tasks.train.graph",
    "nmf": "scripts.tasks.train.topics",
}


def main() -> int:
    return dispatch("scripts.train", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
