from scripts.common.cli import dispatch


COMMANDS = {
    "beatmap2vec": "scripts.tasks.beatmap2vec",
    "lightgcn": "scripts.tasks.lightgcn",
    "nmf": "scripts.tasks.nmf",
}


def main() -> int:
    return dispatch("scripts.train", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
