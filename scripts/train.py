from scripts.common.cli import dispatch


COMMANDS = {
    "align": "scripts.tasks.train.align",
    "beatmap2vec": "scripts.tasks.train.cooccurrence",
    "lightgcn": "scripts.tasks.train.graph",
    "nmf": "scripts.tasks.train.topics",
    "pretrain": "scripts.tasks.train.pretrain",
}


def main() -> int:
    return dispatch("scripts.train", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
