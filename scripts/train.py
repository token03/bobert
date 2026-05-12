from scripts.common.cli import dispatch


COMMANDS = {
    "align": "scripts.tasks.train.align",
    "graph": "scripts.tasks.train.graph",
    "pretrain": "scripts.tasks.train.pretrain",
    "strip": "scripts.tasks.train.strip",
}


def main() -> int:
    return dispatch("scripts.train", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
