from scripts.common.cli import dispatch


COMMANDS = {
    "umap": "scripts.tasks.viz.umap",
}


def main() -> int:
    return dispatch("scripts.viz", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
