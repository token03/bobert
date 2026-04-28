from scripts.common.cli import dispatch


COMMANDS = {
    "collections": "scripts.tasks.eda_collections",
}


def main() -> int:
    return dispatch("scripts.eda", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
