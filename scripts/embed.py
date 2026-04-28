from scripts.common.cli import dispatch


COMMANDS = {
    "bobert": "scripts.tasks.embed.bobert",
}


def main() -> int:
    return dispatch("scripts.embed", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
