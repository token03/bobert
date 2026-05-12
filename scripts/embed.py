from scripts.common.cli import dispatch


COMMANDS = {
    "bobert": "scripts.tasks.embed.bobert",
    "cpu-parity": "scripts.tasks.embed.cpu_parity",
}


def main() -> int:
    return dispatch("scripts.embed", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
