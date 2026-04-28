from scripts.common.cli import dispatch


COMMANDS = {
    "bobert": "scripts.tasks.export_bobert_embeddings",
}


def main() -> int:
    return dispatch("scripts.embed", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
