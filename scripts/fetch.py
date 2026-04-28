from scripts.common.cli import dispatch


COMMANDS = {
    "beatmaps": "scripts.tasks.fetch_beatmaps",
    "beatmapsets": "scripts.tasks.fetch_beatmapsets",
    "osu": "scripts.tasks.fetch_osu",
}


def main() -> int:
    return dispatch("scripts.fetch", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
