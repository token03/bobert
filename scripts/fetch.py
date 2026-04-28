from scripts.common.cli import dispatch


COMMANDS = {
    "beatmaps": "scripts.tasks.fetch.maps",
    "beatmapsets": "scripts.tasks.fetch.sets",
    "osu": "scripts.tasks.fetch.files",
}


def main() -> int:
    return dispatch("scripts.fetch", COMMANDS)


if __name__ == "__main__":
    raise SystemExit(main())
