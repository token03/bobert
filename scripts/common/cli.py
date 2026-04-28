from __future__ import annotations

import runpy
import sys


def dispatch(title: str, commands: dict[str, str], aliases: dict[str, str] | None = None) -> int:
    aliases = aliases or {}
    command_names = sorted(commands)

    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(f"Usage: uv run python -m {title} <command> [args]")
        print()
        print("Commands:")
        for name in command_names:
            print(f"  {name}")
        return 0

    command = aliases.get(sys.argv[1], sys.argv[1])
    module = commands.get(command)
    if module is None:
        print(f"Unknown command: {sys.argv[1]}", file=sys.stderr)
        print(f"Available commands: {', '.join(command_names)}", file=sys.stderr)
        return 2

    sys.argv = [f"{title} {command}", *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")
    return 0
