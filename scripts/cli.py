from __future__ import annotations

import runpy
import sys


MODEL_COMMANDS = {
    "adapter": "scripts.bobert.adapter",
    "align": "scripts.bobert.align",
    "embed": "scripts.bobert.embed",
    "pretrain": "scripts.bobert.pretrain",
    "strip": "scripts.bobert.strip",
    "recommend": "scripts.query.recommend",
    "graph": "scripts.data.graph",
    "mining": "scripts.data.mining",
}

GROUP_COMMANDS = {
    ("collections", "edges"): "scripts.collections.edges",
    ("collections", "ngram"): "scripts.collections.ngram",
    ("collections", "tournaments"): "scripts.collections.tournaments",
    ("collections", "vertices"): "scripts.collections.vertices",
    ("data", "dataset"): "scripts.data.dataset",
    ("data", "motifs"): "scripts.data.motifs",
    ("data", "ratings"): "scripts.data.ratings",
    ("data", "shard"): "scripts.data.shard",
    ("data", "umap"): "scripts.data.umap",
    ("eda", "collections"): "scripts.eda.collections",
    ("fetch", "beatmaps"): "scripts.fetch.maps",
    ("fetch", "beatmapsets"): "scripts.fetch.sets",
    ("fetch", "osu"): "scripts.fetch.files",
}

ALIASES = {
    ("collections", "edge"): ("collections", "edges"),
    ("collections", "tournament"): ("collections", "tournaments"),
    ("collections", "vertex"): ("collections", "vertices"),
    ("data", "lightgcn"): ("data", "graph"),
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        groups = sorted({group for group, _command in GROUP_COMMANDS})
        print("Usage: bobert <command> [args]")
        print("       bobert <group> <command> [args]")
        print()
        print("Commands:")
        for command in sorted(MODEL_COMMANDS):
            print(f"  {command}")
        print()
        print("Groups:")
        for group in groups:
            print(f"  {group}")
        return 0

    module = MODEL_COMMANDS.get(sys.argv[1])
    if module is not None:
        sys.argv = [f"bobert {sys.argv[1]}", *sys.argv[2:]]
        runpy.run_module(module, run_name="__main__")
        return 0

    if len(sys.argv) < 3:
        print(f"Unknown command: {sys.argv[1]}", file=sys.stderr)
        return 2

    key = ALIASES.get((sys.argv[1], sys.argv[2]), (sys.argv[1], sys.argv[2]))
    module = GROUP_COMMANDS.get(key)
    if module is None:
        print(f"Unknown command: {' '.join(sys.argv[1:3])}", file=sys.stderr)
        return 2

    sys.argv = [f"bobert {' '.join(key)}", *sys.argv[3:]]
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
