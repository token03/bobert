import os
from pathlib import Path

import zstandard as zstd

from core.artifacts import replace_atomically

SUFFIX = ".osu.zst"


def map_id(path: str | Path) -> int:
    return int(Path(path).name.removesuffix(SUFFIX))


def map_path(beatmap_id: int, root: str | Path) -> Path:
    return Path(root) / str(beatmap_id)[-2:].zfill(2) / f"{beatmap_id}{SUFFIX}"


def read_map(path: str | Path) -> bytes:
    return zstd.ZstdDecompressor().decompress(Path(path).read_bytes())


def write_map(path: str | Path, data: bytes) -> None:
    compressed = zstd.ZstdCompressor(level=9, write_checksum=True).compress(data)
    replace_atomically(Path(path), lambda temporary: temporary.write_bytes(compressed))


def find_maps(root: str | Path) -> list[Path]:
    files = [
        Path(directory) / name
        for directory, _, names in os.walk(root)
        for name in names
        if name.endswith(SUFFIX)
    ]
    return sorted(files, key=map_id)
