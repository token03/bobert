import os
import tempfile
from pathlib import Path

import zstandard as zstd


def map_id(path: str | Path) -> str:
    return Path(path).name.removesuffix(".zst").removesuffix(".osu")


def map_path(beatmap_id: str | int, root: str | Path) -> Path:
    shard = str(beatmap_id)[-2:].zfill(2)
    path = Path(root) / shard / f"{beatmap_id}.osu.zst"
    plain = path.with_suffix("")
    return plain if not path.exists() and plain.exists() else path


def read_map(path: str | Path) -> bytes:
    path = Path(path)
    data = path.read_bytes()
    return zstd.ZstdDecompressor().decompress(data) if path.suffix == ".zst" else data


def write_map(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    if path.suffix != ".zst":
        path = path.with_name(f"{path.name}.zst")
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = zstd.ZstdCompressor(level=9, write_checksum=True).compress(data)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
        try:
            file.write(compressed)
            file.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def find_maps(root: str | Path, recursive: bool = True) -> list[Path]:
    files = []
    for directory, _, names in os.walk(root):
        present = set(names)
        files.extend(
            Path(directory) / name
            for name in names
            if name.endswith(".osu.zst")
            or (name.endswith(".osu") and f"{name}.zst" not in present)
        )
        if not recursive:
            break
    return sorted(
        files,
        key=lambda path: (
            int(map_id(path)) if map_id(path).isdecimal() else 2**63,
            str(path),
        ),
    )
