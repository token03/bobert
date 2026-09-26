import hashlib
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from scripts.common.paths import PROJECT_ROOT

MODULE_NAME = "osu_native"
SOURCE = PROJECT_ROOT / "core" / "osu.py"
CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    / "bobert"
    / "native"
)


def _cache_key() -> str:
    import mypy.version

    digest = hashlib.sha256(SOURCE.read_bytes())
    digest.update(sys.version.encode())
    digest.update(mypy.version.__version__.encode())
    return digest.hexdigest()[:16]


def compile_parser() -> Path | None:
    try:
        directory = CACHE_DIR / _cache_key()
    except ImportError:
        print("Warning: mypy is not installed; parsing with pure Python")
        return None
    if any(directory.glob(f"{MODULE_NAME}.*.so")):
        return directory

    staging = directory.with_name(directory.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shutil.copyfile(SOURCE, staging / f"{MODULE_NAME}.py")
    print("Compiling the .osu parser with mypyc...")
    result = subprocess.run(
        [sys.executable, "-m", "mypyc", f"{MODULE_NAME}.py"],
        cwd=staging,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        print(
            "Warning: mypyc compilation failed; parsing with pure Python\n"
            + result.stdout[-2000:]
            + result.stderr[-2000:]
        )
        shutil.rmtree(staging, ignore_errors=True)
        return None
    shutil.rmtree(staging / "build", ignore_errors=True)
    shutil.rmtree(directory, ignore_errors=True)
    staging.rename(directory)
    return directory


def load_parser(directory: str | None) -> ModuleType:
    if directory is None:
        return importlib.import_module("core.osu")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module(MODULE_NAME)
