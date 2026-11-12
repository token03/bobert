from collections import Counter
from collections.abc import Iterable

MOD_ORDER = ("EZ", "HT", "HD", "HR", "DT", "FL")
ALIASES = {"NC": "DT", "DC": "HT"}
SPEED_MODS = frozenset({"DT", "HT"})


def normalize_mods(mods: Iterable[str]) -> frozenset[str]:
    return frozenset(ALIASES.get(mod, mod) for mod in mods)


def mods_name(mods: Iterable[str]) -> str:
    mods = set(mods)
    return "".join(mod for mod in MOD_ORDER if mod in mods)


def shared_mods(plays: list[frozenset[str]]) -> frozenset[str]:
    counts = Counter(mod for mods in plays for mod in mods)
    return frozenset(mod for mod, count in counts.items() if 2 * count >= len(plays))
