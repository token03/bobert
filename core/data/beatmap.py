from typing import Dict, List, NamedTuple

DIFFICULTY_ATTRIBUTES = [
    "stars",
    "aim",
    "speed",
    "slider_factor",
    "cs",
    "ar",
    "slider_multiplier",
]

GENRE_NAMES = [
    "unspecified",
    "video_game",
    "anime",
    "rock",
    "pop",
    "other",
    "novelty",
    "hip_hop",
    "electronic",
    "metal",
    "classical",
    "folk",
    "jazz",
]

LANGUAGE_NAMES = [
    "english",
    "chinese",
    "french",
    "german",
    "italian",
    "japanese",
    "korean",
    "spanish",
    "swedish",
    "russian",
    "polish",
    "other",
    "instrumental",
    "unspecified",
]

YEAR_CATEGORIES = [
    "pre-2009",
    "2009",
    "2010",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025+",
]

BPM_CATEGORIES = [
    "0-59",
    "60-79",
    "80-99",
    "100-119",
    "120-139",
    "140-159",
    "160-179",
    "180-199",
    "200-219",
    "220-239",
    "240-259",
    "260-279",
    "280-299",
    "300+",
]

STATUS_CATEGORIES = [
    # we treat pending and wip as graveyard
    # we also treat qualified as ranked
    "graveyard",
    "ranked",
    "approved",
    "loved",
]


class Beatmap(NamedTuple):
    # identifiers, not embedded
    beatmap_id: int
    beatmapset_id: int

    # piecewise linear
    difficulty_rating: float
    aim_rating: float
    speed_rating: float
    slider_factor: float
    ar: float
    cs: float
    od: float
    hp: float

    # manual numerical bins
    bpm: float
    year: int

    # data-derived numerical bins
    total_length: int
    max_combo: int
    play_count: int
    favourite_count: int

    # categorical with unk
    mapper: int
    artist: str
    source: str
    genre: str
    language: str

    # categorical
    status: int

    # tags + weighted tags
    mapper_tags: List[str]
    user_tags: Dict[str, float]
    collection_topics: Dict[str, float]
