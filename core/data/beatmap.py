
from typing import Dict, List, NamedTuple

DIFFICULTY_ATTRIBUTES = [
    'stars', 'aim', 'speed', 'slider_factor',
    'cs', 'ar', 'slider_multiplier'
]


class Metadata(NamedTuple):
    # identifiers
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

    # binned numerical 
    bpm: float
    total_length: int
    max_combo: int
    play_count: int
    favourite_count: int

    # categorical with unk
    user_id: int
    artist: str
    source: str
    genre: str
    language: str

    # categorical
    year: int
    status: int 

    # tags
    mapper_tags: List[str]
    weighted_user_tags: Dict[str, float]
    weighted_collection_tags: Dict[str, float]

