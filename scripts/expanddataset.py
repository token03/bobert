from ossapi import Ossapi, serialize_model
import time
from functools import wraps

api = Ossapi(44046, "v3vXLyPfrnVo2ZrMjp5rxjFVqx9W6h2pcb7OfVBd")  # Note: API key should be revoked

# Example usage for similarity search preparation
# from core.data.transforms import BeatmapNormalizer

# res = api.search_beatmapsets(query="favourites>30 star>5", mode=0, category="graveyard")
res = api.search_beatmapsets(query="tag=\"\"aim/flow\"\"", mode=0, category="graveyard")

print(res.total)

# res = api.beatmapset(beatmap_id=4881714)

# print(res)

# for beatmap in res.beatmaps:
#     print(beatmap.version, beatmap.playcount, beatmap.difficulty_rating)

# Example of how to use transforms for new beatmap processing:
# normalizer = BeatmapNormalizer.from_saved_stats(...)  # Load saved normalizer
# new_beatmap_vectors, new_beatmap_metadata = process_new_beatmap(...)
# normalized_vectors, normalized_metadata = normalizer.normalize_vectors(new_beatmap_vectors), normalizer.normalize_metadata(new_beatmap_metadata)