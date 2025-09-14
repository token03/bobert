from ossapi import Ossapi, serialize_model
import time
from functools import wraps

api = Ossapi(44046, "v3vXLyPfrnVo2ZrMjp5rxjFVqx9W6h2pcb7OfVBd") # REVOKED THE KEY WHOOPS PUSHED TO REPO

# res = api.search_beatmapsets(query="favourites>30 star>5 star<10", mode=0, category="graveyard")
res = api.search_beatmapsets(query="tag=\"\"aim/flow\"\"", mode=0, category="graveyard")

print(res.total)

# res = api.beatmapset(beatmap_id=4881714)

# print(res)

# for beatmap in res.beatmaps:
#     print(beatmap.version, beatmap.playcount, beatmap.difficulty_rating)