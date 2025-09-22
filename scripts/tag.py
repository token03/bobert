from dotenv import load_dotenv
import os
from rich import inspect

from ossapi import Ossapi

load_dotenv()  

client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")

api = Ossapi(client_id, client_secret)


beatmap = api.beatmap(221777)
inspect(beatmap, all=True, methods=False)
