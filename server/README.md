# BoBERT API

FastAPI service for recommendations, osu! login and embedding maps missing from the index. It loads one index per mode at startup and one CPU encoder per mode on first use.

```sh
docker compose up --build api
curl http://127.0.0.1:8008/health
```

`compose.yaml` runs the API and the chat bot from one image; `compose.prod.yaml` adds the tunnel and memory limits.

## Files

| File | Role |
| --- | --- |
| [`app.py`](app.py) | Routes |
| [`runtime.py`](runtime.py) | Indexes, search and the SQLite cache of online embeddings |
| [`osu.py`](osu.py) | osu! API client; all requests share one queue at 1 per second |
| [`auth.py`](auth.py) | Signed session cookie |
| [`plays.py`](plays.py), [`mods.py`](mods.py) | Grouping top plays by mods and style |
| [`bot.py`](bot.py) | Bancho IRC bot |
| [`release.py`](release.py) | Downloads changed files of a release during deploy |

## Configuration

| Variable | Role |
| --- | --- |
| `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET` | osu! OAuth application |
| `OSU_REDIRECT_URI` | `https://<site>/api/auth/callback` |
| `SESSION_SECRET` | Signs session cookies |
| `BOBERT_DATA_DIR`, `BOBERT_RUN_DIR`, `BOBERT_CACHE_DB` | Paths, default under `/app` |
| `OSU_IRC_USERNAME`, `OSU_IRC_PASSWORD` | Bot account |

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/auth/login`, `/api/auth/callback` | osu! login |
| POST | `/api/auth/logout` | Clears the session |
| GET | `/api/me` | Logged-in player |
| GET | `/api/me/groups` | Logged-in player's grouped top plays |
| GET | `/api/search/{mode}` | Client search catalog |
| GET | `/api/recommend` | Default list |
| POST | `/api/recommend` | Recommendations from source maps; at most 5 not yet embedded |
| GET | `/api/players/{username}`, `/api/users/{id}/top` | Bot only |
| POST | `/api/beatmaps/groups` | Bot only |

The full schema is at `/api/docs`.

Login is stateless: the callback reads the player once and stores the ID and name in an HMAC-signed cookie.

## Bot

`/np` returns maps like the current one. `!r` returns maps like one of the sender's top-play groups, optionally filtered by mods (`!r dt`).

## Hosting

Requests go Cloudflare Pages → gateway Worker → Tunnel → API. `uv run deploy -v <tag>` pulls the code on the server, downloads the changed release files, switches `runs/current`, and rolls back if the API does not become healthy.
