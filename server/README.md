# BoBERT API

FastAPI service for beatmap recommendations, osu! OAuth, and on-demand CPU embedding of unindexed maps.

```sh
docker compose up --build api
curl http://127.0.0.1:8008/health
```

`compose.yaml` runs `api` and `bot` from one image; `compose.prod.yaml` adds `cloudflared` and memory limits.

## Files

| File | Role |
| --- | --- |
| [`app.py`](app.py) | Routes and request logging |
| [`runtime.py`](runtime.py) | Indexes, CSLS search, SQLite embedding cache |
| [`osu.py`](osu.py) | osu! API client |
| [`auth.py`](auth.py) | Session cookie signing |
| [`plays.py`](plays.py), [`mods.py`](mods.py) | Top-play grouping |
| [`bot.py`](bot.py) | Bancho IRC bot (`/np`, `!r [mods]`) |
| [`release.py`](release.py) | Release artifact fetcher for deploys |

## Configuration

| Variable | Role |
| --- | --- |
| `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET` | osu! OAuth credentials |
| `OSU_REDIRECT_URI` | `https://<site>/api/auth/callback` |
| `SESSION_SECRET` | Session cookie HMAC key |
| `BOBERT_DATA_DIR`, `BOBERT_RUN_DIR`, `BOBERT_CACHE_DB` | Artifact/cache paths (default `/app`) |
| `OSU_IRC_USERNAME`, `OSU_IRC_PASSWORD` | Bancho IRC bot account |

## Endpoints

Full OpenAPI schema at `/api/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/auth/login`, `/api/auth/callback` | osu! OAuth flow |
| `POST` | `/api/auth/logout` | Clear session |
| `GET` | `/api/me` | Current session |
| `GET` | `/api/me/groups` | Grouped top plays |
| `GET` | `/api/search/{mode}` | Search catalog |
| `GET` | `/api/recommend` | Default recommendations |
| `POST` | `/api/recommend` | Recommendations from source IDs (max 5 unindexed) |
| `GET` | `/api/players/{username}`, `/api/users/{id}/top` | Internal (bot only) |
| `POST` | `/api/beatmaps/groups` | Internal (bot only) |

## Hosting

`Cloudflare Pages → gateway Worker → Cloudflare Tunnel → API` (deploy via `uv run deploy -v <tag>`).
