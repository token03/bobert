# BoBERT API

FastAPI service for recommendations, metadata lookup, and embedding previously unindexed query maps. The retrieval runtime uses NumPy, Polars, and a CPU PyTorch encoder.

## Run

From the repository root, prepare the artifacts below and configure osu! OAuth credentials in `.env` using [`.env.example`](../.env.example):

```sh
docker compose up --build api
curl http://127.0.0.1:8008/health
```

Compose exposes container port 8000 on host loopback port 8008. Selecting `api` starts the local service without the production tunnel.

For a direct Python launch, install the Python dependencies and set the path variables below to local paths before running `python -m server.app`. Default runtime paths are container paths under `/app`.

## Artifact contract

| File | Role |
| --- | --- |
| `data/beatmaps.parquet` | Beatmap metadata and search/filter columns |
| `data/beatmapsets.parquet` | Beatmapset metadata |
| `data/strains.parquet` | Difficulty and strain catalog |
| `runs/current/bobert.pt` | Encoder state, model arguments, feature normalization statistics |
| `runs/current/embeddings.parquet` | `beatmap_id`, fixed-size `embedding` vectors, `density` values |
| `runs/current/embeddings.json` | Layer centering, adapter metadata, CSLS retrieval settings |

Use artifacts from the same export. The runtime reads the index during startup and validates its shape and retrieval metadata. Missing files prevent startup. The model is loaded when online encoding is needed.

Compose mounts `data/` and `runs/` read-only and `cache/` read-write. `runs/current` selects the deployed run. SQLite caches online embeddings, metadata, and unavailable beatmaps; it is local runtime state.

## Configuration

| Variable | Default / requirement |
| --- | --- |
| `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET` | Required osu! OAuth credentials |
| `BOBERT_DATA_DIR` | `/app/data` |
| `BOBERT_RUN_DIR` | `/app/runs/current` |
| `BOBERT_MODEL_PATH` | `BOBERT_RUN_DIR/bobert.pt` |
| `BOBERT_EMBEDDINGS_PATH` | `BOBERT_RUN_DIR/embeddings.parquet`; sidecar uses the same stem |
| `BOBERT_CACHE_DB` | `/app/cache/runtime.sqlite` |
| `TORCH_NUM_THREADS` | `2`; application caps threads at two and available CPU count |

Compose explicitly passes its configured environment to the container. For custom paths, edit the service environment and mounts together.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service health |
| GET | `/api/recommend` | Default recommendation list |
| POST | `/api/recommend` | Recommendations from source beatmaps |
| GET | `/api/beatmaps/{beatmap_id}` | Beatmap metadata |
| GET | `/api/beatmaps/{beatmap_id}/summary` | Normalized summary |
| GET | `/api/docs` | Interactive Swagger UI |
| GET | `/api/openapi.json` | OpenAPI schema |

```sh
curl http://127.0.0.1:8008/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"beatmap_ids":[75],"top_k":20,"filters":{"min_sr":3,"max_sr":7}}'
```

Requests accept 1–10 positive beatmap IDs and up to 1,000 results. Filters cover difficulty, object attributes, tempo, length, status, and dates; consult the API documentation for the full schema. Responses include source metadata, cache status, and scored results.

## Hosted deployment

The hosted request path is Cloudflare Pages → Pages Function → gateway Worker → VPC service / Tunnel → API. The Worker applies burst and sustained recommendation rate limits. Local Vite development proxies directly to port 8008.

[`scripts/deploy.py`](../scripts/deploy.py) validates and uploads a versioned run over SSH; its shell companion rebuilds the API, switches `runs/current`, waits for health, and attempts rollback on failure. It targets an already provisioned deployment. `DEPLOY_SSH_TARGET`, `DEPLOY_REMOTE_ROOT`, and `CLOUDFLARE_TUNNEL_TOKEN` configure this workflow.
