# BoBERT API

FastAPI service for recommendations, metadata lookup, and embedding previously unindexed query maps. The retrieval runtime uses NumPy, Polars, and one lazily loaded CPU PyTorch encoder per mode. Each mode keeps its own index, centering transform, vector width and cache identity. Catalogs are loaded once at startup.

## Run

From the repository root, prepare the artifacts below and configure osu! OAuth credentials in `.env` using [`.env.example`](../.env.example):

```sh
docker compose up --build api
curl http://127.0.0.1:8008/health
```

Compose exposes container port 8000 on host loopback port 8008. The default `compose.yaml` contains only the API; `compose.prod.yaml` adds the production tunnel and host-specific settings.

For a direct Python launch, install with `uv sync --no-default-groups --group serve`, set the path variables below to local paths, and run `uv run --no-default-groups --group serve python -m server.app`. Default runtime paths are container paths under `/app`.

## Artifact contract

| File | Role |
| --- | --- |
| `data/beatmaps.parquet` | Beatmap metadata and search/filter columns |
| `data/beatmapsets.parquet` | Beatmapset metadata |
| `data/strains.parquet` | Difficulty and strain catalog |
| `runs/current/{mode}/model[.a].safetensors` | Encoder and adapter weights, normalization tensors, model configuration and mode in the header |
| `runs/current/{mode}/embeddings[.a].parquet` | IDs, vectors, densities; pooling and retrieval settings in file metadata |

`runs/current/release.json` must contain `{"format":1,"modes":{"std":"adapted","taiko":"adapted","catch":"adapted","mania":"adapted"}}`. A run may select a subset of modes. Every selected mode needs its matching model and index; flat legacy runs are not supported.

Use artifacts from the same export. The runtime reads both files through [`core/artifacts.py`](../core/artifacts.py) and checks the model checksum recorded in the index. Missing or mismatched artifacts prevent startup.

Compose mounts `data/` and `runs/` read-only and uses a named Docker volume for the writable SQLite cache. The image runs as a non-root user. `runs/current` selects the active multimode release; the catalogs in `data/` are shared. Separate per-mode SQLite caches hold online embeddings, metadata, and unavailable beatmaps; it is local runtime state.

## Configuration

| Variable | Default / requirement |
| --- | --- |
| `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET` | Required osu! OAuth credentials |
| `BOBERT_DATA_DIR` | `/app/data` |
| `BOBERT_RUN_DIR` | `/app/runs/current` |
| `BOBERT_CACHE_DB` | `/app/cache/runtime.sqlite` (filename stem; mode suffix added) |
| `TORCH_NUM_THREADS` | `2`; application caps threads at two and available CPU count |

Compose explicitly passes its configured environment to the container. For custom paths, edit the service environment and mounts together.

The production memory limit is 4.5 GiB for four indexes and lazily loaded encoders. Provision enough host RAM for that limit plus the OS and tunnel. Online inference is serialized across modes to bound peak memory.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service health |
| GET | `/api/modes` | Loaded rulesets |
| GET | `/api/recommend` | Default recommendation list |
| POST | `/api/recommend` | Recommendations from source beatmaps in one mode |
| GET | `/api/beatmaps/{beatmap_id}` | Beatmap metadata |
| GET | `/api/beatmaps/{beatmap_id}/summary` | Normalized summary |
| GET | `/api/docs` | Interactive Swagger UI |
| GET | `/api/openapi.json` | OpenAPI schema |

```sh
curl http://127.0.0.1:8008/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"beatmap_ids":[75],"top_k":20,"filters":{"min_sr":3,"max_sr":7}}'
```

GET recommendations and summary/detail routes accept `?mode=std|taiko|catch|mania`. POST recommendations infer the mode from the first source when `mode` is omitted; catalog and cached maps resolve locally, and unknown IDs use osu! metadata. An explicit `mode` restricts the query to that ruleset. All sources must share one mode; mixed-mode queries return 422 without poisoning the unavailable-map cache. Responses include the canonical mode.

Requests accept 1–10 positive beatmap IDs and up to 1,000 results. Filters cover difficulty, object attributes, tempo, length, status, and dates; consult the API documentation for the full schema. Responses include source metadata, cache status, and scored results.

## Hosted deployment

The hosted request path is Cloudflare Pages → Pages Function → gateway Worker → VPC service / Tunnel → API. The Worker applies burst and sustained recommendation rate limits. Local Vite development proxies directly to port 8008.

[`scripts/deploy.py`](../scripts/deploy.py) checks that a release tag exists on Hugging Face, then runs its shell companion on the server. The server pulls the code, downloads all manifest-selected mode artifacts and shared catalogs, rebuilds the API image, validates the downloaded artifacts, switches `runs/current` and `data/`, waits for health, and attempts rollback on failure. It targets an already provisioned deployment. `DEPLOY_SSH_TARGET`, `DEPLOY_REMOTE_ROOT`, and `CLOUDFLARE_TUNNEL_TOKEN` configure this workflow.

```sh
uv run publish-run -v my-run
uv run deploy -v my-run
```

Run both commands from your local checkout. `publish-run` uploads the release to Hugging Face. `deploy` connects to the VPS over SSH and runs the downloads there; models, embeddings, and catalogs go directly from Hugging Face to the VPS. No SCP or local `fetch-run` step is needed. The configured SSH target and remote root are sufficient for the existing provisioned host; public releases need no Hugging Face token on the VPS.

Publishing is required first; deployments consume the released tag. The production overlay retains the IPv6-only tunnel routing used by the hosted instance. Adjust it for your network. Catalogs are swapped with the release, so rollback restores the complete artifact set.
