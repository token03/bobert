# Pipeline and development tools

Run commands from the repository root with `uv run <command>`. Install the development environment with `uv sync --frozen`; its current target is Linux x86-64, Python 3.12, and CUDA 12.8. Most commands expose `--help`; some acquisition tools are interactive.

```text
sources → dataset + strain targets → pretrain → embed → recommend / evaluate
                                             ↓
collection data → adaptation pairs → adapt → embed
```

`core/` implements the encoder and features; `training/` implements learning objectives and data loading. These scripts connect them to files, external sources, and command-line workflows. Shared acquisition, query, and path helpers live in `common/`.

## Command map

Paths below describe the default workspace layout. Check each command's arguments for output overrides and source selection.

| Stage | Command | Inputs | Outputs / effect |
| --- | --- | --- | --- |
| Sources | `fetch-beatmaps` | Source IDs and metadata APIs | `data/beatmaps.parquet` |
| Sources | `download-beatmaps` | Beatmap catalog and download sources | Local `.osu` files under `data/beatmaps/` |
| Sources | `import-beatmaps` | Local tar archive of `.osu` files | Imported beatmap files |
| Sources | `fetch-beatmapsets` | Beatmap catalog and metadata APIs | `data/beatmapsets.parquet` |
| Sources | `fetch-mapper` | Mapper selection and osu! API | Updates beatmap catalog and fetch progress |
| Sources | `shard-beatmaps` | Local beatmap files | Beatmap archive shards |
| Collections | `fetch-collection-vertices` | Collection sources | `data/collections/vertices.parquet` |
| Collections | `fetch-collection-edges` | Collection vertices and source access | `data/collections/edges.parquet` |
| Collections | `fetch-tournaments` | osu!collector tournament API | `data/collections/tournaments.parquet` |
| Dataset | `build-dataset` | Beatmap files and metadata | Prepared dataset under `data/dataset/` |
| Dataset | `build-strains` | Dataset / raw beatmaps, config, parsecore | `data/strains.parquet` |
| Model | `pretrain` | Dataset, strains, `config.yaml` | Run config, checkpoints, exported `bobert.pt` |
| Model | `export-model` | Training checkpoint and matching config | Inference `bobert.pt` |
| Model | `embed` | Exported encoder and dataset | `embeddings.parquet` and `embeddings.json`, including density index |
| Model | `adapt` | Source run and collection-derived pairs | Target run with trained linear adapter |
| Evaluation | `evaluate` | Run embeddings, catalogs, evaluation labels | Per-run evaluation JSON and terminal report |
| Evaluation | `recommend` | Source maps / mappers and embedding index | Ranked recommendations |
| Evaluation | `mine` | Retrieval inputs and evaluation data | Mined retrieval examples |
| Evaluation | `umap` | Embeddings and metadata | Visualization coordinates, attributes, and neighbors |
| Evaluation | `build-eval-graph` | Collection edges, vertices, ngrams, catalogs | Graph embeddings in `data/graph.parquet` |
| Evaluation | `build-eval-ngrams` | Collection vertices and edges | `data/collections/ngrams.txt` |
| Deployment | `deploy` | Versioned run, optional catalogs, SSH configuration | Upload and activation on a provisioned server |

## Credentials

Start with [`.env.example`](../.env.example). Credentials depend on the selected source; local dataset building and training do not require network credentials once inputs are available.

| Commands / source | Environment |
| --- | --- |
| osu! metadata acquisition, mapper fetching, online recommendation queries | `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET` |
| Beatconnect acquisition routes | `BEATCONNECT_API_TOKEN` |
| Collection-edge acquisition | Source-specific cookies: `osu_session`, `osu_stats_session`, `session_data`, `xsrf_token` |
| `deploy` | `DEPLOY_SSH_TARGET`, `DEPLOY_REMOTE_ROOT` |
| Hosted Compose tunnel | `CLOUDFLARE_TUNNEL_TOKEN` |

## Training runs

After preparing `data/dataset/` and `data/strains.parquet`:

```sh
uv run pretrain --config config.yaml --full -v my-run
uv run embed -v my-run
```

Use `pretrain --help` for proxy/validation modes, batch-size overrides, and checkpoint resume. The training runner saves the resolved configuration in the run directory. Keep it with the checkpoints: exporting a checkpoint requires its matching configuration and normalization statistics.

An export consists of encoder weights and feature normalization statistics in `bobert.pt`. Embedding export adds a corpus-dependent transform and retrieval metadata in `embeddings.json`; keep it paired with `embeddings.parquet`. Re-embed after changing the encoder or adapter.

For service setup, see the [API guide](../server/README.md). The deployment command updates an existing host; initial provisioning and artifact preparation happen separately.
