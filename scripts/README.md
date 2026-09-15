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
| Model | `pretrain` | Dataset, strains, `configs/default.yaml` | Run config, checkpoints, exported `model.safetensors` |
| Model | `export-model` | Training checkpoint and matching config | Inference `model.safetensors` |
| Model | `embed` | Exported encoder and dataset | `embeddings.parquet`, including pooling metadata and density index |
| Model | `adapt` | Source run and collection-derived pairs | Target run with trained linear adapter |
| Release | `fetch-run` | Hugging Face repository and release tag | Run under `runs/`, catalogs under `data/`; updates `runs/current` |
| Release | `publish-run` | Run, catalogs, Hugging Face write token | Model card, artifacts, and immutable release tag |
| Evaluation | `evaluate` | Run embeddings, catalogs, evaluation labels | Per-run evaluation JSON and terminal report |
| Evaluation | `recommend` | Source maps / mappers and embedding index | Ranked recommendations |
| Evaluation | `mine` | Retrieval inputs and evaluation data | Mined retrieval examples |
| Evaluation | `umap` | Embeddings and metadata | Visualization coordinates, attributes, and neighbors |
| Evaluation | `build-eval-graph` | Collection edges, vertices, ngrams, catalogs | Graph embeddings in `data/graph.parquet` |
| Web | `build-search` | Embedding index, beatmap catalog | `web/public/search.bin` client search artifact |
| Evaluation | `build-eval-ngrams` | Collection vertices and edges | `data/collections/ngrams.txt` |
| Deployment | `deploy` | Published release tag, SSH configuration | Fetches the release on the server, rebuilds the API, and swaps the run and catalogs |

## Credentials

Start with [`.env.example`](../.env.example). Credentials depend on the selected source; local dataset building and training do not require network credentials once inputs are available.

| Commands / source | Environment |
| --- | --- |
| osu! metadata acquisition, mapper fetching, online recommendation queries | `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET` |
| Beatconnect acquisition routes | `BEATCONNECT_API_TOKEN` |
| Collection-edge acquisition | Source-specific cookies: `osu_session`, `osu_stats_session`, `session_data`, `xsrf_token` |
| `deploy` | `DEPLOY_SSH_TARGET`, `DEPLOY_REMOTE_ROOT` |
| Hosted Compose tunnel | `CLOUDFLARE_TUNNEL_TOKEN` |
| `publish-run` | `HF_TOKEN` (write access); project `.env` takes precedence over the shell |

## Training runs

After preparing `data/dataset/` and `data/strains.parquet`:

```sh
uv run pretrain --config configs/default.yaml --full -v my-run
uv run embed -v my-run
```

Use `pretrain --help` for proxy/validation modes, batch-size overrides, and checkpoint resume. The training runner saves the resolved configuration in the run directory. Keep it with the checkpoints: exporting a checkpoint requires its matching configuration and normalization statistics.

An export stores encoder/adapter weights and feature normalization tensors in `model.safetensors`, with model configuration and the feature schema in its header. `embeddings.parquet` carries its own corpus-centering statistics, adapter provenance, retrieval settings, and model checksum. There are no JSON sidecars. Re-embed after changing the encoder or adapter. Lightning `.ckpt` files remain training checkpoints; the inference loader accepts safetensors only.

Use `configs/default.yaml` as the starting point for experiments. Local copies named `configs/local*.yaml` are ignored; each run saves its resolved configuration as `training.yaml`. This snapshot is also included in published releases. For older checkpoints, pass their saved configuration explicitly with `export-model --config <path>`.

## Releases

```sh
uv run publish-run -v my-run --data-dir data
uv run --no-default-groups --group serve fetch-run --revision my-run
```

Both commands default to `token03/bobert`; override with `--repo` or `BOBERT_HF_REPO`. Publishing uploads the model, index, and three catalogs in one commit and tags that commit. Existing tags cannot be overwritten. `fetch-run` resolves a tag to a commit, validates the download, then activates it atomically. Use `--version` for another local run name, `--runs-dir` for a separate runs directory, or `--data-dir` for another catalog directory. Existing run directories are preserved; shared catalogs are overwritten by design.

Run artifacts stay in `runs/<version>/`; catalogs live in the workspace's `data/` directory and are shared across runs.

`deploy -v <tag>` runs on the provisioned server: it pulls the code, downloads that release from Hugging Face with `curl`, rebuilds the API image, swaps the run and catalogs, and rolls back on a failed health check. It requires the release to be published first and uses `DEPLOY_SSH_TARGET` and `DEPLOY_REMOTE_ROOT`. The disposable SQLite cache is reset on every switch so container users cannot conflict over stale files. Rollback only needs the previous run directory, which is retained on the server.

For service setup, see the [API guide](../server/README.md). The deployment command updates an existing host; initial provisioning happens separately.
