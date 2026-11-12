# Pipeline

Run commands from the repository root with `uv run <command>`. Each has `--help`.

```text
sources → features + strains → pretrain → embed → adapt → labels + search → publish → deploy
                                                   ↑
                               collection data ────┘
```

## Commands

| Stage | Command | Output |
| --- | --- | --- |
| Update | `update` | Runs sync, import, download, features, strains, embed and search for a run |
| Sources | `sync-beatmaps` | `data/beatmaps.parquet`; new or changed IDs in `data/redownload.txt` |
| Sources | `import-beatmaps` | Raw maps from a local or streamed `.tar` archive |
| Sources | `download-beatmaps` | Raw maps under `data/beatmaps/` |
| Sources | `fetch-beatmapsets` | `data/beatmapsets.parquet` |
| Collections | `fetch-collection-vertices`, `fetch-collection-edges` | `data/collections/` |
| Dataset | `build-features` | `data/features/{mode}_features.bin` |
| Dataset | `build-strains` | `data/strains.parquet` |
| Model | `pretrain` | A run with `model.safetensors` and `training.yaml` |
| Model | `embed` | A new shard of the maps the run lacks, and densities |
| Model | `adapt` | A new run with the adapted model and index |
| Model | `build-labels` | `labels.safetensors`, which names play groups (std) |
| Release | `build-search` | `search.bin` per mode |
| Release | `publish` | Hugging Face commit and immutable tag |
| Release | `deploy` | Server switched to a published tag |
| Evaluation | `evaluate` | Retrieval report for one or more runs |
| Evaluation | `mine` | Neighbours one run finds and another misses; `mined.csv` for `adapt` |

## Credentials

| Commands | `.env` |
| --- | --- |
| osu! metadata | `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET` |
| Beatconnect downloads | `BEATCONNECT_API_TOKEN` |
| Collection edges | `osu_session`, `osu_stats_session`, `session_data`, `xsrf_token` |
| `publish` | `HF_TOKEN` with write access |
| `deploy` | `DEPLOY_SSH_TARGET`, `DEPLOY_REMOTE_ROOT` |

## Data

Raw maps are stored as `data/beatmaps/{00-99}/{id}.osu.zst`. `build-features` writes one memory-mapped file per mode with one zstd block per map. Filters and the validation split apply at load time, so only feature changes need a rebuild. `--update` parses only new maps.

| Mode | Token | Features |
| --- | --- | --- |
| std | Hit object | Position, jump, timing, slider and spinner shape ([`core/features.py`](../core/features.py)) |
| taiko | Hit object | Timing, note type, big notes, drumroll and swell length |
| catch | Hit object | Position, movement, timing, juice stream and banana shape |
| mania | Time where a note starts or a hold ends | Timing, key count, and the state of each column in hand-relative slots |

[`core/modes.py`](../core/modes.py) declares each mode's feature groups, strain targets and loss weights.

## Runs

`model.safetensors` holds the encoder, pooling means, adapter and normalization, with its config in the header. The first `embed` computes the pooling means. After that, a map's vector depends only on the model and the map, so `embed` appends a shard for missing maps only. A later shard replaces earlier rows of the same map.

Start experiments from `configs/pretrain.yaml` and `configs/adapt.yaml`; `configs/local*.yaml` is ignored by Git.

## Updates and releases

```sh
uv run update v16 --archive https://data.ppy.sh/2026_11_05_osu_files.tar.bz2
uv run publish v16 --tag v16.1
uv run deploy -v v16.1
```

Hugging Face and the server transfer only changed files. Tags are immutable, so each deploy needs a new one.
