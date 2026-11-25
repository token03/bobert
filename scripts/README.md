# Pipeline

Run from the repository root with `uv run <command>` (`--help` for flags).

```text
update ─→ features + strains ─→ pretrain ─→ adapt ─→ distill ─→ labels + search ─→ publish ─→ deploy
                                               ↑
                          fetch-collections ───┘
```

## Commands

| Stage | Command | Purpose |
| --- | --- | --- |
| Data | `update` | Incremental osu! sync, `.osu` download, features/strains, embed, and `search.bin` rebuild |
| Data | `fetch-collections` | Refresh osu!collector and osu!stats collections in `data/collections/` |
| Data | `build-features` | Rebuild `data/features/{mode}_features.bin` from stored maps |
| Data | `build-strains` | Compute `data/strains.parquet` |
| Model | `pretrain` | Pretrain a base run |
| Model | `embed` | Embed unindexed maps into a run's `embeddings.parquet` |
| Model | `adapt` | Train collection adapter into a new run |
| Model | `distill` | Prune and distill a run into a smaller student run |
| Model | `build-labels` | Train the play-group label head (`std`) |
| Release | `build-search` | Build `search.bin` per mode |
| Release | `publish` | Upload run and catalogs to Hugging Face under a tag |
| Release | `deploy` | Deploy a published tag to the server |
| Eval | `evaluate` | Retrieval and probe benchmarks |
| Eval | `mine` | Export disagreement neighbors between two runs to `mined.csv` |

## Data & run layout

- `data/beatmaps/{00-99}/{id}.osu.zst` — raw beatmaps
- `data/features/{mode}_features.bin` — extracted features (schema in [`core/modes.py`](../core/modes.py); only rebuild with `build-features` when feature definitions change)
- `runs/<name>/{std,taiko,catch,mania}/` — `config.json`, `model.safetensors`, `embeddings.parquet`, `search.bin`
- `configs/` — base training configs (`configs/local*.yaml` is gitignored)

## Releases

```sh
uv run update v16-mini
uv run publish v16-mini --tag v16-mini-2026-11-25
uv run deploy -v v16-mini-2026-11-25
```

## Credentials

| Consumer | `.env` keys |
| --- | --- |
| osu! API | `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET` |
| Beatconnect | `BEATCONNECT_API_TOKEN` |
| Large osu!stats collections | `osu_session`, `osu_stats_session`, `session_data`, `xsrf_token` |
| `publish` | `HF_TOKEN` |
| `deploy` | `DEPLOY_SSH_TARGET`, `DEPLOY_REMOTE_ROOT` |
