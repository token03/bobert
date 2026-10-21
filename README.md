# BoBERT
**Bidirectional osu! Beatmap Encoder Representations from Transformers**

BoBERT encodes osu! beatmaps as fixed-size vectors. It supports osu!standard, taiko, catch and mania, with one encoder for each ruleset. Each encoder reads the hit objects of a map and outputs one vector. An adapter, trained on collection data, then tunes each encoder for recommendation and similarity search.

The repository contains the data pipeline, the training and adaptation code, the evaluation tools and the web application.

**[Try the search tool](https://bobert-web.pages.dev/)**

## Architecture

BoBERT is a bidirectional Transformer encoder based on [ModernBERT](https://arxiv.org/abs/2412.13663). Each hit object is one token. A tokenizer in the style of [FT-Transformer](https://arxiv.org/abs/2106.11959) embeds the features of each hit object. Each ruleset has its own feature set. The encoder alternates local and global attention. The beatmap vector is the mean of the outputs of the global layers.

![BoBERT architecture: a per-ruleset feature tokenizer embeds each hit object's feature groups as 16-d vectors, concatenates them and projects them to a 384-d token. Tokens pass through nine pre-norm attention and SwiGLU blocks, with ±128-object local attention except at layers 3, 6 and 9, which attend globally. Those three layers are mean-pooled and passed through a linear adapter to give the 384-d beatmap vector.](docs/assets/architecture.png)

### Encoder

| Setting | Default in [`configs/pretrain.yaml`](configs/pretrain.yaml) |
| --- | --- |
| Parameters | 16.1M |
| Depth / width / heads | 9 layers / 384 / 6 |
| Sequence length limit | 4,096 hit objects |
| Attention | Global at layers 3, 6 and 9; ±128 hit objects at all other layers |

### Data

| Ruleset | Pretraining maps | Search index |
| --- | ---: | ---: |
| osu!standard | 692k | 499k |
| osu!taiko | 78k | 95k |
| osu!catch | 20k | 25k |
| osu!mania | 278k | 315k |

### Pretraining

- **Masked reconstruction.** The model masks 30% of the hit objects in short spans and predicts all of their features. Masking follows the 80/10/10 recipe from BERT. The features of the neighbors at the span borders are also corrupted.
- **Strain regression.** A linear head predicts difficulty targets from the pooled vector. The targets are skill strains for osu!standard and star rating for the other rulesets.

### Adaptation

A linear adapter transforms the pretrained vector. The adapter starts as the identity. Contrastive training on pairs of maps that share collections then updates it. If a file of mined pairs is available, a second phase trains on these pairs.

## Limitations

- The model reads hit objects only. It does not read AR, CS, OD or HP, so the vectors do not include these settings.
- Maps with few hit objects give the model less data, so their nearest neighbors are less reliable.
- The encoder reads the first 4,096 hit objects only.
- 1/3 and 1/4 rhythms are in different regions of the vector space. Thus, the same stream in 1/3 and in 1/4 is not similar.
- The rulesets share the architecture but not the weights, tokenizer, heads or normalization. Search compares maps of the same ruleset only.

## Run locally

The API runs on CPU in Docker. The frontend uses Bun. In local development, Vite connects directly to the API.

### Get the artifacts

Download a model release, the embedding index and the metadata catalogs from [Hugging Face](https://huggingface.co/token03/bobert):

```sh
uv run --no-default-groups --group serve fetch-run --revision v15
```

This command uses the CPU dependencies. It puts the run files in `runs/` and the catalogs in `data/`:

```text
data/
  beatmaps.parquet
  beatmapsets.parquet
  strains.parquet
runs/
  current -> <run>
  v15/
    release.json
    std/
      model.safetensors
      embeddings.parquet
      model.a.safetensors
      embeddings.a.parquet
      training.yaml
    taiko/
    catch/
    mania/
```

All runs use the same catalogs. Each ruleset folder contains the base files and the collection-adapted (`.a`) files. `release.json` selects the variant that the server uses for each ruleset. A published release contains only the selected files. Releases v15 and later use this format.

### Start the API and the frontend

Copy [`.env.example`](.env.example) to `.env`. Set `OSU_CLIENT_ID` and `OSU_CLIENT_SECRET` from an [osu! OAuth application](https://osu.ppy.sh/home/account/edit). Then, from the repository root:

```sh
docker compose up --build api
```

The API is at `http://127.0.0.1:8008`. The interactive documentation is at `/api/docs`. For configuration and endpoints, refer to the [server guide](server/README.md).

From `web/`:

```sh
bun install
bun run dev
```

Open the address that Vite shows. After you install the frontend dependencies, `mise run dev` starts both services from the repository root. For more information, refer to [web development](web/README.md).

For the existing VPS, run `uv run publish-run -v v15`, then `uv run deploy -v v15` from your local checkout when ready to deploy. The VPS downloads the published artifacts directly from Hugging Face, so no SCP step is needed. See the [hosted deployment guide](server/README.md#hosted-deployment) for configuration and rollback behavior.

## Training

The Python environment targets **Linux x86-64, Python 3.12, PyTorch 2.11 and CUDA 12.8**, with a pinned FlashAttention wheel. Install it with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --frozen
uv run pretrain --help
```

Build the features (`build-features`) and the strain targets (`build-strains`). Then train and embed a named run:

```sh
uv run pretrain --mode std --profile full -v my-run
uv run embed -v my-run --mode std --variant base
```

`pretrain` saves the run configuration and exports `model.safetensors`. `export-model` exports existing checkpoints. `adapt` trains the adapter. `evaluate`, `mine` and `umap` are for retrieval analysis. The [scripts guide](scripts/README.md) lists the inputs and outputs of each command.

`uv sync` installs the default CUDA training group. For CPU serving and release downloads, use `uv sync --no-default-groups --group serve`. `uv.lock` contains both environments.

## Repository guide

| Directory | Contents |
| --- | --- |
| [`core/`](core/) | Beatmap parsing, features, encoder and dataset primitives |
| [`training/`](training/) | Lightning modules, data loading, objectives and optimizers |
| [`scripts/`](scripts/README.md) | Data download, feature extraction, training entry points and evaluation |
| [`server/`](server/README.md) | FastAPI application, retrieval runtime and embedding cache |
| [`web/`](web/README.md) | React application and Cloudflare gateway |
| `data/`, `runs/`, `cache/` | Local datasets, model artifacts and runtime state |

## Acknowledgements

- [MusicBERT](https://arxiv.org/abs/2106.05630): representation learning from symbolic music, a major inspiration.
- [CM3P](https://github.com/OliBomby/CM3P) by OliBomby: inspiration for the encoder architecture.
- [ModernBERT](https://github.com/AnswerDotAI/ModernBERT): bidirectional encoders with rotary embeddings and alternating attention.
- [FT-Transformer](https://github.com/yandex-research/rtdl-revisiting-models): feature-wise embeddings for numerical and categorical inputs.
- [parsecore](https://github.com/Apart-Studio/parsecore): star ratings and strain values for the pretraining targets.
- [osu!collector](https://osucollector.com/) and [osu!stats](https://osustats.ppy.sh/): collection data for adaptation and evaluation.
- [Beatconnect](https://beatconnect.io/): `.osu` files for the dataset.
- [osu!](https://osu.ppy.sh/) and its mapping community: the beatmaps and metadata.

## License

The code is available under the [MIT License](LICENSE).
