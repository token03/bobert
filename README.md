# BoBERT
**Bidirectional osu! Beatmap Encoder Representations from Transformers**

BoBERT learns dense representations of osu!standard beatmaps from their hit-object sequences. Spatial patterns, rhythm, and slider geometry become fixed-size embeddings. The encoder was pretrained on roughly **500,000 beatmaps**, then adapted for recommendation and similarity search using labelled data from thousands of collections.

While recommendation and similarity search motivated the project, the pretrained embeddings can be adapted to a wider range of representation-learning tasks, including classification, clustering, and attribute prediction. The repository includes the data pipeline, encoder training and adaptation, evaluation tools, and a web application.

**[Click here to try the search tool](https://bobert-web.pages.dev/)**

## Architecture

BoBERT is a bidirectional Transformer encoder in the style of [ModernBERT](https://arxiv.org/abs/2412.13663), with one token per hit object. An [FT-Transformer](https://arxiv.org/abs/2106.11959)-style tokenizer embeds each object's spatial, rhythmic, and slider features, the encoder mixes local and global attention, and pooled outputs from the global layers become a 384-dimensional beatmap vector.

![BoBERT architecture: each hit object's 28 features are grouped into six continuous and five categorical 16-d embeddings, concatenated and projected to a 384-d token. Tokens pass through nine pre-norm attention and SwiGLU blocks, with ±128-object local attention except at layers 3, 6 and 9, which attend globally. Those three layers are mean-pooled and passed through a linear adapter to give the 384-d beatmap vector.](docs/assets/architecture.png)

### Encoder

| Setting | Default in [`configs/default.yaml`](configs/default.yaml) |
| --- | --- |
| Parameters | 16.1M |
| Depth / width / heads | 9 layers / 384 / 6 |
| Sequence length limit | 4,096 hit objects |
| Attention | Global at layers 3, 6, 9; ±128-object window otherwise |

### Pretraining

- **Masked reconstruction.** 30% of hit objects are masked in short spans, and the model predicts all of their features. We follow the original BERT's 80/10/10 masking recipe while also corrupting features on border neighbors.
- **Strain regression.** A linear head predicts difficulty strain values from the mean-pooled global layers. This objective supervises the mean-pooled embedding directly and makes sure it carries strain information.


The beatmap vector is the mean-pooled output of the three global layers. Pretraining with masked reconstruction and strain regression alone already gives strong retrieval. A linear adapter, initialized to identity and trained contrastively on maps that share collections, then refines the ranking of neighbors and makes similarity scores more meaningful.

## Limitations

- The model consumes hit objects only. Map attributes like AR, CS, OD, and HP are not read, so embeddings do not reflect them.
- Maps with few hit objects give the model fewer observations to average over, so their nearest neighbors are less reliable.
- Only the first 4,096 hit objects are encoded; anything past that is discarded.
- Within a map, 1/3 and 1/4 rhythms are distinguished, but across maps they sit in separate regions. The same stream mapped in 1/3 and in 1/4 will not be treated as similar.
- Only osu!standard is supported; other modes would need their own features and training.

## Run locally

The API runs on CPU in Docker; the frontend uses Bun. Local development connects Vite directly to the API.

### Prepare artifacts

Download a versioned model, embedding index, and metadata catalogs from [Hugging Face](https://huggingface.co/token03/bobert):

```sh
uv run --no-default-groups --group serve fetch-run --revision v14.1
```

This uses CPU dependencies, installs the run artifacts under `runs/`, and places the metadata catalogs in `data/`:

```text
data/
  beatmaps.parquet
  beatmapsets.parquet
  strains.parquet
runs/
  current -> <run>
  <run>/
    model.safetensors
    embeddings.parquet
    training.yaml
```

The catalogs are shared across runs; re-fetching a release overwrites them. Model, index, and training config stay together per run.

### Start the API and frontend

Copy [`.env.example`](.env.example) to `.env` and set `OSU_CLIENT_ID` and `OSU_CLIENT_SECRET` using an [osu! OAuth application](https://osu.ppy.sh/home/account/edit). From the repository root:

```sh
docker compose up --build api
```

The API is available at `http://127.0.0.1:8008`; interactive documentation is at `/api/docs`. See the [server guide](server/README.md) for configuration and endpoints.

From `web/`:

```sh
bun install
bun run dev
```

Open the address printed by Vite. After installing frontend dependencies, `mise run dev` starts both development services from the repository root. See [web development](web/README.md).

## Training and exploration

The Python development environment targets **Linux x86-64, Python 3.12, PyTorch 2.11, and CUDA 12.8**, with a pinned FlashAttention wheel. Install it with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --frozen
uv run pretrain --help
```

After preparing the dataset and strain targets, train with a named run:

```sh
uv run pretrain --config configs/default.yaml --full -v my-run
uv run embed -v my-run
```

Pretraining saves a run configuration and exports `model.safetensors`. `export-model` also exports existing checkpoints. `adapt` trains the embedding adapter; `evaluate`, `mine`, and `umap` support retrieval analysis and exploration. The [scripts guide](scripts/README.md) maps each command to its inputs and outputs.

`uv sync` installs the default CUDA training group. For CPU serving and release downloads, use `uv sync --no-default-groups --group serve`. Both environments are resolved in `uv.lock`.

## Repository guide

| Directory | Purpose |
| --- | --- |
| [`core/`](core/) | Beatmap parsing, features, encoder, and dataset primitives |
| [`training/`](training/) | Lightning modules, data loading, objectives, and optimizers |
| [`scripts/`](scripts/README.md) | Data acquisition, dataset construction, training entry points, and evaluation |
| [`server/`](server/README.md) | FastAPI application, retrieval runtime, and online embedding cache |
| [`web/`](web/README.md) | React application and Cloudflare gateway |
| `data/`, `runs/`, `cache/` | Local datasets, model artifacts, and runtime state |

## Acknowledgements and references

- [MusicBERT](https://arxiv.org/abs/2106.05630): a major inspiration for representation learning from structured musical sequences.
- [CM3P](https://github.com/OliBomby/CM3P) by OliBomby: inspiration for the encoder architecture.
- [ModernBERT](https://github.com/AnswerDotAI/ModernBERT): efficient bidirectional encoders with rotary embeddings and alternating attention.
- [FT-Transformer](https://github.com/yandex-research/rtdl-revisiting-models): feature-wise embeddings for numerical and categorical inputs.
- [parsecore](https://github.com/Apart-Studio/parsecore): star ratings and structural strain factors used as the auxiliary training targets.
- [osu!collector](https://osucollector.com/) and [osu!stats](https://osustats.ppy.sh/): labelled collection data used for adaptation and evaluation.
- [Beatconnect](https://beatconnect.io/): `.osu` files used to build the dataset.
- [osu!](https://osu.ppy.sh/) and its mapping community: beatmaps and metadata underlying the project.

## License

Code is available under the [MIT License](LICENSE).
