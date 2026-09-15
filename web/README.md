# BoBERT web

React and TypeScript frontend for finding similar osu! beatmaps, filtering recommendations, and previewing audio. Built with Vite, TanStack Router, TanStack Query, and TanStack Form.

## Development

Start the [API](../server/README.md) on `127.0.0.1:8008`, then run from this directory:

```sh
bun install
bun run dev
```

Open the address printed by Vite. [`vite.config.ts`](vite.config.ts) proxies `/api` to the local backend. From the repository root, `mise run dev` starts both services after dependencies and model artifacts are prepared.

`bun run dev` generates the Git-ignored `public/search.bin` using Python/uv, `../data/beatmaps.parquet`, and `../runs/current/embeddings.parquet`. Unchanged inputs skip regeneration; `bun run generate:search` forces a rebuild. Production builds download the checksum-verified GitHub release pinned in [`search-catalog.json`](search-catalog.json).

## Commands

| Command | Purpose |
| --- | --- |
| `bun run dev` | Development server with API proxy |
| `bun run build` | Fetch the pinned search catalog, type-check, and build into `dist/` |
| `bun run lint` | ESLint |
| `bun run preview` | Preview the static build; an API route must be provided separately |
| `bun run generate:api` | Regenerate `src/shared/schema.d.ts` from the running local API |

## Source guide

- `src/features/recommend/`: form, filters, results, and beatmap cards.
- `src/features/search/`: local beatmap search engine, worker, and combobox picker.
- `src/features/audio/`: shared audio-preview state and controls.
- `src/routes/`: TanStack route definitions; `routeTree.gen.ts` is generated.
- `src/shared/`: typed API client, generated schema, formatting, beatmap ID parsing, and UI utilities.
- `src/styles/`: global styles and theme.
- `functions/`: Cloudflare Pages API forwarding function.
- `gateway/`: separately deployed Cloudflare Worker.

## Cloudflare deployment

Build the frontend with `bun run build` and publish `dist/` through Cloudflare Pages with the `functions/` directory. The Pages Function forwards API requests through a service binding named `GATEWAY`.

The search catalog and its Brotli headers ship in `dist/`; Git-triggered builds need no Parquet inputs. Catalog updates are independent of model releases: upload a new `search-catalog-…` release asset and update its URL and SHA-256 in `search-catalog.json`.

The gateway uses an `API` VPC service binding and two rate-limit bindings, `RECOMMEND_BURST` and `RECOMMEND_SUSTAINED`. Copy [`gateway/wrangler.example.jsonc`](gateway/wrangler.example.jsonc) to `gateway/wrangler.jsonc` and fill in your provisioned resource IDs. The actual deployment configuration is local and Git-ignored. The VPC service connects to the Python API through Cloudflare Tunnel.

From `gateway/`, install dependencies with `bun install` and deploy with `bun run deploy`. The Pages service binding and backend connectivity must also be configured in Cloudflare. Local frontend development uses the Vite proxy and does not require these bindings.
