# BoBERT web

React and TypeScript frontend for finding similar osu! beatmaps, filtering recommendations, and previewing audio. Built with Vite, TanStack Router, and TanStack Query.

## Development

Start the [API](../server/README.md) on `127.0.0.1:8008`, then run from this directory:

```sh
bun install
bun run dev
```

Open the address printed by Vite. [`vite.config.ts`](vite.config.ts) proxies `/api` to the local backend. From the repository root, `mise run dev` starts both services after dependencies and model artifacts are prepared.

`bun run dev` builds the ignored `public/search-{std,taiko,catch,mania}.bin` catalogs from `data/beatmaps.parquet` and the selected indices in `runs/current/release.json`. Each mode has a separate worker-loaded catalog. Production builds use the checksum-verified GitHub release assets pinned in `search-catalog.json`. Upload the binaries to the `search-catalog-<version>` GitHub release before publishing the frontend. Mode selection is stored in the page URL, and pasted osu! mode links select their ruleset automatically.

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

The per-mode search catalogs and their Brotli headers ship in `dist/`; Git-triggered builds need no Parquet inputs. Build the catalogs and upload them to GitHub Releases before triggering a production build. Each URL and SHA-256 in `search-catalog.json` must match a published catalog; a local cached file can let a build succeed before the release is available remotely. Models, embedding indices, and metadata catalogs are published separately to Hugging Face with `publish-run`.

The gateway uses an `API` VPC service binding and two rate-limit bindings, `RECOMMEND_BURST` and `RECOMMEND_SUSTAINED`. Copy [`gateway/wrangler.example.jsonc`](gateway/wrangler.example.jsonc) to `gateway/wrangler.jsonc` and fill in your provisioned resource IDs. The actual deployment configuration is local and Git-ignored. The VPC service connects to the Python API through Cloudflare Tunnel.

From `gateway/`, install dependencies with `bun install` and deploy with `bun run deploy`. The Pages service binding and backend connectivity must also be configured in Cloudflare. Local frontend development uses the Vite proxy and does not require these bindings.

For v15, deploy the multimode API before the frontend. Pages should build from `web/` with `bun run build`, publish `dist/`, and retain its `GATEWAY` binding to `bobert-gateway`. The existing gateway forwards mode parameters and `/api/modes` without new bindings or routes.
