# BoBERT web

React frontend built with Vite, TanStack Router and TanStack Query.

```sh
bun install
bun run dev
```

Vite proxies `/api` to the local API on port 8008.

| Command | Purpose |
| --- | --- |
| `bun run build` | Type-check and build into `dist/` |
| `bun run lint` | ESLint |
| `bun run generate:api` | Regenerate `src/shared/schema.d.ts` from the local API |

## Source

| Path | Contents |
| --- | --- |
| `src/features/recommend/` | Page, filters, results and play-group tabs |
| `src/features/search/` | Beatmap search worker and picker |
| `src/features/audio/` | Audio previews |
| `src/shared/` | API client, generated schema and helpers |
| `functions/` | Pages Function that forwards `/api` to the gateway |
| `gateway/` | Cloudflare Worker: per-IP rate limits, blocks bot-only routes, logs requests |

## Deploy

Copy `gateway/wrangler.example.jsonc` to `gateway/wrangler.jsonc` and fill in the resource IDs. Deploy the API first, then the gateway (`bun run deploy` in `gateway/`), then `dist/` to Pages.
