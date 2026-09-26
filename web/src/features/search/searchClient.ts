import type { CatalogStats, SearchSet } from './search'

export type SearchRequest =
  | { type: 'search'; query: string }
  | { type: 'set'; beatmapId: number }
  | { type: 'title'; beatmapId: number }
  | { type: 'stats' }
type SearchReply = { id: number; result?: unknown; error?: string }

let worker: Worker | null = null
let requestId = 0
const pending = new Map<number, { resolve: (result: unknown) => void; reject: (error: Error) => void }>()

export function startSearch() {
  if (worker) return
  worker = new Worker(new URL('./search.worker.ts', import.meta.url), { type: 'module' })
  worker.onmessage = ({ data }: MessageEvent<SearchReply>) => {
    const request = pending.get(data.id)
    pending.delete(data.id)
    if (data.error) request?.reject(new Error(data.error))
    else request?.resolve(data.result)
  }
  worker.onerror = () => {
    for (const request of pending.values()) request.reject(new Error('Could not load search. Try a beatmap ID or link.'))
    pending.clear()
    worker?.terminate()
    worker = null
  }
}

function send<T>(request: SearchRequest): Promise<T> {
  startSearch()
  const id = ++requestId
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve: resolve as (result: unknown) => void, reject })
    worker!.postMessage({ ...request, id })
  })
}

export function searchBeatmaps(query: string): Promise<SearchSet[]> {
  return send({ type: 'search', query })
}

export function lookupBeatmapSet(beatmapId: number): Promise<number | null> {
  return send({ type: 'set', beatmapId })
}

export function lookupBeatmapTitle(beatmapId: number): Promise<string | null> {
  return send({ type: 'title', beatmapId })
}

export function fetchCatalogStats(): Promise<CatalogStats> {
  return send({ type: 'stats' })
}
