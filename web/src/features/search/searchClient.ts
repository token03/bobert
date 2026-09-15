import type { SearchSet } from './search'

export type SearchRequest = { query: string } | { beatmapId: number }
type SearchReply = { id: number; results?: SearchSet[]; setId?: number | null; error?: string }

let worker: Worker | null = null
let requestId = 0
const pending = new Map<number, { resolve: (reply: SearchReply) => void; reject: (error: Error) => void }>()

export function startSearch() {
  if (worker) return
  worker = new Worker(new URL('./search.worker.ts', import.meta.url), { type: 'module' })
  worker.onmessage = ({ data }: MessageEvent<SearchReply>) => {
    const request = pending.get(data.id)
    pending.delete(data.id)
    if (data.error) request?.reject(new Error(data.error))
    else request?.resolve(data)
  }
  worker.onerror = () => {
    for (const request of pending.values()) request.reject(new Error('Could not load search. Try a beatmap ID or link.'))
    pending.clear()
    worker?.terminate()
    worker = null
  }
}

function send(request: SearchRequest): Promise<SearchReply> {
  startSearch()
  const id = ++requestId
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    worker!.postMessage({ ...request, id })
  })
}

export async function searchBeatmaps(query: string): Promise<SearchSet[]> {
  return (await send({ query })).results!
}

export async function lookupBeatmapSet(beatmapId: number): Promise<number | null> {
  return (await send({ beatmapId })).setId ?? null
}
