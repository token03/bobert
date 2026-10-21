import { beatmapTitle, catalogStats, decodeCorpus, findSetId, searchCorpus } from './search'
import type { SearchCorpus } from './search'
import type { SearchRequest } from './searchClient'

const catalogs = new Map<string, Promise<SearchCorpus>>()

function loadCorpus(mode: string): Promise<SearchCorpus> {
  let catalog = catalogs.get(mode)
  if (!catalog) {
    catalog = fetch(`/search-${mode}.bin`).then(async (response) => {
      if (!response.ok) throw new Error('Missing search artifact')
      return decodeCorpus(await response.arrayBuffer())
    })
    catalogs.set(mode, catalog)
    void catalog.catch(() => catalogs.delete(mode))
  }
  return catalog
}

function handle(corpus: SearchCorpus, request: SearchRequest) {
  switch (request.type) {
    case 'search': return searchCorpus(corpus, request.query)
    case 'set': return findSetId(corpus, request.beatmapId)
    case 'title': return beatmapTitle(corpus, request.beatmapId)
    case 'stats': return catalogStats(corpus)
  }
}

self.onmessage = async ({ data }: MessageEvent<SearchRequest & { id: number }>) => {
  try {
    self.postMessage({ id: data.id, result: handle(await loadCorpus(data.mode), data) })
  } catch (error) {
    console.error(error)
    self.postMessage({ id: data.id, error: 'Search is unavailable. Try a beatmap ID or link.' })
  }
}
