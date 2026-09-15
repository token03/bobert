import { decodeCorpus, findSetId, searchCorpus } from './search'
import type { SearchRequest } from './searchClient'

const ready = fetch('/search.bin').then(async (response) => {
  if (!response.ok) throw new Error('Missing search artifact')
  return decodeCorpus(await response.arrayBuffer())
})

ready.catch(() => {})

self.onmessage = async ({ data }: MessageEvent<SearchRequest & { id: number }>) => {
  try {
    const corpus = await ready
    self.postMessage('query' in data
      ? { id: data.id, results: searchCorpus(corpus, data.query) }
      : { id: data.id, setId: findSetId(corpus, data.beatmapId) })
  } catch (error) {
    console.error(error)
    self.postMessage({ id: data.id, error: 'Search is unavailable. Try a beatmap ID or link.' })
  }
}
