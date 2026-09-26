import createClient from 'openapi-fetch'
import type { components, paths } from './schema'
import type { RecommendRequest } from './types'

const client = createClient<paths>()
type Schemas = components['schemas']

export class UnavailableBeatmapError extends Error {
  readonly beatmapId: number

  constructor(beatmapId: number) {
    super(`Beatmap ${beatmapId} is unavailable`)
    this.beatmapId = beatmapId
  }
}

export async function recommendBeatmaps(body: RecommendRequest, signal?: AbortSignal): Promise<Schemas['RecommendResponse']> {
  const { data, error, response } = await client.POST('/api/recommend', {
    body,
    signal,
  })
  const unavailable = String((error as { detail?: unknown } | undefined)?.detail ?? '').match(/^beatmap (\d+) is unavailable$/)
  if (response.status === 404 && unavailable) {
    throw new UnavailableBeatmapError(Number(unavailable[1]))
  }
  if (!data) {
    throw new Error(`Request failed with ${response.status}`)
  }

  return data
}

export async function fetchDefaultRecommendations(signal?: AbortSignal): Promise<Schemas['DefaultRecommendResponse']> {
  const { data, response } = await client.GET('/api/recommend', { signal })
  if (!data) {
    throw new Error(`Request failed with ${response.status}`)
  }

  return data
}
