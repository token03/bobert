import createClient from 'openapi-fetch'
import type { components, paths } from './schema'
import type { Mode, RecommendRequest } from './types'

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
    body: { ...body, mode: undefined },
    signal,
  })
  const unavailable = String((error as { detail?: unknown } | undefined)?.detail ?? '').match(/^beatmap (\d+) is unavailable$/)
  if (response.status === 404 && unavailable) {
    throw new UnavailableBeatmapError(Number(unavailable[1]))
  }
  if (!data) {
    throw new Error(typeof (error as { detail?: unknown })?.detail === 'string' ? String((error as { detail?: unknown })?.detail) : `Request failed with ${response.status}`)
  }

  return data
}

export async function fetchDefaultRecommendations(mode: Mode, signal?: AbortSignal): Promise<Schemas['DefaultRecommendResponse']> {
  const { data, response } = await client.GET('/api/recommend', { signal, params: { query: { mode } } })
  if (!data) {
    throw new Error(`Request failed with ${response.status}`)
  }

  return data
}

export type PlayKind = 'best' | 'recent'

export async function fetchPlays(userId: number, kind: PlayKind, mode: Mode, signal?: AbortSignal): Promise<Schemas['PlaysResponse']> {
  const { data, error, response } = await client.GET('/api/users/{user_id}/plays', {
    signal,
    params: { path: { user_id: userId }, query: { kind, mode, limit: 100 } },
  })
  if (!data) {
    throw new Error(response.status === 404 ? `User ${userId} not found` : String((error as { detail?: unknown } | undefined)?.detail ?? `Request failed with ${response.status}`))
  }

  return data
}

export async function groupPlays(mode: Mode, plays: Schemas['GroupPlay'][], signal?: AbortSignal): Promise<Schemas['GroupResponse']> {
  const { data, response } = await client.POST('/api/beatmaps/groups', {
    body: { mode, plays },
    signal,
  })
  if (!data) {
    throw new Error(`Request failed with ${response.status}`)
  }

  return data
}
