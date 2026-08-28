import createClient from 'openapi-fetch'
import type { components, paths } from './schema'
import type { RecommendRequest } from './types'

const client = createClient<paths>()
type Schemas = components['schemas']

export async function recommendBeatmaps(body: RecommendRequest, turnstileToken?: string, signal?: AbortSignal): Promise<Schemas['RecommendResponse']> {
  const { data, response } = await client.POST('/api/recommend', {
    body,
    headers: turnstileToken ? { 'X-Turnstile-Token': turnstileToken } : undefined,
    signal,
  })
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

export async function fetchBeatmapSummary(beatmapId: number, signal?: AbortSignal): Promise<Schemas['BeatmapSummary']> {
  const { data, response } = await client.GET('/api/beatmaps/{beatmap_id}/summary', {
    params: { path: { beatmap_id: beatmapId } },
    signal,
  })
  if (!data) {
    throw new Error(`Request failed with ${response.status}`)
  }

  return data
}
