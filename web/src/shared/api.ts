import createClient from 'openapi-fetch'
import type { components, paths } from './schema'
import type { RecommendRequest } from './types'

const client = createClient<paths>()
type Schemas = components['schemas']

export async function recommendBeatmaps(body: RecommendRequest, signal?: AbortSignal): Promise<Schemas['RecommendResponse']> {
  const { data, response } = await client.POST('/api/recommend', {
    body,
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
