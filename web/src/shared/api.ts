import type { BeatmapMetadata, DefaultRecommendResponse, RecommendRequest, RecommendResponse } from './types'

const apiUrl = '/api'

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const result = await fetch(`${apiUrl}${path}`, init)
  const text = await result.text()
  const data = text ? JSON.parse(text) : null

  if (!result.ok) {
    throw new Error(data?.detail ?? `Request failed with ${result.status}`)
  }

  return data as T
}

export async function recommendBeatmaps(request: RecommendRequest, signal?: AbortSignal): Promise<RecommendResponse> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (request.turnstileToken) {
    headers['X-Turnstile-Token'] = request.turnstileToken
  }

  return api('/recommend', {
    method: 'POST',
    headers,
    signal,
    body: JSON.stringify({
      beatmap_id: request.beatmapId,
      top_k: request.topK,
      filters: request.filters,
    }),
  })
}

export async function fetchDefaultRecommendations(signal?: AbortSignal): Promise<DefaultRecommendResponse> {
  return api('/recommend', { signal })
}

export async function fetchBeatmapSummary(beatmapId: number, signal?: AbortSignal): Promise<BeatmapMetadata> {
  return api(`/beatmaps/${beatmapId}/summary`, { signal })
}
