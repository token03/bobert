import type { DefaultRecommendResponse, RecommendRequest, RecommendResponse } from './types'

const apiUrl = import.meta.env.VITE_API_URL ?? '/api'

export async function recommendBeatmaps(request: RecommendRequest): Promise<RecommendResponse> {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  }

  if (request.turnstileToken) {
    headers['X-Turnstile-Token'] = request.turnstileToken
  }

  const result = await fetch(`${apiUrl}/recommend`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      beatmap_id: request.beatmapId,
      top_k: request.topK,
      filters: request.filters,
    }),
  })
  const text = await result.text()
  const data = text ? JSON.parse(text) : null

  if (!result.ok) {
    throw new Error(data?.detail ?? `Request failed with ${result.status}`)
  }

  return data as RecommendResponse
}

export async function fetchDefaultRecommendations(): Promise<DefaultRecommendResponse> {
  const result = await fetch(`${apiUrl}/recommend`)
  const text = await result.text()
  const data = text ? JSON.parse(text) : null

  if (!result.ok) {
    throw new Error(data?.detail ?? `Request failed with ${result.status}`)
  }

  return data as DefaultRecommendResponse
}
