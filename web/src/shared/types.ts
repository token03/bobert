export type BeatmapMetadata = {
  beatmap_id: number
  beatmapset_id: number | null
  artist: string | null
  title: string | null
  creator: string | null
  creator_id?: number | null
  user_id?: number | null
  version: string | null
  status: string | null
  stars: number | null
  ar: number | null
  cs: number | null
  accuracy: number | null
  drain: number | null
  bpm: number | null
  total_length: number | null
  url: string | null
  score?: number
}

export type RecommendResponse = {
  query: {
    beatmap_id: number
    cache: string
    metadata: BeatmapMetadata
  }
  count: number
  results: BeatmapMetadata[]
}

export type DefaultRecommendResponse = {
  count: number
  results: BeatmapMetadata[]
}

export type RecommendRequest = {
  beatmapId: number
  topK: number
  filters: Record<string, boolean | number | string | null>
  turnstileToken?: string
}
