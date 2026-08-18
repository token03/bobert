import type { BeatmapMetadata } from './types'

export function beatmapUrl(beatmap: BeatmapMetadata): string {
  return beatmap.url ?? `https://osu.ppy.sh/beatmaps/${beatmap.beatmap_id}`
}

export function previewUrl(beatmapsetId: number): string {
  return `https://b.ppy.sh/preview/${beatmapsetId}.mp3`
}

export function coverUrl(beatmapsetId: number, twoX = false): string {
  return `https://assets.ppy.sh/beatmaps/${beatmapsetId}/covers/list${twoX ? '@2x' : ''}.jpg`
}

export function cardCoverUrl(beatmapsetId: number): string {
  return `https://assets.ppy.sh/beatmaps/${beatmapsetId}/covers/card.jpg`
}

export function userUrl(userId: number): string {
  return `https://osu.ppy.sh/users/${userId}`
}
