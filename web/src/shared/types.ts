import type { components } from './schema'

export type BeatmapMetadata = components['schemas']['BeatmapSummary'] | components['schemas']['ScoredBeatmapSummary']

export type RecommendRequest = components['schemas']['RecommendRequest']
export type RecommendFilters = components['schemas']['RecommendFilters']
