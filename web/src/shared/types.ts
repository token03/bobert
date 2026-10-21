import type { components } from './schema'

export type BeatmapMetadata = components['schemas']['BeatmapSummary'] | components['schemas']['ScoredBeatmapSummary']

export type RecommendRequest = components['schemas']['RecommendRequest']
export type RecommendFilters = components['schemas']['RecommendFilters']

export const modes = ['std', 'taiko', 'catch', 'mania'] as const
export type Mode = typeof modes[number]
export const modeLabels: Record<Mode, string> = { std: 'osu!', taiko: 'taiko', catch: 'catch', mania: 'mania' }
