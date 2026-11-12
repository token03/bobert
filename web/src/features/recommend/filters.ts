import { z } from 'zod'
import { modes } from '../../shared/types'
import type { RecommendFilters, RecommendRequest } from '../../shared/types'
import { rangeFilters } from './rangeFilters'
import type { RangeParam } from './rangeFilters'

export const maxBeatmaps = 10
export const topK = 100
export const statuses = ['ranked', 'loved', 'unranked'] as const

const monthSchema = z.string().regex(/^\d{4}-(0[1-9]|1[0-2])$/).optional().catch(undefined)
const numberSchema = z.number().finite().nonnegative().optional().catch(undefined)

export const recommendSearchSchema = z.object({
  mode: z.enum(modes).catch('std'),
  beatmap: z.preprocess((value) => value === undefined || value === null ? '' : String(value), z.string()).catch(''),
  ...Object.fromEntries(rangeFilters.flatMap((filter) => filter.params.map((param) => [param, numberSchema]))) as Record<RangeParam, typeof numberSchema>,
  status: z.enum(statuses).optional().catch(undefined),
  minDate: monthSchema,
  maxDate: monthSchema,
  group: z.number().int().nonnegative().catch(0),
})

export type RecommendSearch = z.infer<typeof recommendSearchSchema>

export const defaultSearch: RecommendSearch = { beatmap: '', mode: 'std', group: 0 }

export function clearFilters(search: RecommendSearch): RecommendSearch {
  return { beatmap: search.beatmap, mode: search.mode, group: search.group }
}

export function buildRecommendRequest(search: RecommendSearch, beatmapIds: number[]): RecommendRequest {
  const filters: RecommendFilters = {
    status: search.status ?? null,
    min_date: search.minDate ?? null,
    max_date: search.maxDate ?? null,
    exclude_same_set: true,
  }
  for (const { params, api } of rangeFilters) {
    filters[api[0]] = search[params[0]] ?? null
    filters[api[1]] = search[params[1]] ?? null
  }
  return {
    mode: search.mode,
    beatmap_ids: beatmapIds,
    top_k: topK,
    filters,
    distinct: false,
  }
}
