import { z } from 'zod'
import { parseBeatmapIds } from '../../shared/beatmapIds'
import type { RecommendFilters, RecommendRequest } from '../../shared/types'
import { rangeFilters } from './rangeFilters'
import type { RangeParam } from './rangeFilters'

export const maxBeatmaps = 10
export const topK = 100
export const statuses = ['ranked', 'loved', 'unranked'] as const

const monthSchema = z.string().regex(/^\d{4}-(0[1-9]|1[0-2])$/).optional().catch(undefined)
const numberSchema = z.number().finite().nonnegative().optional().catch(undefined)

export const recommendSearchSchema = z.object({
  beatmap: z.preprocess((value) => value === undefined || value === null ? '' : String(value), z.string()).catch(''),
  ...Object.fromEntries(rangeFilters.flatMap((filter) => filter.params.map((param) => [param, numberSchema]))) as Record<RangeParam, typeof numberSchema>,
  status: z.enum(statuses).optional().catch(undefined),
  minDate: monthSchema,
  maxDate: monthSchema,
})

export type RecommendSearch = z.infer<typeof recommendSearchSchema>

export const defaultSearch: RecommendSearch = { beatmap: '' }

export function clearFilters(search: RecommendSearch): RecommendSearch {
  return { beatmap: search.beatmap }
}

export function buildRecommendRequest(search: RecommendSearch): RecommendRequest {
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
    beatmap_ids: parseBeatmapIds(search.beatmap)!,
    top_k: topK,
    filters,
  }
}
