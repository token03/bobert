import type { RecommendResponse } from '../../shared/types'
import { defaultFilters, normalizeBeatmapInput } from './filters'
import type { RecommendFormValues } from './filters'

const cachePrefix = 'bobert:recommend:'
const beatmapParam = 'beatmap'
const fields = [
  'topK',
  'minSr',
  'maxSr',
  'minBpm',
  'maxBpm',
  'minLength',
  'maxLength',
  'minAr',
  'maxAr',
  'minCs',
  'maxCs',
  'minOd',
  'maxOd',
  'minHp',
  'maxHp',
  'status',
] as const satisfies readonly (keyof RecommendFormValues)[]

type CachedRecommendation = {
  values: RecommendFormValues
  response: RecommendResponse
}

export function recommendSearchParams(values: RecommendFormValues) {
  const params = new URLSearchParams()
  params.set(beatmapParam, normalizeBeatmapInput(values.beatmapInput))

  for (const field of fields) {
    if (values[field] !== defaultFilters[field]) {
      params.set(field, values[field])
    }
  }

  if (values.excludeSameSet !== defaultFilters.excludeSameSet) {
    params.set('excludeSameSet', String(values.excludeSameSet))
  }

  return params.toString()
}

export function valuesFromRecommendSearch(search: string): RecommendFormValues | null {
  const params = new URLSearchParams(search)
  const beatmapInput = params.get(beatmapParam)

  if (!beatmapInput) {
    return null
  }

  const values: RecommendFormValues = {
    ...defaultFilters,
    beatmapInput: normalizeBeatmapInput(beatmapInput),
  }

  for (const field of fields) {
    values[field] = params.get(field) ?? defaultFilters[field]
  }

  values.excludeSameSet = params.get('excludeSameSet') === null ? defaultFilters.excludeSameSet : params.get('excludeSameSet') === 'true'

  return values
}

export function readCachedRecommendation(values: RecommendFormValues): CachedRecommendation | null {
  try {
    const item = sessionStorage.getItem(cacheKey(values))
    return item ? JSON.parse(item) as CachedRecommendation : null
  } catch {
    return null
  }
}

export function writeCachedRecommendation(values: RecommendFormValues, response: RecommendResponse) {
  try {
    sessionStorage.setItem(cacheKey(values), JSON.stringify({ values, response }))
  } catch {
    return
  }
}

function cacheKey(values: RecommendFormValues) {
  return `${cachePrefix}${recommendSearchParams(values)}`
}
