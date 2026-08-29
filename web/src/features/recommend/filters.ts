import { z } from 'zod'
import type { RecommendFilters, RecommendRequest } from '../../shared/types'

export const defaultFilters = {
  beatmap: '',
  topK: '100',
  minSr: '',
  maxSr: '',
  minBpm: '',
  maxBpm: '',
  minLength: '',
  maxLength: '',
  minAr: '',
  maxAr: '',
  minCs: '',
  maxCs: '',
  minOd: '0',
  maxOd: '10',
  minHp: '0',
  maxHp: '10',
  status: '',
  dateWindow: '' as const,
  excludeSameSet: true,
}

const dateWindowSchema = z.union([
  z.literal(''),
  z.enum(['last_week', 'last_month', 'last_3_months', 'last_6_months', 'last_year', 'last_2_years', 'last_5_years', 'all_time']),
])

export const recommendFormSchema = z.object({
  beatmap: z.string().refine((value) => parseBeatmapId(value) !== null, 'Enter a beatmap ID or a beatmap link ending in an ID.'),
  topK: z.string().refine((value) => {
    const number = Number(value)
    return Number.isSafeInteger(number) && number > 0
  }, 'Rows must be a positive whole number.'),
  minSr: z.string(),
  maxSr: z.string(),
  minBpm: z.string(),
  maxBpm: z.string(),
  minLength: z.string(),
  maxLength: z.string(),
  minAr: z.string(),
  maxAr: z.string(),
  minCs: z.string(),
  maxCs: z.string(),
  minOd: z.string(),
  maxOd: z.string(),
  minHp: z.string(),
  maxHp: z.string(),
  status: z.string(),
  dateWindow: dateWindowSchema,
  excludeSameSet: z.boolean(),
})

export type RecommendFormValues = z.infer<typeof recommendFormSchema>

const searchString = (fallback: string) => z.preprocess(
  (value) => value === undefined || value === null ? fallback : String(value),
  z.string(),
).catch(fallback)

export const recommendSearchSchema = z.object({
  beatmap: searchString(defaultFilters.beatmap),
  topK: searchString(defaultFilters.topK),
  minSr: searchString(defaultFilters.minSr),
  maxSr: searchString(defaultFilters.maxSr),
  minBpm: searchString(defaultFilters.minBpm),
  maxBpm: searchString(defaultFilters.maxBpm),
  minLength: searchString(defaultFilters.minLength),
  maxLength: searchString(defaultFilters.maxLength),
  minAr: searchString(defaultFilters.minAr),
  maxAr: searchString(defaultFilters.maxAr),
  minCs: searchString(defaultFilters.minCs),
  maxCs: searchString(defaultFilters.maxCs),
  minOd: searchString(defaultFilters.minOd),
  maxOd: searchString(defaultFilters.maxOd),
  minHp: searchString(defaultFilters.minHp),
  maxHp: searchString(defaultFilters.maxHp),
  status: z.preprocess((value) => {
    const status = value === undefined || value === null ? defaultFilters.status : String(value)
    return {
      '1': 'ranked',
      '2': 'ranked',
      '3': 'ranked',
      '4': 'loved',
      '-2': 'unranked',
      '-1': 'unranked',
      '0': 'unranked',
    }[status] ?? status
  }, z.string()).catch(defaultFilters.status),
  dateWindow: z.preprocess(
    (value) => value === undefined || value === null ? defaultFilters.dateWindow : String(value),
    dateWindowSchema,
  ).catch(defaultFilters.dateWindow),
  excludeSameSet: z.preprocess(
    (value) => value === undefined || value === null ? defaultFilters.excludeSameSet : value === true || value === 'true',
    z.boolean(),
  ).catch(defaultFilters.excludeSameSet),
})

export function buildRecommendRequest(values: RecommendFormValues) {
  const filters: RecommendFilters = {
    min_sr: numericOrNull(values.minSr),
    max_sr: numericOrNull(values.maxSr),
    min_ar: numericOrNull(values.minAr),
    max_ar: numericOrNull(values.maxAr),
    min_cs: numericOrNull(values.minCs),
    max_cs: numericOrNull(values.maxCs),
    min_accuracy: numericOrNull(values.minOd),
    max_accuracy: numericOrNull(values.maxOd),
    min_drain: numericOrNull(values.minHp),
    max_drain: numericOrNull(values.maxHp),
    status: values.status || null,
    date_window: values.dateWindow || null,
    exclude_same_set: values.excludeSameSet,
  }
  const minBpmValue = numericOrNull(values.minBpm)
  const maxBpmValue = numericOrNull(values.maxBpm)
  const minLengthValue = numericOrNull(values.minLength)
  const maxLengthValue = numericOrNull(values.maxLength)

  if (minBpmValue !== null) {
    filters.min_bpm = minBpmValue
  }
  if (maxBpmValue !== null) {
    filters.max_bpm = maxBpmValue
  }
  if (minLengthValue !== null) {
    filters.min_length = minLengthValue
  }
  if (maxLengthValue !== null) {
    filters.max_length = maxLengthValue
  }

  return {
    beatmap_id: parseBeatmapId(values.beatmap)!,
    top_k: Number(values.topK),
    filters,
  } satisfies RecommendRequest
}

export function normalizeBeatmapInput(value: string): string {
  return String(parseBeatmapId(value) ?? value.trim())
}

export function parseBeatmapId(value: string): number | null {
  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }

  const urlMatch = trimmed.match(/^https?:\/\/[^/?#]+([^?#]*)(?:\?[^#]*)?(#.*)?$/i)
  const searchable = urlMatch ? `${urlMatch[1]}${urlMatch[2] ?? ''}` : trimmed

  const matches = [...searchable.matchAll(/(?:^|[/#])(\d+)(?=$|[/#])/g)]
  if (matches.length === 0) {
    return null
  }

  const id = Number(matches[matches.length - 1][1])
  return Number.isSafeInteger(id) && id > 0 ? id : null
}

export function numericOrNull(value: string): number | null {
  if (value.trim() === '') {
    return null
  }

  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

export function sliderValue(value: string, fallback: number): number {
  const number = Number(value)
  return Number.isFinite(number) ? number : fallback
}

export function clampSliderValue(value: string, min: number, max: number): string {
  const number = Number(value)
  return Math.min(max, Math.max(min, number)).toFixed(1)
}
