import { z } from 'zod'

export const defaultFilters = {
  beatmapInput: '',
  topK: '100',
  minSr: '',
  maxSr: '',
  minBpm: '',
  maxBpm: '',
  minLength: '',
  maxLength: '',
  minAr: '0',
  maxAr: '10',
  minCs: '0',
  maxCs: '10',
  minOd: '0',
  maxOd: '10',
  minHp: '0',
  maxHp: '10',
  status: '',
  excludeSameSet: true,
}

export const recommendFormSchema = z.object({
  beatmapInput: z.string().refine((value) => parseBeatmapId(value) !== null, 'Enter a beatmap ID or a beatmap link ending in an ID.'),
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
  excludeSameSet: z.boolean(),
})

export type RecommendFormValues = z.infer<typeof recommendFormSchema>

export function buildRecommendRequest(values: RecommendFormValues) {
  const filters: Record<string, boolean | number | string | null> = {
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
    beatmapId: parseBeatmapId(values.beatmapInput)!,
    topK: Number(values.topK),
    filters,
  }
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
