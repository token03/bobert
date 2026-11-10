import { z } from 'zod'
import type { components } from '../../shared/schema'
import type { BeatmapMetadata } from '../../shared/types'
import { defaultSearch, recommendSearchSchema } from './filters'

export type Play = components['schemas']['Play']

export const playsSearchSchema = recommendSearchSchema.extend({
  group: z.number().int().nonnegative().catch(0),
})

export type PlaysSearch = z.infer<typeof playsSearchSchema>

export const defaultPlaysSearch: PlaysSearch = { ...defaultSearch, group: 0 }

export const playsParams = {
  parse: ({ userId }: { userId: string }) => ({ userId: z.coerce.number().int().positive().parse(userId) }),
  stringify: ({ userId }: { userId: number }) => ({ userId: String(userId) }),
}

const modOrder = ['EZ', 'NF', 'HT', 'HD', 'HR', 'DT', 'NC', 'FL', 'SD', 'PF']

export function modCombo(mods: string[]): string {
  const sorted = [...mods].sort((a, b) => modOrder.indexOf(a) - modOrder.indexOf(b))
  return sorted.length ? sorted.join('') : 'NM'
}

function median(values: (number | null)[]): number | null {
  const present = values.filter((value): value is number => value !== null).sort((a, b) => a - b)
  return present.length ? present[Math.floor(present.length / 2)] : null
}

export function groupSummary(beatmaps: BeatmapMetadata[]) {
  return {
    stars: median(beatmaps.map((beatmap) => beatmap.stars)),
    bpm: median(beatmaps.map((beatmap) => beatmap.bpm)),
  }
}

export function readExcludedPlays(userId: number): Set<number> {
  return new Set(JSON.parse(localStorage.getItem(excludedKey(userId)) ?? '[]') as number[])
}

export function writeExcludedPlays(userId: number, excluded: Set<number>) {
  localStorage.setItem(excludedKey(userId), JSON.stringify([...excluded]))
}

function excludedKey(userId: number): string {
  return `bobert:excluded-plays:${userId}`
}
