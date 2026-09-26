import type { BeatmapMetadata } from './types'

export function displayArtist(beatmap: BeatmapMetadata): string {
  return beatmap.artist ?? 'Unknown artist'
}

export function displayTitle(beatmap: BeatmapMetadata): string {
  return beatmap.title ?? `Beatmap ${beatmap.beatmap_id}`
}

export function formatNumber(value: number | null, digits: number): string {
  if (value === null || Number.isNaN(value)) {
    return '-'
  }
  return value.toFixed(digits).replace(/\.0+$/, '')
}

export function formatFixedNumber(value: number | null, digits: number): string {
  if (value === null || Number.isNaN(value)) {
    return '-'
  }
  return value.toFixed(digits)
}

export function formatLength(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return '-'
  }

  const totalSeconds = Math.max(0, Math.round(value))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = String(totalSeconds % 60).padStart(2, '0')
  return `${minutes}:${seconds}`
}

const dateFormat = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' })

export function formatDate(value: string): string {
  const date = new Date(`${value.slice(0, 10)}T00:00:00Z`)
  return Number.isNaN(date.getTime()) ? value : dateFormat.format(date)
}

const countFormat = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1, maximumSignificantDigits: 3, roundingPriority: 'lessPrecision' })

export function formatCount(value: number): string {
  return countFormat.format(value)
}

export function formatDifficultyStat(value: number | null): string {
  return value === 10 ? '10\u2008' : formatFixedNumber(value, 1)
}

export function formatMatch(value: number): string {
  return `${(value * 100).toFixed(0)}%`
}

export function statusLabel(value: string | number | null): string {
  const statuses: Record<string, string> = {
    '-2': 'graveyard',
    '-1': 'wip',
    '0': 'pending',
    '1': 'ranked',
    '2': 'approved',
    '3': 'qualified',
    '4': 'loved',
  }

  if (!value) {
    return 'unknown status'
  }

  const status = String(value)
  return statuses[status] ?? status
}
