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

export function formatMatch(value: number): string {
  return `${(value * 100).toFixed(2)}%`
}

export function resultsLabel(count: number): string {
  return `${count} RESULTS`
}

export function statusLabel(value: string | null): string {
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

  return statuses[value] ?? value
}

export function statusClass(value: string | null): string {
  const label = statusLabel(value)
  return `status-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`
}
