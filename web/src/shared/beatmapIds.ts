import type { Mode } from './types'

export function parseBeatmapIds(value: string): number[] | null {
  const parts = value.trim().split(/[\s,;]+/).filter(Boolean)
  if (!parts.length) {
    return null
  }

  const ids = parts.map(parseBeatmapId)
  if (ids.some((id) => id === null)) {
    return null
  }
  return [...new Set(ids as number[])]
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

export function modeFromUrl(value: string): Mode | null {
  const match = value.match(/osu\.ppy\.sh\/beatmapsets\/\d+#(osu|taiko|fruits|mania)\//)
  return match ? ({ osu: 'std', taiko: 'taiko', fruits: 'catch', mania: 'mania' } as const)[match[1] as 'osu' | 'taiko' | 'fruits' | 'mania'] : null
}
