import type { ReactNode } from 'react'
import { Clock, Metronome, Star } from '@phosphor-icons/react'
import type { CatalogStat } from '../search/search'
import type { RecommendFilters } from '../../shared/types'

export const sliderResolution = 1000

type Overflow = 'lower' | 'higher' | null
type NumericFilter = { [K in keyof RecommendFilters]-?: NonNullable<RecommendFilters[K]> extends number ? K : never }[keyof RecommendFilters]
export type RangeParam = 'minSr' | 'maxSr' | 'minAr' | 'maxAr' | 'minCs' | 'maxCs' | 'minBpm' | 'maxBpm' | 'minLength' | 'maxLength'

export type RangeFilterConfig = {
  key: CatalogStat
  params: readonly [RangeParam, RangeParam]
  api: readonly [NumericFilter, NumericFilter]
  label: string
  unit: string
  triggerLabel?: string
  defaultLabel: string
  icon?: ReactNode
  min: number
  max: number
  step: number
  largeStep: number
  precision: number
  overflowMin?: boolean
  overflowMax?: boolean
  ticks: readonly number[]
  bins: readonly number[]
  binLabels?: readonly string[]
  formatValue: (value: number) => string
  formatTick: (value: number) => string
  parseValue: (text: string) => number | null
  formatAriaValue: (value: number, overflow: Overflow) => string
  toSliderValue: (value: number) => number
  fromSliderValue: (position: number) => number
}

function piecewiseScale(knots: ReadonlyArray<readonly [number, number]>) {
  const segment = (value: number, from: 0 | 1) => {
    let index = 1
    while (index < knots.length - 1 && value > knots[index][from]) index++
    return [knots[index - 1], knots[index]] as const
  }
  return {
    toSliderValue: (value: number) => {
      const [[v0, p0], [v1, p1]] = segment(value, 0)
      const position = p0 + (p1 - p0) * (value - v0) / (v1 - v0)
      return sliderResolution * Math.min(1, Math.max(0, position))
    },
    fromSliderValue: (slider: number) => {
      const position = slider / sliderResolution
      const [[v0, p0], [v1, p1]] = segment(position, 1)
      return v0 + (v1 - v0) * (position - p0) / (p1 - p0)
    },
  }
}

const lengthMax = 20 * 60
const lengthScale = Math.log1p(lengthMax / 60)

function range(from: number, to: number, step: number) {
  return Array.from({ length: Math.round((to - from) / step) + 1 }, (_, index) => from + index * step)
}

function parseNumber(text: string) {
  const number = Number.parseFloat(text.replace(/[^\d.]/g, ''))
  return Number.isFinite(number) ? number : null
}

function parseDuration(text: string) {
  const parts = text.trim().split(':')
  if (parts.length > 3 || parts.some((part) => !/^\d+(\.\d+)?$/.test(part))) {
    return null
  }
  return parts.reduce((total, part) => total * 60 + Number(part), 0)
}

export function formatDuration(value: number) {
  const hours = Math.floor(value / 3600)
  const minutes = Math.floor(value % 3600 / 60)
  const seconds = String(Math.round(value % 60)).padStart(2, '0')
  return hours > 0 ? `${hours}:${String(minutes).padStart(2, '0')}:${seconds}` : `${minutes}:${seconds}`
}

function formatDurationAria(value: number, overflow: Overflow) {
  const minutes = Math.floor(value / 60)
  const seconds = value % 60
  const parts = [minutes ? `${minutes} minute${minutes === 1 ? '' : 's'}` : '', seconds ? `${seconds} second${seconds === 1 ? '' : 's'}` : ''].filter(Boolean)
  return `${parts.join(' ') || '0 seconds'}${overflow ? ` or ${overflow === 'higher' ? 'longer' : 'shorter'}` : ''}`
}

const formatDecimal = (value: number) => value.toFixed(Number.isInteger(Math.round(value * 100) / 10) ? 1 : 2)
const ariaValue = (unit: string, format: (value: number) => string) => (value: number, overflow: Overflow) => `${format(value)} ${unit}${overflow ? ` or ${overflow}` : ''}`

export const rangeFilters: readonly RangeFilterConfig[] = [
  {
    key: 'stars',
    params: ['minSr', 'maxSr'],
    api: ['min_sr', 'max_sr'],
    label: 'Star rating',
    unit: '★',
    defaultLabel: 'Stars',
    icon: <Star />,
    min: 0,
    max: 12,
    step: 0.1,
    largeStep: 0.5,
    precision: 0.01,
    overflowMax: true,
    ticks: [0, 3, 6, 9, 12],
    bins: range(0, 12, 1),
    formatValue: formatDecimal,
    formatTick: String,
    parseValue: parseNumber,
    formatAriaValue: ariaValue('stars', formatDecimal),
    ...piecewiseScale([[0, 0], [3, 0.16], [9, 0.88], [12, 1]]),
  },
  {
    key: 'ar',
    params: ['minAr', 'maxAr'],
    api: ['min_ar', 'max_ar'],
    label: 'Approach rate',
    unit: 'AR',
    triggerLabel: 'AR',
    defaultLabel: '—',
    min: 0,
    max: 10,
    step: 0.1,
    largeStep: 0.5,
    precision: 0.1,
    ticks: [0, 7, 8, 9, 10],
    bins: [...range(0, 7, 1), ...range(7.5, 10, 0.5)],
    formatValue: (value) => value.toFixed(1),
    formatTick: String,
    parseValue: parseNumber,
    formatAriaValue: ariaValue('approach rate', (value) => value.toFixed(1)),
    ...piecewiseScale([[0, 0], [7, 0.2], [8, 0.36], [10, 1]]),
  },
  {
    key: 'cs',
    params: ['minCs', 'maxCs'],
    api: ['min_cs', 'max_cs'],
    label: 'Circle size',
    unit: 'CS',
    triggerLabel: 'CS',
    defaultLabel: '—',
    min: 0,
    max: 10,
    step: 0.1,
    largeStep: 0.5,
    precision: 0.1,
    ticks: [0, 3, 4, 5, 10],
    bins: [0, 1, 2, ...range(2.5, 6, 0.5), 7, 8, 10],
    formatValue: (value) => value.toFixed(1),
    formatTick: String,
    parseValue: parseNumber,
    formatAriaValue: ariaValue('circle size', (value) => value.toFixed(1)),
    ...piecewiseScale([[0, 0], [2, 0.08], [3, 0.2], [5, 0.8], [7, 0.93], [10, 1]]),
  },
  {
    key: 'bpm',
    params: ['minBpm', 'maxBpm'],
    api: ['min_bpm', 'max_bpm'],
    label: 'BPM',
    unit: 'BPM',
    defaultLabel: 'BPM',
    icon: <Metronome />,
    min: 60,
    max: 320,
    step: 5,
    largeStep: 20,
    precision: 1,
    overflowMin: true,
    overflowMax: true,
    ticks: [150, 200, 250],
    bins: [60, 80, ...range(100, 260, 10), 280, 300, 320],
    formatValue: (value) => String(Math.round(value)),
    formatTick: String,
    parseValue: parseNumber,
    formatAriaValue: ariaValue('beats per minute', (value) => String(Math.round(value))),
    ...piecewiseScale([[60, 0], [100, 0.1], [260, 0.9], [320, 1]]),
  },
  {
    key: 'length',
    params: ['minLength', 'maxLength'],
    api: ['min_length', 'max_length'],
    label: 'Length',
    unit: '',
    defaultLabel: 'Length',
    icon: <Clock />,
    min: 0,
    max: lengthMax,
    step: 5,
    largeStep: 30,
    precision: 1,
    overflowMax: true,
    ticks: [60, 180, 300, 600],
    bins: [0, 60, 120, 300, lengthMax],
    binLabels: ['Ringtone', 'TV size', 'Full', 'Marathon'],
    formatValue: formatDuration,
    formatTick: formatDuration,
    parseValue: parseDuration,
    formatAriaValue: formatDurationAria,
    toSliderValue: (value) => sliderResolution * Math.log1p(value / 60) / lengthScale,
    fromSliderValue: (position) => 60 * Math.expm1(position / sliderResolution * lengthScale),
  },
]
