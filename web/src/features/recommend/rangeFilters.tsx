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
  binWidth: number
  formatValue: (value: number) => string
  formatTick: (value: number) => string
  parseValue: (text: string) => number | null
  formatAriaValue: (value: number, overflow: Overflow) => string
}

const lengthMax = 10 * 60

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
    binWidth: 1,
    formatValue: formatDecimal,
    formatTick: String,
    parseValue: parseNumber,
    formatAriaValue: ariaValue('stars', formatDecimal),
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
    ticks: [0, 2, 4, 6, 8, 10],
    binWidth: 1,
    formatValue: (value) => value.toFixed(1),
    formatTick: String,
    parseValue: parseNumber,
    formatAriaValue: ariaValue('approach rate', (value) => value.toFixed(1)),
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
    ticks: [0, 2, 4, 6, 8, 10],
    binWidth: 1,
    formatValue: (value) => value.toFixed(1),
    formatTick: String,
    parseValue: parseNumber,
    formatAriaValue: ariaValue('circle size', (value) => value.toFixed(1)),
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
    ticks: [100, 150, 200, 250, 300],
    binWidth: 20,
    formatValue: (value) => String(Math.round(value)),
    formatTick: String,
    parseValue: parseNumber,
    formatAriaValue: ariaValue('beats per minute', (value) => String(Math.round(value))),
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
    ticks: [0, 120, 300, 600],
    binWidth: 60,
    formatValue: formatDuration,
    formatTick: formatDuration,
    parseValue: parseDuration,
    formatAriaValue: formatDurationAria,
  },
]
