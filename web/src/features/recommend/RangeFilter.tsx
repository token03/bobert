import { useState } from 'react'
import type { KeyboardEvent, MouseEvent } from 'react'
import { Slider } from '@base-ui/react/slider'
import { useQuery } from '@tanstack/react-query'
import { fetchCatalogStats } from '../search/searchClient'
import type { CatalogHistogram } from '../search/search'
import { FilterPopover } from './FilterPopover'
import { sliderResolution } from './rangeFilters'
import type { RangeFilterConfig } from './rangeFilters'
import styles from './RecommendForm.module.css'

type Range = readonly [number, number]

type RangeFilterProps = {
  config: RangeFilterConfig
  min: number | undefined
  max: number | undefined
  onValueCommit: (min: number | undefined, max: number | undefined) => void
}

const stepKeys: Record<string, number> = { ArrowRight: 1, ArrowUp: 1, PageUp: 1, ArrowLeft: -1, ArrowDown: -1, PageDown: -1 }

function roundTo(value: number, precision: number) {
  const decimals = Math.max(0, -Math.floor(Math.log10(precision)))
  return Number((Math.round(value / precision) * precision).toFixed(decimals))
}

function clamp(value: number, config: RangeFilterConfig) {
  return Math.min(config.max, Math.max(config.min, value))
}

function isFullRange([min, max]: Range, config: RangeFilterConfig) {
  return min <= config.min && max >= config.max
}

function formatRange(value: Range, config: RangeFilterConfig) {
  const [min, max] = value
  if (min === max) {
    return config.formatValue(min)
  }
  if (min <= config.min) {
    return `≤${config.formatValue(max)}`
  }
  if (max >= config.max) {
    return `≥${config.formatValue(min)}`
  }
  return `${config.formatValue(min)}–${config.formatValue(max)}`
}

function sliderPercent(value: number, config: RangeFilterConfig) {
  return `${config.toSliderValue(value) / sliderResolution * 100}%`
}

function binCounts(config: RangeFilterConfig, histogram: CatalogHistogram) {
  const counts = new Array<number>(config.bins.length - 1).fill(0)
  let bin = 0
  histogram.counts.forEach((count, bucket) => {
    const value = bucket * histogram.resolution + 1e-9
    while (bin < counts.length - 1 && value >= config.bins[bin + 1]) bin++
    counts[bin] += count
  })
  return counts
}

export function RangeFilter({ config, min, max, onValueCommit }: RangeFilterProps) {
  const [open, setOpen] = useState(false)
  const externalKey = `${min}:${max}`
  const [local, setLocal] = useState<{ value: Range; key: string } | null>(null)
  const value: Range = local?.key === externalKey ? local.value : [min ?? config.min, max ?? config.max]
  const active = !isFullRange(value, config)
  const stats = useQuery({ queryKey: ['catalog-stats'], queryFn: fetchCatalogStats, staleTime: Infinity, enabled: open })

  function preview(next: Range) {
    setLocal({ value: next, key: externalKey })
  }

  function commit(next: Range) {
    const nextMin = next[0] <= config.min ? undefined : next[0]
    const nextMax = next[1] >= config.max ? undefined : next[1]
    setLocal({ value: next, key: `${nextMin}:${nextMax}` })
    onValueCommit(nextMin, nextMax)
  }

  function snap(position: number) {
    return clamp(roundTo(config.fromSliderValue(position), config.step), config)
  }

  function stepThumb(event: KeyboardEvent<HTMLInputElement>, index: 0 | 1) {
    const direction = stepKeys[event.key]
    if (!direction) {
      return
    }
    event.preventDefault()
    const size = event.shiftKey || event.key.startsWith('Page') ? config.largeStep : config.step
    const moved = clamp(roundTo(value[index] + direction * size, config.precision), config)
    commit(index === 0 ? [Math.min(moved, value[1]), value[1]] : [value[0], Math.max(moved, value[0])])
  }

  function commitBound(index: 0 | 1, next: number) {
    const bounded = clamp(roundTo(next, config.precision), config)
    commit(index === 0 ? [bounded, Math.max(bounded, value[1])] : [Math.min(bounded, value[0]), bounded])
  }

  function thumbAria(index: 0 | 1) {
    return (_: string, position: number) => {
      const actual = snap(position)
      const overflow = index === 0 ? config.overflowMin && actual <= config.min && 'lower' : config.overflowMax && actual >= config.max && 'higher'
      return config.formatAriaValue(actual, overflow || null)
    }
  }

  return (
    <FilterPopover
      filterKey={config.key}
      label={config.label}
      mark={config.icon ?? config.triggerLabel}
      display={active ? formatRange(value, config) : config.defaultLabel}
      active={active}
      open={open}
      onOpenChange={setOpen}
      onClear={() => commit([config.min, config.max])}
      popupClassName={styles['range-popover-slider']}
    >
      <div className={styles['range-inputs']}>
        <RangeInput config={config} index={0} value={value[0]} onCommit={(next) => commitBound(0, next)} />
        <span className={styles['range-inputs-dash']} aria-hidden="true">–</span>
        <RangeInput config={config} index={1} value={value[1]} onCommit={(next) => commitBound(1, next)} />
      </div>

      <div className={styles['range-scale']}>
        <Histogram config={config} histogram={stats.data?.[config.key]} value={value} active={active} onSelect={commit} />

        <Slider.Root
          className={styles['range-slider']}
          data-active={active || undefined}
          value={value.map(config.toSliderValue)}
          min={0}
          max={sliderResolution}
          step={1}
          thumbCollisionBehavior="none"
          thumbAlignment="center"
          onValueChange={(positions) => preview([snap(positions[0]), snap(positions[1])])}
          onValueCommitted={(positions) => commit([snap(positions[0]), snap(positions[1])])}
        >
          <Slider.Control className={styles['range-slider-control']}>
            <Slider.Track className={styles['range-slider-track']}>
              <Slider.Indicator className={styles['range-slider-indicator']} />
              {([0, 1] as const).map((index) => (
                <Slider.Thumb
                  key={index}
                  className={styles['range-slider-thumb']}
                  index={index}
                  getAriaLabel={() => `${config.label} ${index === 0 ? 'minimum' : 'maximum'}`}
                  getAriaValueText={thumbAria(index)}
                  onKeyDown={(event) => stepThumb(event, index)}
                />
              ))}
            </Slider.Track>
          </Slider.Control>
        </Slider.Root>

        <div className={styles['range-ticks']} aria-hidden="true">
          {config.ticks.map((tick) => (
            <span key={tick} style={{ left: sliderPercent(tick, config) }}>{config.formatTick(tick)}</span>
          ))}
        </div>
      </div>
    </FilterPopover>
  )
}

type RangeInputProps = {
  config: RangeFilterConfig
  index: 0 | 1
  value: number
  onCommit: (value: number) => void
}

function RangeInput({ config, index, value, onCommit }: RangeInputProps) {
  const [draft, setDraft] = useState<string | null>(null)
  const bound = index === 0 ? config.min : config.max
  const atBound = index === 0 ? value <= bound : value >= bound
  const placeholder = `${config.formatValue(bound)}${index === 1 && config.overflowMax ? '+' : ''}`

  function commitDraft() {
    if (draft === null) {
      return
    }
    const parsed = draft.trim() ? config.parseValue(draft) : bound
    setDraft(null)
    if (parsed !== null) {
      onCommit(parsed)
    }
  }

  return (
    <input
      className={styles['range-input']}
      type="text"
      inputMode={config.key === 'length' ? 'text' : 'decimal'}
      autoComplete="off"
      spellCheck={false}
      aria-label={`${config.label} ${index === 0 ? 'minimum' : 'maximum'}`}
      placeholder={placeholder}
      value={draft ?? (atBound ? '' : config.formatValue(value))}
      onFocus={(event) => event.currentTarget.select()}
      onChange={(event) => setDraft(event.currentTarget.value)}
      onBlur={commitDraft}
      onKeyDown={(event) => {
        if (event.key === 'Enter') {
          event.preventDefault()
          commitDraft()
        } else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
          event.preventDefault()
          setDraft(null)
          onCommit(value + (event.key === 'ArrowUp' ? 1 : -1) * (event.shiftKey ? config.largeStep : config.step))
        }
      }}
    />
  )
}

type HistogramProps = {
  config: RangeFilterConfig
  histogram: CatalogHistogram | undefined
  value: Range
  active: boolean
  onSelect: (value: Range) => void
}

function Histogram({ config, histogram, value, active, onSelect }: HistogramProps) {
  const counts = histogram ? binCounts(config, histogram) : null
  const total = counts ? counts.reduce((sum, count) => sum + count, 0) : 0
  const peak = counts ? Math.max(1, ...counts) : 1
  const last = config.bins.length - 2
  const [min, max] = value

  function select(event: MouseEvent<HTMLButtonElement>, low: number, high: number) {
    if (event.shiftKey && active) {
      onSelect([Math.min(min, low), Math.max(max, high)])
    } else if (min === low && max === high) {
      onSelect([config.min, config.max])
    } else {
      onSelect([low, high])
    }
  }

  return (
    <div className={styles['range-histogram']} data-ready={counts !== null || undefined}>
      {config.bins.slice(0, -1).map((low, index) => {
        const high = index === last ? config.max : config.bins[index + 1]
        const inRange = active && (min === max ? low <= min && (min < high || index === last) : low < max && high > min)
        const count = counts?.[index] ?? 0
        const range = `${config.formatTick(low)}–${config.formatTick(high)}${index === last && config.overflowMax ? '+' : ''}`
        const name = config.binLabels ? `${config.binLabels[index]} (${range})` : `${range}${config.unit ? ` ${config.unit}` : ''}`
        const share = total ? `${(count / total * 100).toFixed(count / total < 0.01 ? 1 : 0)}%` : ''
        return (
          <button
            key={low}
            className={styles['range-bin']}
            type="button"
            tabIndex={-1}
            data-in-range={inRange || undefined}
            style={{ left: sliderPercent(low, config), right: `calc(100% - ${sliderPercent(high, config)})` }}
            title={share ? `${name} · ${share} of maps` : name}
            aria-label={`Select ${name}`}
            onClick={(event) => select(event, low, high)}
          >
            <span style={{ height: `${Math.sqrt(count / peak) * 100}%` }} />
          </button>
        )
      })}
    </div>
  )
}
