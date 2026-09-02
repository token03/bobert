import { useState } from 'react'
import type { ReactNode } from 'react'
import { Popover } from '@base-ui/react/popover'
import { Slider } from '@base-ui/react/slider'
import { ChevronDown, X } from 'lucide-react'
import styles from './RecommendForm.module.css'

export type RangeFilterConfig = {
  key: string
  label: string
  triggerLabel?: string
  defaultLabel?: string
  icon?: ReactNode
  min: number
  max: number
  step: number
  largeStep: number
  sliderMin?: number
  sliderMax?: number
  sliderStep?: number
  sliderLargeStep?: number
  overflowMin?: boolean
  overflowMax?: boolean
  formatValue: (value: number) => string
  formatAriaValue: (value: number, overflow: 'lower' | 'higher' | null) => string
  toSliderValue?: (value: number) => number
  fromSliderValue?: (value: number) => number
}

type RangeFilterProps = {
  config: RangeFilterConfig
  minValue: string
  maxValue: string
  onValueCommit: (min: string, max: string) => void
}

function parseValue(value: string, fallback: number) {
  const number = Number(value)
  return value.trim() !== '' && Number.isFinite(number) ? number : fallback
}

function clampValue(value: number, config: RangeFilterConfig) {
  return Math.min(config.max, Math.max(config.min, value))
}

function toSliderValue(value: number, config: RangeFilterConfig) {
  return config.toSliderValue?.(value) ?? value
}

function fromSliderValue(value: number, config: RangeFilterConfig) {
  return config.fromSliderValue?.(value) ?? value
}

function isFullRange(value: readonly number[], config: RangeFilterConfig) {
  return value[0] === config.min && value[1] === config.max
}

function formatRange(value: readonly number[], config: RangeFilterConfig, emptyLabel = 'Any') {
  const [min, max] = value
  if (isFullRange(value, config)) {
    return emptyLabel
  }
  if (min === max) {
    return config.formatValue(min)
  }
  if (min === config.min) {
    return `≤${config.formatValue(max)}`
  }
  if (max === config.max) {
    return `≥${config.formatValue(min)}`
  }
  return `${config.formatValue(min)}–${config.formatValue(max)}`
}

export function RangeFilter({ config, minValue, maxValue, onValueCommit }: RangeFilterProps) {
  const externalMin = parseValue(minValue, config.min)
  const externalMax = parseValue(maxValue, config.max)
  const clampedMin = clampValue(externalMin, config)
  const clampedMax = clampValue(externalMax, config)
  const sliderMin = config.sliderMin ?? config.min
  const sliderMax = config.sliderMax ?? config.max
  const mappedMin = toSliderValue(clampedMin, config)
  const mappedMax = toSliderValue(clampedMax, config)
  const sliderValue = [Math.min(mappedMin, mappedMax), Math.max(mappedMin, mappedMax)]
  const [value, setValue] = useState<readonly number[]>(sliderValue)
  const [open, setOpen] = useState(false)
  const actualValue = value.map((item) => fromSliderValue(item, config))
  const displayedValue = open ? actualValue : [externalMin, externalMax]
  const active = !isFullRange(displayedValue, config)

  function commit(nextValue: readonly number[]) {
    setValue(nextValue)
    const [nextMin, nextMax] = nextValue.map((item) => fromSliderValue(item, config))
    onValueCommit(
      nextMin === config.min ? '' : String(nextMin),
      nextMax === config.max ? '' : String(nextMax),
    )
  }

  function clear() {
    commit([sliderMin, sliderMax])
  }

  const currentMin = `${config.formatValue(actualValue[0])}${config.overflowMin && actualValue[0] === config.min ? '−' : ''}`
  const currentMax = `${config.formatValue(actualValue[1])}${config.overflowMax && actualValue[1] === config.max ? '+' : ''}`

  return (
    <Popover.Root
      open={open}
      onOpenChange={(nextOpen) => {
        setOpen(nextOpen)
        if (nextOpen) {
          setValue(sliderValue)
        }
      }}
    >
      <span className={styles['range-filter-wrap']} data-active={active || undefined} data-filter={config.key}>
        <Popover.Trigger
          className={styles['range-filter-trigger']}
          data-active={active || undefined}
          aria-label={`${config.label}: ${formatRange(displayedValue, config)}`}
        >
          <span className={styles['range-trigger-mark']} aria-hidden="true">
            {config.icon ?? config.triggerLabel}
          </span>
          <span className={styles['range-trigger-value']}>{formatRange(displayedValue, config, config.defaultLabel)}</span>
          {!active ? <ChevronDown className={styles['range-trigger-chevron']} aria-hidden="true" /> : null}
        </Popover.Trigger>
        {active ? (
          <button className={styles['range-trigger-clear']} type="button" onClick={clear} aria-label={`Clear ${config.label} filter`}>
            <X />
          </button>
        ) : null}
      </span>

      <Popover.Portal>
        <Popover.Positioner
          className={styles['range-popover-positioner']}
          positionMethod="fixed"
          sideOffset={3}
          align="center"
          collisionAvoidance={{ side: 'none', align: 'shift' }}
        >
          <Popover.Popup className={styles['range-popover']}>
            <Popover.Title className={styles['sr-only']}>{config.label}</Popover.Title>

            <Slider.Root
              className={styles['range-slider']}
              data-active={active || undefined}
              value={value}
              min={sliderMin}
              max={sliderMax}
              step={config.sliderStep ?? config.step}
              largeStep={config.sliderLargeStep ?? config.largeStep}
              thumbCollisionBehavior="none"
              thumbAlignment="center"
              onValueChange={setValue}
              onValueCommitted={commit}
            >
              <Slider.Control className={styles['range-slider-control']}>
                <Slider.Track className={styles['range-slider-track']}>
                  <Slider.Indicator className={styles['range-slider-indicator']} />
                  <Slider.Thumb
                    className={styles['range-slider-thumb']}
                    index={0}
                    getAriaLabel={() => `${config.label} minimum`}
                    getAriaValueText={(_, thumbValue) => {
                      const actual = fromSliderValue(thumbValue, config)
                      return config.formatAriaValue(actual, config.overflowMin === true && actual === config.min ? 'lower' : null)
                    }}
                  >
                    <span className={styles['range-thumb-value']} aria-hidden="true">{currentMin}</span>
                  </Slider.Thumb>
                  <Slider.Thumb
                    className={styles['range-slider-thumb']}
                    index={1}
                    getAriaLabel={() => `${config.label} maximum`}
                    getAriaValueText={(_, thumbValue) => {
                      const actual = fromSliderValue(thumbValue, config)
                      return config.formatAriaValue(actual, config.overflowMax === true && actual === config.max ? 'higher' : null)
                    }}
                  >
                    <span className={styles['range-thumb-value']} aria-hidden="true">{currentMax}</span>
                  </Slider.Thumb>
                </Slider.Track>
              </Slider.Control>
            </Slider.Root>
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  )
}
