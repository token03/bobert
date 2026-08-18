import type { CSSProperties } from 'react'
import { clampSliderValue, sliderValue } from './filters'

type SliderRangeFieldsProps = {
  label: string
  min: string
  max: string
  setMin: (value: string) => void
  setMax: (value: string) => void
}

type SliderStyle = CSSProperties & {
  '--range-left': string
  '--range-right': string
}

export function SliderRangeFields({ label, min, max, setMin, setMax }: SliderRangeFieldsProps) {
  const minValue = sliderValue(min, 0)
  const maxValue = sliderValue(max, 10)
  const left = `${minValue * 10}%`
  const right = `${100 - maxValue * 10}%`

  return (
    <div className="slider-field">
      <span className="slider-label">{label}</span>
      <strong>{minValue.toFixed(1)}</strong>
      <div className="dual-slider" style={{ '--range-left': left, '--range-right': right } as SliderStyle}>
        <input
          min="0"
          max="10"
          step="0.1"
          type="range"
          value={minValue}
          onChange={(event) => setMin(clampSliderValue(event.target.value, 0, maxValue))}
          aria-label={`${label} minimum`}
        />
        <input
          min="0"
          max="10"
          step="0.1"
          type="range"
          value={maxValue}
          onChange={(event) => setMax(clampSliderValue(event.target.value, minValue, 10))}
          aria-label={`${label} maximum`}
        />
      </div>
      <strong>{maxValue.toFixed(1)}</strong>
    </div>
  )
}
