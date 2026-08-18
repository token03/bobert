import type { ReactNode } from 'react'

type RangeFieldsProps = {
  label: string
  icon: ReactNode
  min: string
  max: string
  setMin: (value: string) => void
  setMax: (value: string) => void
}

export function RangeFields({ label, icon, min, max, setMin, setMax }: RangeFieldsProps) {
  function updateNumber(value: string, update: (value: string) => void) {
    if (/^\d*\.?\d*$/.test(value)) {
      update(value)
    }
  }

  return (
    <div className="range-field">
      <span aria-hidden="true">{icon}</span>
      <span className="range-inputs">
        <input
          inputMode="decimal"
          type="text"
          value={min}
          onChange={(event) => updateNumber(event.target.value, setMin)}
          placeholder="min"
          aria-label={`${label} minimum`}
        />
        <span aria-hidden="true">|</span>
        <input
          inputMode="decimal"
          type="text"
          value={max}
          onChange={(event) => updateNumber(event.target.value, setMax)}
          placeholder="max"
          aria-label={`${label} maximum`}
        />
      </span>
    </div>
  )
}
