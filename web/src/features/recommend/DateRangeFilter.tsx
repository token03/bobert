import { useRef, useState } from 'react'
import { Popover } from '@base-ui/react/popover'
import { Select } from '@base-ui/react/select'
import { CalendarDots, CaretDown, CaretLeft, CaretRight, X } from '@phosphor-icons/react'
import styles from './RecommendForm.module.css'

export const minDateBound = '2007-10'
const boundYear = 2007
const monthLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

type DateRangeFilterProps = {
  minValue: string
  maxValue: string
  onValueCommit: (min: string, max: string) => void
}

function maxDateBound() {
  const now = new Date()
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`
}

function parseMonth(value: string) {
  const match = value.match(/^(\d{4})-(0[1-9]|1[0-2])$/)
  return match ? { year: match[1], month: match[2] } : null
}

function formatShort(value: string) {
  const parsed = parseMonth(value)
  if (!parsed) {
    return value
  }
  return `${monthLabels[Number(parsed.month) - 1]} ’${parsed.year.slice(2)}`
}

function formatRange(min: string, max: string) {
  if (!min && !max) {
    return 'Any'
  }
  if (min && max) {
    return min === max ? formatShort(min) : `${formatShort(min)}–${formatShort(max)}`
  }
  return min ? `≥${formatShort(min)}` : `≤${formatShort(max)}`
}

export function DateRangeFilter({ minValue, maxValue, onValueCommit }: DateRangeFilterProps) {
  const [open, setOpen] = useState(false)
  const upper = maxDateBound()
  const upperYear = upper.slice(0, 4)
  const [fromYear, setFromYear] = useState(() => parseMonth(minValue)?.year ?? upperYear)
  const [toYear, setToYear] = useState(() => parseMonth(maxValue)?.year ?? upperYear)
  const active = Boolean(minValue || maxValue)
  const display = active ? formatRange(minValue, maxValue) : 'Date'

  function clampRange(nextMin: string, nextMax: string, changed: 'min' | 'max'): [string, string] {
    if (nextMin && nextMin < minDateBound) {
      nextMin = minDateBound
    }
    if (nextMin && nextMin > upper) {
      nextMin = upper
    }
    if (nextMax && nextMax < minDateBound) {
      nextMax = minDateBound
    }
    if (nextMax && nextMax > upper) {
      nextMax = upper
    }
    if (nextMin && nextMax && nextMin > nextMax) {
      if (changed === 'min') {
        nextMax = nextMin
      } else {
        nextMin = nextMax
      }
    }
    return [nextMin, nextMax]
  }

  function pick(side: 'min' | 'max', year: string, month: string) {
    const value = `${year}-${month}`
    const current = side === 'min' ? minValue : maxValue
    if (current === value) {
      onValueCommit(side === 'min' ? '' : minValue, side === 'min' ? maxValue : '')
      return
    }
    const [nextMin, nextMax] = clampRange(
      side === 'min' ? value : minValue,
      side === 'min' ? maxValue : value,
      side,
    )
    if (nextMin) {
      setFromYear(nextMin.slice(0, 4))
    }
    if (nextMax) {
      setToYear(nextMax.slice(0, 4))
    }
    onValueCommit(nextMin, nextMax)
  }

  function monthDisabled(side: 'min' | 'max', year: string, month: string) {
    const value = `${year}-${month}`
    if (value < minDateBound || value > upper) {
      return true
    }
    const other = side === 'min' ? maxValue : minValue
    return Boolean(other && (side === 'min' ? value > other : value < other))
  }

  return (
    <Popover.Root
      open={open}
      onOpenChange={(nextOpen) => {
        setOpen(nextOpen)
        if (nextOpen) {
          setFromYear(parseMonth(minValue)?.year ?? upperYear)
          setToYear(parseMonth(maxValue)?.year ?? upperYear)
        }
      }}
    >
      <span className={styles['range-filter-wrap']} data-active={active || undefined} data-filter="date">
        <Popover.Trigger
          className={styles['range-filter-trigger']}
          data-active={active || undefined}
          aria-label={`Date: ${formatRange(minValue, maxValue)}`}
        >
          <span className={styles['range-trigger-mark']} aria-hidden="true">
            <CalendarDots />
          </span>
          <span className={styles['range-trigger-value']}>{display}</span>
          {!active ? <CaretDown className={styles['range-trigger-chevron']} aria-hidden="true" /> : null}
        </Popover.Trigger>
        {active ? (
          <button
            className={styles['range-trigger-clear']}
            type="button"
            onClick={() => onValueCommit('', '')}
            aria-label="Clear date filter"
          >
            <X />
          </button>
        ) : null}
      </span>

      <Popover.Portal>
        <Popover.Positioner
          className={styles['range-popover-positioner']}
          positionMethod="fixed"
          sideOffset={6}
          align="center"
          collisionAvoidance={{ side: 'flip', align: 'shift' }}
        >
          <Popover.Popup className={`${styles['range-popover']} ${styles['date-popover']}`}>
            <Popover.Title className={styles['sr-only']}>Date range</Popover.Title>
            <div className={styles['date-columns']}>
              <MonthPanel
                label="From"
                viewYear={fromYear}
                selected={parseMonth(minValue)}
                minValue={minValue}
                maxValue={maxValue}
                upperYear={upperYear}
                onViewYear={setFromYear}
                onPick={(year, month) => pick('min', year, month)}
                isDisabled={(year, month) => monthDisabled('min', year, month)}
              />
              <MonthPanel
                label="To"
                viewYear={toYear}
                selected={parseMonth(maxValue)}
                minValue={minValue}
                maxValue={maxValue}
                upperYear={upperYear}
                onViewYear={setToYear}
                onPick={(year, month) => pick('max', year, month)}
                isDisabled={(year, month) => monthDisabled('max', year, month)}
              />
            </div>
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  )
}

type MonthPanelProps = {
  label: string
  viewYear: string
  selected: { year: string; month: string } | null
  minValue: string
  maxValue: string
  upperYear: string
  onViewYear: (year: string) => void
  onPick: (year: string, month: string) => void
  isDisabled: (year: string, month: string) => boolean
}

function MonthPanel({ label, viewYear, selected, minValue, maxValue, upperYear, onViewYear, onPick, isDisabled }: MonthPanelProps) {
  const year = Number(viewYear)
  const listRef = useRef<HTMLDivElement>(null)

  function centerSelected() {
    requestAnimationFrame(() => {
      const list = listRef.current
      const item = list?.querySelector<HTMLElement>('[data-selected]')
      if (list && item) {
        list.scrollTop += item.getBoundingClientRect().top - list.getBoundingClientRect().top - list.clientHeight / 2 + item.clientHeight / 2
      }
    })
  }

  return (
    <div className={styles['date-column']}>
      <span className={styles['date-label']}>{label}</span>
      <div className={styles['date-year']}>
        <button
          className={styles['date-step']}
          type="button"
          disabled={year <= boundYear}
          onClick={() => onViewYear(String(year - 1))}
          aria-label={`${label} previous year`}
        >
          <CaretLeft />
        </button>
        <Select.Root
          value={viewYear}
          onValueChange={(value) => { if (value !== null) onViewYear(value) }}
          onOpenChange={(open) => { if (open) centerSelected() }}
        >
          <Select.Trigger className={styles['date-year-value']} aria-label={`${label} year`}>
            <Select.Value />
            <Select.Icon className={styles['select-icon']}><CaretDown /></Select.Icon>
          </Select.Trigger>
          <Select.Portal>
            <Select.Positioner
              className={styles['select-positioner']}
              sideOffset={6}
              align="center"
              alignItemWithTrigger={false}
            >
              <Select.Popup className={`${styles['select-popup']} ${styles['date-year-popup']}`}>
                <Select.List ref={listRef} className={`${styles['select-list']} ${styles['date-year-list']}`}>
                  {Array.from({ length: Number(upperYear) - boundYear + 1 }, (_, index) => {
                    const optionYear = String(Number(upperYear) - index)
                    return (
                      <Select.Item
                        className={`${styles['select-item']} ${styles['date-year-item']}`}
                        key={optionYear}
                        value={optionYear}
                      >
                        <Select.ItemText>{optionYear}</Select.ItemText>
                      </Select.Item>
                    )
                  })}
                </Select.List>
              </Select.Popup>
            </Select.Positioner>
          </Select.Portal>
        </Select.Root>
        <button
          className={styles['date-step']}
          type="button"
          disabled={year >= Number(upperYear)}
          onClick={() => onViewYear(String(year + 1))}
          aria-label={`${label} next year`}
        >
          <CaretRight />
        </button>
      </div>
      <div className={styles['date-months']} role="group" aria-label={`${label} month`}>
        {monthLabels.map((monthLabel, index) => {
          const month = String(index + 1).padStart(2, '0')
          const isSelected = selected?.year === viewYear && selected.month === month
          const value = `${viewYear}-${month}`
          const inRange = Boolean(minValue || maxValue) && value >= (minValue || minDateBound) && value <= (maxValue || maxDateBound())
          return (
            <button
              key={month}
              className={styles['date-month']}
              type="button"
              data-selected={isSelected || undefined}
              data-in-range={inRange || undefined}
              data-endpoint={value === minValue || value === maxValue || undefined}
              disabled={isDisabled(viewYear, month)}
              onClick={() => onPick(viewYear, month)}
              aria-pressed={isSelected}
              aria-label={`${monthLabel} ${viewYear}`}
            >
              {monthLabel}
            </button>
          )
        })}
      </div>
    </div>
  )
}
