import type { ReactNode } from 'react'
import { Popover } from '@base-ui/react/popover'
import { CaretDown, X } from '@phosphor-icons/react'
import styles from './RecommendForm.module.css'

type FilterPopoverProps = {
  filterKey: string
  label: string
  mark: ReactNode
  display: string
  active: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
  onClear: () => void
  popupClassName?: string
  children: ReactNode
}

export function FilterPopover({ filterKey, label, mark, display, active, open, onOpenChange, onClear, popupClassName, children }: FilterPopoverProps) {
  return (
    <Popover.Root open={open} onOpenChange={onOpenChange}>
      <span className={styles['range-filter-wrap']} data-active={active || undefined} data-filter={filterKey}>
        <Popover.Trigger className={styles['range-filter-trigger']} data-active={active || undefined} aria-label={`${label}: ${active ? display : 'Any'}`}>
          <span className={styles['range-trigger-mark']} aria-hidden="true">{mark}</span>
          <span className={styles['range-trigger-value']}>{display}</span>
          {!active ? <CaretDown className={styles['range-trigger-chevron']} aria-hidden="true" /> : null}
        </Popover.Trigger>
        {active ? (
          <button className={styles['range-trigger-clear']} type="button" onClick={onClear} aria-label={`Clear ${label} filter`}>
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
          <Popover.Popup className={`${styles['range-popover']}${popupClassName ? ` ${popupClassName}` : ''}`} initialFocus={(openType) => openType === 'keyboard'}>
            <Popover.Title className={styles['sr-only']}>{label}</Popover.Title>
            {children}
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  )
}
