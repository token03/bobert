import { useState } from 'react'
import { Tag } from '@phosphor-icons/react'
import { FilterPopover } from './FilterPopover'
import { statuses } from './filters'
import styles from './RecommendForm.module.css'

type Status = (typeof statuses)[number]

type StatusFilterProps = {
  value: Status | undefined
  onValueCommit: (value: Status | undefined) => void
}

const statusLabels: Record<Status, string> = { ranked: 'Ranked', loved: 'Loved', unranked: 'Unranked' }

export function StatusFilter({ value, onValueCommit }: StatusFilterProps) {
  const [open, setOpen] = useState(false)

  return (
    <FilterPopover
      filterKey="status"
      label="Status"
      mark={<Tag />}
      display={value ? statusLabels[value] : 'Status'}
      active={value !== undefined}
      open={open}
      onOpenChange={setOpen}
      onClear={() => onValueCommit(undefined)}
      popupClassName={styles['status-popover']}
    >
      <div className={styles['status-options']} role="group" aria-label="Status">
        {statuses.map((status) => (
          <button
            key={status}
            className={styles['status-option']}
            type="button"
            data-status={status}
            aria-pressed={value === status}
            onClick={() => {
              onValueCommit(value === status ? undefined : status)
              setOpen(false)
            }}
          >
            {statusLabels[status]}
          </button>
        ))}
      </div>
    </FilterPopover>
  )
}
