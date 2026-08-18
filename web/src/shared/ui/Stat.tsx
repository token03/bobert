import type { ReactNode } from 'react'

type StatProps = {
  featured?: boolean
  label: ReactNode
  value: string
}

export function Stat({ featured = false, label, value }: StatProps) {
  return (
    <div className={featured ? 'stat-item featured-stat' : 'stat-item'}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
