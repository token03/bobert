import type { ReactNode } from 'react'

type StatProps = {
  className?: string
  featured?: boolean
  label: ReactNode
  value: string
}

export function Stat({ className, featured = false, label, value }: StatProps) {
  return (
    <div className={className} data-featured={featured || undefined}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
