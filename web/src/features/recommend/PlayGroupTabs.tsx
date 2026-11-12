import { useState } from 'react'
import { PreviewCard } from '@base-ui/react/preview-card'
import { displayTitle, formatFixedNumber } from '../../shared/format'
import type { BeatmapMetadata } from '../../shared/types'
import { beatmapUrl, coverUrl } from '../../shared/urls'
import styles from './PlayGroupTabs.module.css'

export type PlayGroup = {
  label: string
  beatmaps: BeatmapMetadata[]
}

type PlayGroupTabsProps = {
  groups: PlayGroup[]
  selected: number
  onSelect: (index: number) => void
}

export function PlayGroupTabs({ groups, selected, onSelect }: PlayGroupTabsProps) {
  const [previewed, setPreviewed] = useState<number | null>(null)

  return (
    <div className={styles['group-picker']}>
      <span>maps like your top plays:</span>
      <div className={styles['group-tabs']} role="group" aria-label="Top play groups">
        {groups.map((group, index) => (
          <PreviewCard.Root
            key={index}
            open={previewed === index}
            onOpenChange={(open, details) => {
              const fromPointer = details.reason === 'trigger-focus' && !(details.event.target as Element).matches(':focus-visible')
              if (!fromPointer) setPreviewed(open ? index : null)
            }}
          >
            <PreviewCard.Trigger
              delay={300}
              closeDelay={150}
              render={(
                <button
                  type="button"
                  aria-pressed={index === selected}
                  onClick={() => {
                    setPreviewed(null)
                    onSelect(index)
                  }}
                />
              )}
            >
              <span className={styles['group-label']}>{group.label}</span>
              <span className={styles['group-count']}>{group.beatmaps.length}</span>
            </PreviewCard.Trigger>
            <PreviewCard.Portal>
              <PreviewCard.Positioner className={styles['group-positioner']} sideOffset={6} align="start" collisionAvoidance={{ side: 'flip', align: 'shift' }}>
                <PreviewCard.Popup className={styles['group-popup']}>
                  {group.beatmaps.map((beatmap) => (
                    <a
                      key={beatmap.beatmap_id}
                      className={styles['group-map']}
                      href={beatmapUrl(beatmap)}
                      target="_blank"
                      rel="noreferrer"
                      title={`${displayTitle(beatmap)} [${beatmap.version}] · ${formatFixedNumber(beatmap.stars, 2)}★`}
                    >
                      {beatmap.beatmapset_id === null ? null : <img src={coverUrl(beatmap.beatmapset_id, true)} alt="" loading="lazy" />}
                    </a>
                  ))}
                </PreviewCard.Popup>
              </PreviewCard.Positioner>
            </PreviewCard.Portal>
          </PreviewCard.Root>
        ))}
      </div>
    </div>
  )
}
