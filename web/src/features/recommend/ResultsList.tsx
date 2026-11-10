import type { CSSProperties } from 'react'
import type { BeatmapMetadata } from '../../shared/types'
import { ResultBeatmapCard } from './BeatmapCard'
import type { SweepDirection } from './BeatmapCard'
import cardStyles from './BeatmapCard.module.css'
import styles from './ResultsList.module.css'

type ResultsListProps = {
  beatmaps: BeatmapMetadata[]
  onCopy: (beatmapId: number) => Promise<void>
  onSearch: (beatmap: BeatmapMetadata, direction: SweepDirection) => Promise<void>
  isLoading: boolean
  onPlayPreview: (beatmap: BeatmapMetadata) => Promise<void>
  activePreviewSetId: number | null
  isPreviewPlaying: boolean
}

export function ResultsList({ beatmaps, onCopy, onSearch, isLoading, onPlayPreview, activePreviewSetId, isPreviewPlaying }: ResultsListProps) {
  return (
    <div className={styles['result-list']} data-results-list>
      {beatmaps.map((beatmap) => (
        <ResultBeatmapCard
          key={beatmap.beatmap_id}
          beatmap={beatmap}
          onCopy={onCopy}
          onSearch={onSearch}
          isLoading={isLoading}
          onPlayPreview={onPlayPreview}
          activePreviewSetId={activePreviewSetId}
          isPreviewPlaying={isPreviewPlaying}
        />
      ))}
    </div>
  )
}

export function LoadingResultsList({ count = 100 }: { count?: number }) {
  return (
    <div className={`${styles['result-list']} ${styles['loading-result-list']}`} role="status" aria-label="Loading recommendations">
      {Array.from({ length: count }, (_, index) => (
        <div className={`${cardStyles['beatmap-card']} ${cardStyles['beatmap-card-result']} ${cardStyles['beatmap-row']} ${styles['loading-result-card']}`} key={index} style={{ '--i': index } as CSSProperties} aria-hidden="true">
          <div className={`${cardStyles['cover-preview']} ${styles['loading-result-cover']}`} />
          <div className={`${cardStyles['beatmap-card-content']} ${cardStyles['map-content']}`}>
            <div className={cardStyles['map-main']}>
              <div className={cardStyles['result-summary']}>
                <div className={`${cardStyles['result-copy']} ${styles['loading-result-copy']}`}>
                  <span className={`${styles['loading-result-line']} ${styles['loading-result-title']}`} />
                  <span className={`${styles['loading-result-line']} ${styles['loading-result-artist']}`} />
                  <span className={`${styles['loading-result-line']} ${styles['loading-result-version']}`} />
                </div>
                <div className={`${cardStyles['result-meta']} ${styles['loading-result-meta']}`}>
                  <span className={`${styles['loading-result-line']} ${styles['loading-result-creator']}`} />
                  <span className={`${styles['loading-result-line']} ${styles['loading-result-status']}`} />
                </div>
              </div>
            </div>
            <div className={cardStyles['stat-strip']}>
              <div className={cardStyles['result-stat-separator']} aria-hidden="true" />
              <div className={`${cardStyles['stat-row']} ${cardStyles['stat-row-main']}`}>
                {Array.from({ length: 3 }, (_, stat) => (
                  <div key={stat} className={cardStyles['stat-item']}>
                    <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-icon']}`} />
                    <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-main']}`} />
                  </div>
                ))}
              </div>
              <div className={cardStyles['stat-side']}>
                <div className={`${cardStyles['stat-row']} ${cardStyles['stat-row-sub']}`}>
                  {Array.from({ length: 4 }, (_, stat) => (
                    <div key={stat} className={cardStyles['stat-item']}>
                      <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-label']}`} />
                      <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-sub']}`} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
