import type { BeatmapMetadata } from '../../shared/types'
import { BeatmapCard } from './BeatmapCard'
import type { SweepDirection } from './BeatmapCard'
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
      {beatmaps.map((beatmap, index) => (
        <BeatmapCard
          key={beatmap.beatmap_id}
          beatmap={beatmap}
          onCopy={onCopy}
          onSearch={onSearch}
          isLoading={isLoading}
          onPlayPreview={onPlayPreview}
          activePreviewSetId={activePreviewSetId}
          isPreviewPlaying={isPreviewPlaying}
          revealIndex={index < 12 ? index : undefined}
        />
      ))}
    </div>
  )
}
