import type { BeatmapMetadata } from '../../shared/types'
import { BeatmapCard } from './BeatmapCard'

type ResultsListProps = {
  beatmaps: BeatmapMetadata[]
  onCopy: (beatmapId: number) => Promise<void>
  onSearch: (beatmapId: number) => Promise<void>
  isLoading: boolean
  onPlayPreview: (beatmap: BeatmapMetadata) => Promise<void>
  activePreviewSetId: number | null
  isPreviewPlaying: boolean
}

export function ResultsList({ beatmaps, onCopy, onSearch, isLoading, onPlayPreview, activePreviewSetId, isPreviewPlaying }: ResultsListProps) {
  return (
    <div className="result-list">
      {beatmaps.map((beatmap) => (
        <BeatmapCard
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
