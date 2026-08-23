import { Check, Clock, Copy, Download, Metronome, Pause, Play, Search, Star } from 'lucide-react'
import { useState } from 'react'
import type { FocusEvent, KeyboardEvent, MouseEvent, ReactNode } from 'react'
import { displayArtist, displayTitle, formatFixedNumber, formatLength, formatMatch, formatNumber, statusClass, statusLabel } from '../../shared/format'
import { Stat } from '../../shared/ui/Stat'
import type { BeatmapMetadata } from '../../shared/types'
import { beatmapUrl, cardCoverUrl, coverUrl, userUrl } from '../../shared/urls'

type SourceBeatmapCardProps = {
  variant: 'source'
  beatmap: BeatmapMetadata
  onCopy: (beatmapId: number) => Promise<void>
  sweepDirection?: SweepDirection
  sweepPhase?: 'in' | 'out'
  onSweepEnd?: () => void
}

export type SweepDirection = 'left' | 'right'

type ResultBeatmapCardProps = {
  variant?: 'result'
  beatmap: BeatmapMetadata
  onCopy: (beatmapId: number) => Promise<void>
  onSearch: (beatmap: BeatmapMetadata, direction: SweepDirection) => Promise<void>
  isLoading: boolean
  onPlayPreview: (beatmap: BeatmapMetadata) => Promise<void>
  activePreviewSetId: number | null
  isPreviewPlaying: boolean
}

type BeatmapCardProps = SourceBeatmapCardProps | ResultBeatmapCardProps

export function BeatmapCard(props: BeatmapCardProps) {
  const { beatmap, onCopy } = props
  const [copied, setCopied] = useState(false)
  const variant = props.variant ?? 'result'
  const isSource = variant === 'source'
  const openBeatmap = () => window.open(beatmapUrl(beatmap), '_blank', 'noreferrer')
  const handleCardKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if ((event.target as HTMLElement).closest('a, button')) {
      return
    }

    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      openBeatmap()
    }
  }
  const handleCardBlur = (event: FocusEvent<HTMLElement>) => {
    if (!event.currentTarget.contains(event.relatedTarget)) {
      setCopied(false)
    }
  }

  if (isSource) {
    const sourceProps = props as SourceBeatmapCardProps
    const sweepClass = sourceProps.sweepDirection && sourceProps.sweepPhase ? ` source-sweep-${sourceProps.sweepPhase}-${sourceProps.sweepDirection}` : ''

    return (
      <article
        className={`beatmap-card beatmap-card-source source-card clickable-card${sweepClass}`}
        role="link"
        tabIndex={0}
        onClick={openBeatmap}
        onKeyDown={handleCardKeyDown}
        onMouseLeave={() => setCopied(false)}
        onBlur={handleCardBlur}
        onAnimationEnd={() => sourceProps.onSweepEnd?.()}
      >
        <BeatmapCover beatmap={beatmap} variant="source" />

        <div className="beatmap-card-content beatmap-card-content-source source-content">
          <div className="source-main">
            <div className="source-heading">
              <BeatmapSummary beatmap={beatmap} />
            </div>
          </div>

          <div className="source-meta">
            <CreatorLink beatmap={beatmap} />
            <span className={`status-label ${statusClass(beatmap.status)}`}>{statusLabel(beatmap.status)}</span>
          </div>
          <div className="source-meta-separator" aria-hidden="true" />

          <BeatmapStats beatmap={beatmap} variant="source" actions={<CardActions beatmap={beatmap} onCopy={onCopy} copied={copied} onCopiedChange={setCopied} />} />
        </div>
      </article>
    )
  }

  const hasPreview = beatmap.beatmapset_id !== null
  const resultProps = props as ResultBeatmapCardProps
  const isActivePreview = hasPreview && resultProps.activePreviewSetId === beatmap.beatmapset_id
  const isCoverActive = isActivePreview && resultProps.isPreviewPlaying

  return (
    <article
      className="beatmap-card beatmap-card-result beatmap-row clickable-card"
      role="link"
      tabIndex={0}
      onClick={openBeatmap}
      onKeyDown={handleCardKeyDown}
      onMouseLeave={() => setCopied(false)}
      onBlur={handleCardBlur}
    >
      <BeatmapCover beatmap={beatmap} variant="result" isCoverActive={isCoverActive} onPlayPreview={resultProps.onPlayPreview} />

      <div className="beatmap-card-content beatmap-card-content-result map-content">
        <div className="map-main">
          <div className="result-summary">
            <div className="result-copy">
              <div className="title-line">
                <span className="map-title">{displayTitle(beatmap)}</span>
              </div>
              <div className="artist-line">by {displayArtist(beatmap)}</div>
              <div className="version-line">
                <span>{beatmap.version ?? 'Unknown difficulty'}</span>
              </div>
            </div>
            <div className="result-meta">
              <CreatorLink beatmap={beatmap} />
              <div className="result-meta-slot">
                <div className="result-meta-details">
                  {beatmap.score !== undefined ? <span className="match-pill">{formatMatch(beatmap.score)} match</span> : null}
                  <span className={`status-label ${statusClass(beatmap.status)}`}>{statusLabel(beatmap.status)}</span>
                </div>
                <div className="result-meta-actions">
                  <CardActions beatmap={beatmap} onCopy={onCopy} onSearch={resultProps.onSearch} isLoading={resultProps.isLoading} copied={copied} onCopiedChange={setCopied} />
                </div>
              </div>
            </div>
          </div>
        </div>

        <BeatmapStats beatmap={beatmap} />
      </div>
    </article>
  )
}

function BeatmapCover({
  beatmap,
  variant,
  isCoverActive = false,
  onPlayPreview,
}: {
  beatmap: BeatmapMetadata
  variant: 'result' | 'source'
  isCoverActive?: boolean
  onPlayPreview?: (beatmap: BeatmapMetadata) => Promise<void>
}) {
  if (variant === 'source') {
    return (
      <div className="source-cover" aria-hidden="true">
        {beatmap.beatmapset_id ? <img src={cardCoverUrl(beatmap.beatmapset_id)} alt="" /> : <span className="cover-placeholder">osu!</span>}
      </div>
    )
  }

  const hasPreview = beatmap.beatmapset_id !== null

  return (
    <button
      className={isCoverActive ? 'cover-preview is-audio-active' : 'cover-preview'}
      type="button"
      disabled={!hasPreview}
      onClick={(event) => {
        event.stopPropagation()
        onPlayPreview?.(beatmap)
      }}
      aria-label={hasPreview ? (isCoverActive ? 'Pause preview' : 'Play preview') : 'No preview available'}
      title={hasPreview ? (isCoverActive ? 'Pause preview' : 'Play preview') : 'No preview available'}
    >
      {beatmap.beatmapset_id ? <img src={coverUrl(beatmap.beatmapset_id)} alt="" loading="lazy" decoding="async" /> : <span className="cover-placeholder">osu!</span>}
      {hasPreview ? (
        <span className="cover-play-overlay" aria-hidden="true">
          <span className="cover-play-button">{isCoverActive ? <Pause className="filled-icon" /> : <Play className="filled-icon" />}</span>
        </span>
      ) : null}
    </button>
  )
}

function BeatmapSummary({ beatmap }: { beatmap: BeatmapMetadata }) {
  return (
    <div className="beatmap-summary source-copy">
      <div className="title-line">
        <span className="map-title">{displayTitle(beatmap)}</span>
      </div>
      <div className="artist-line">by {displayArtist(beatmap)}</div>
      <div className="version-line">
        <span>{beatmap.version ?? 'Unknown difficulty'}</span>
      </div>
    </div>
  )
}

function BeatmapStats({ beatmap, variant = 'result', actions }: { beatmap: BeatmapMetadata; variant?: 'result' | 'source'; actions?: ReactNode }) {
  return (
    <div className="stat-strip">
      {variant === 'result' ? <div className="result-stat-separator" aria-hidden="true" /> : null}
      <div className="stat-row stat-row-main">
        <Stat label={<Star aria-label="Star" strokeWidth={2.5} />} value={formatFixedNumber(beatmap.stars, 2)} featured />
        <Stat label={<Clock aria-label="Length" strokeWidth={2.5} />} value={formatLength(beatmap.total_length)} featured />
        <Stat label={<Metronome aria-label="BPM" strokeWidth={2.5} />} value={formatNumber(beatmap.bpm, 0)} featured />
      </div>
      <div className="stat-side">
        {variant === 'source' ? <div className="source-stat-separator" aria-hidden="true" /> : null}
        <div className="stat-row stat-row-sub">
          <Stat label="AR" value={formatDifficultyStat(beatmap.ar)} />
          <Stat label="CS" value={formatDifficultyStat(beatmap.cs)} />
          <Stat label="OD" value={formatDifficultyStat(beatmap.accuracy)} />
          <Stat label="HP" value={formatDifficultyStat(beatmap.drain)} />
        </div>
        {actions}
      </div>
    </div>
  )
}

function formatDifficultyStat(value: number | null) {
  return value === 10 ? '10\u2008' : formatFixedNumber(value, 1)
}

function CardActions({
  beatmap,
  onCopy,
  onSearch,
  isLoading = false,
  copied,
  onCopiedChange,
}: {
  beatmap: BeatmapMetadata
  onCopy: (beatmapId: number) => Promise<void>
  onSearch?: (beatmap: BeatmapMetadata, direction: SweepDirection) => Promise<void>
  isLoading?: boolean
  copied: boolean
  onCopiedChange: (copied: boolean) => void
}) {
  return (
    <div className="row-actions" onClick={(event) => event.stopPropagation()}>
      {onSearch ? (
        <button
          type="button"
          disabled={isLoading}
          onClick={(event) =>
            handleActionClick(event, () => {
              const card = event.currentTarget.closest('.beatmap-card')
              const list = card?.closest('.result-list')
              const cardRect = card?.getBoundingClientRect()
              const listRect = list?.getBoundingClientRect()
              const direction = cardRect && listRect && cardRect.left >= listRect.left + listRect.width / 2 ? 'right' : 'left'
              return onSearch(beatmap, direction)
            })
          }
          aria-label="Search similar"
          title="Search similar"
        >
          <Search />
        </button>
      ) : null}
      <button
        type="button"
        className={copied ? 'copy-action is-copied' : 'copy-action'}
        onClick={(event) =>
          handleActionClick(event, () => {
            onCopiedChange(true)
            return onCopy(beatmap.beatmap_id)
          })
        }
        aria-label={copied ? 'Beatmap ID copied' : 'Copy beatmap ID'}
        title={copied ? 'Copied' : 'Copy ID'}
      >
        <Copy className="copy-action-icon" />
        <Check className="copy-check-icon" />
      </button>
      <button type="button" onClick={(event) => handleActionClick(event, () => window.location.assign(`osu://b/${beatmap.beatmap_id}`))} aria-label="Open beatmap in osu!" title="Open in osu!">
        <Download />
      </button>
    </div>
  )
}

function handleActionClick(event: MouseEvent<HTMLButtonElement>, action: () => void | Promise<void>) {
  event.stopPropagation()
  action()
}

function CreatorLink({ beatmap }: { beatmap: BeatmapMetadata }) {
  const creatorName = beatmap.creator ?? beatmap.user_id ?? 'unknown'

  if (beatmap.user_id) {
    return (
      <span className="creator-credit">
        mapped by{' '}
        <a className="mapper-link" href={userUrl(beatmap.user_id)} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}>
          {creatorName}
        </a>
      </span>
    )
  }

  return <span className="creator-credit">mapped by {creatorName}</span>
}
