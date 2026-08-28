import { Check, Clock, Copy, Download, Metronome, Pause, Play, Search, Star } from 'lucide-react'
import { useState } from 'react'
import type { CSSProperties, FocusEvent, KeyboardEvent, MouseEvent, ReactNode } from 'react'
import { displayArtist, displayTitle, formatFixedNumber, formatLength, formatMatch, formatNumber, statusLabel } from '../../shared/format'
import { Stat } from '../../shared/ui/Stat'
import type { BeatmapMetadata } from '../../shared/types'
import { beatmapUrl, cardCoverUrl, coverUrl, userUrl } from '../../shared/urls'
import styles from './BeatmapCard.module.css'

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
  revealIndex?: number
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
    return (
      <article
        className={`${styles['beatmap-card']} ${styles['source-card']} ${styles['clickable-card']}`}
        data-beatmap-card
        data-card-variant="source"
        data-sweep-direction={sourceProps.sweepDirection}
        data-sweep-phase={sourceProps.sweepPhase}
        role="link"
        tabIndex={0}
        onClick={openBeatmap}
        onKeyDown={handleCardKeyDown}
        onMouseLeave={() => setCopied(false)}
        onBlur={handleCardBlur}
        onAnimationEnd={() => sourceProps.onSweepEnd?.()}
      >
        <BeatmapCover beatmap={beatmap} variant="source" />

        <div className={`${styles['beatmap-card-content']} ${styles['source-content']}`}>
          <div className={styles['source-main']}>
            <div className={styles['source-heading']}>
              <BeatmapSummary beatmap={beatmap} />
            </div>
          </div>

          <div className={styles['source-meta']}>
            <CreatorLink beatmap={beatmap} />
            <span className={styles['status-label']} data-status={statusLabel(beatmap.status)}>{statusLabel(beatmap.status)}</span>
          </div>
          <div className={styles['source-meta-separator']} aria-hidden="true" />

          <BeatmapStats beatmap={beatmap} variant="source" actions={<CardActions beatmap={beatmap} onCopy={onCopy} copied={copied} onCopiedChange={setCopied} />} />
        </div>
      </article>
    )
  }

  const hasPreview = beatmap.beatmapset_id !== null
  const resultProps = props as ResultBeatmapCardProps
  const isActivePreview = hasPreview && resultProps.activePreviewSetId === beatmap.beatmapset_id
  const isCoverActive = isActivePreview && resultProps.isPreviewPlaying
  const shouldReveal = resultProps.revealIndex !== undefined

  return (
    <article
      className={`${styles['beatmap-card']} ${styles['beatmap-card-result']} ${styles['beatmap-row']} ${styles['clickable-card']}`}
      data-beatmap-card
      data-reveal={shouldReveal || undefined}
      style={shouldReveal ? { '--result-reveal-index': resultProps.revealIndex } as CSSProperties : undefined}
      role="link"
      tabIndex={0}
      onClick={openBeatmap}
      onKeyDown={handleCardKeyDown}
      onMouseLeave={() => setCopied(false)}
      onBlur={handleCardBlur}
    >
      <BeatmapCover beatmap={beatmap} variant="result" isCoverActive={isCoverActive} onPlayPreview={resultProps.onPlayPreview} />

      <div className={`${styles['beatmap-card-content']} ${styles['map-content']}`}>
        <div className={styles['map-main']}>
          <div className={styles['result-summary']}>
            <div className={styles['result-copy']}>
              <div className={styles['title-line']}>
                <span className={styles['map-title']}>{displayTitle(beatmap)}</span>
              </div>
              <div className={styles['artist-line']}>by {displayArtist(beatmap)}</div>
              <div className={styles['version-line']}>
                <span>{beatmap.version ?? 'Unknown difficulty'}</span>
              </div>
            </div>
            <div className={styles['result-meta']}>
              <CreatorLink beatmap={beatmap} />
              <div className={styles['result-meta-slot']}>
                <div className={styles['result-meta-details']}>
                  {beatmap.score !== undefined ? <span className={styles['match-pill']}>{formatMatch(beatmap.score)} match</span> : null}
                  <span className={styles['status-label']} data-status={statusLabel(beatmap.status)}>{statusLabel(beatmap.status)}</span>
                </div>
                <div className={styles['result-meta-actions']}>
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
      <div className={styles['source-cover']} aria-hidden="true">
        {beatmap.beatmapset_id ? <img key={beatmap.beatmapset_id} src={cardCoverUrl(beatmap.beatmapset_id)} alt="" onLoad={(event) => { event.currentTarget.dataset.loaded = 'true'; event.currentTarget.dataset.revealing = 'true' }} onAnimationEnd={(event) => { delete event.currentTarget.dataset.revealing }} onError={(event) => { event.currentTarget.hidden = true }} /> : null}
      </div>
    )
  }

  const hasPreview = beatmap.beatmapset_id !== null

  return (
    <button
      className={styles['cover-preview']}
      data-audio-active={isCoverActive || undefined}
      type="button"
      disabled={!hasPreview}
      onClick={(event) => {
        event.stopPropagation()
        onPlayPreview?.(beatmap)
      }}
      aria-label={hasPreview ? (isCoverActive ? 'Pause preview' : 'Play preview') : 'No preview available'}
      title={hasPreview ? (isCoverActive ? 'Pause preview' : 'Play preview') : 'No preview available'}
    >
      {beatmap.beatmapset_id ? <img key={beatmap.beatmapset_id} src={coverUrl(beatmap.beatmapset_id)} alt="" loading="lazy" decoding="async" onLoad={(event) => { event.currentTarget.dataset.loaded = 'true'; event.currentTarget.dataset.revealing = 'true' }} onAnimationEnd={(event) => { delete event.currentTarget.dataset.revealing }} onError={(event) => { event.currentTarget.hidden = true }} /> : null}
      {hasPreview ? (
        <span className={styles['cover-play-overlay']} aria-hidden="true">
          <span className={styles['cover-play-button']}>{isCoverActive ? <Pause className={styles['filled-icon']} /> : <Play className={styles['filled-icon']} />}</span>
        </span>
      ) : null}
    </button>
  )
}

function BeatmapSummary({ beatmap }: { beatmap: BeatmapMetadata }) {
  return (
    <div className={styles['source-copy']}>
      <div className={styles['title-line']}>
        <span className={styles['map-title']}>{displayTitle(beatmap)}</span>
      </div>
      <div className={styles['artist-line']}>by {displayArtist(beatmap)}</div>
      <div className={styles['version-line']}>
        <span>{beatmap.version ?? 'Unknown difficulty'}</span>
      </div>
    </div>
  )
}

function BeatmapStats({ beatmap, variant = 'result', actions }: { beatmap: BeatmapMetadata; variant?: 'result' | 'source'; actions?: ReactNode }) {
  return (
    <div className={styles['stat-strip']}>
      {variant === 'result' ? <div className={styles['result-stat-separator']} aria-hidden="true" /> : null}
      <div className={`${styles['stat-row']} ${styles['stat-row-main']}`}>
        <Stat className={styles['stat-item']} label={<Star aria-label="Star" strokeWidth={2.5} />} value={formatFixedNumber(beatmap.stars, 2)} featured />
        <Stat className={styles['stat-item']} label={<Clock aria-label="Length" strokeWidth={2.5} />} value={formatLength(beatmap.total_length)} featured />
        <Stat className={styles['stat-item']} label={<Metronome aria-label="BPM" strokeWidth={2.5} />} value={formatNumber(beatmap.bpm, 0)} featured />
      </div>
      <div className={styles['stat-side']}>
        {variant === 'source' ? <div className={styles['source-stat-separator']} aria-hidden="true" /> : null}
        <div className={`${styles['stat-row']} ${styles['stat-row-sub']}`}>
          <Stat className={styles['stat-item']} label="AR" value={formatDifficultyStat(beatmap.ar)} />
          <Stat className={styles['stat-item']} label="CS" value={formatDifficultyStat(beatmap.cs)} />
          <Stat className={styles['stat-item']} label="OD" value={formatDifficultyStat(beatmap.accuracy)} />
          <Stat className={styles['stat-item']} label="HP" value={formatDifficultyStat(beatmap.drain)} />
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
    <div className={styles['row-actions']} onClick={(event) => event.stopPropagation()}>
      {onSearch ? (
        <button
          type="button"
          disabled={isLoading}
          onClick={(event) =>
            handleActionClick(event, () => {
              const card = event.currentTarget.closest('[data-beatmap-card]')
              const list = card?.closest('[data-results-list]')
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
        className={styles['copy-action']}
        data-copied={copied || undefined}
        onClick={(event) =>
          handleActionClick(event, () => {
            onCopiedChange(true)
            return onCopy(beatmap.beatmap_id)
          })
        }
        aria-label={copied ? 'Beatmap ID copied' : 'Copy beatmap ID'}
        title={copied ? 'Copied' : 'Copy ID'}
      >
        <Copy className={styles['copy-action-icon']} />
        <Check className={styles['copy-check-icon']} />
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
      <span className={styles['creator-credit']}>
        mapped by{' '}
        <a className={styles['mapper-link']} href={userUrl(beatmap.user_id)} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}>
          {creatorName}
        </a>
      </span>
    )
  }

  return <span className={styles['creator-credit']}>mapped by {creatorName}</span>
}
