import { ArrowSquareOut, Check, Clock, Copy, Heart, Metronome, Pause, Play, MagnifyingGlass, Star } from '@phosphor-icons/react'
import { useState } from 'react'
import type { FocusEvent, KeyboardEvent, MouseEvent, ReactNode } from 'react'
import { displayArtist, displayTitle, formatCount, formatDate, formatDifficultyStat, formatFixedNumber, formatLength, formatMatch, formatNumber, statusLabel } from '../../shared/format'
import { Stat } from '../../shared/ui/Stat'
import type { BeatmapMetadata } from '../../shared/types'
import { beatmapUrl, cardCoverUrl, coverUrl, userUrl } from '../../shared/urls'
import styles from './BeatmapCard.module.css'

export type SweepDirection = 'left' | 'right'

type CardProps = {
  beatmap: BeatmapMetadata
  onCopy: (beatmapId: number) => Promise<void>
  onPlayPreview: (beatmap: BeatmapMetadata) => Promise<void>
  activePreviewSetId: number | null
  isPreviewPlaying: boolean
}

type SourceBeatmapCardProps = CardProps & {
  sweepDirection?: SweepDirection
  sweepPhase?: 'in' | 'out'
  onSweepEnd?: () => void
}

type ResultBeatmapCardProps = CardProps & {
  onSearch: (beatmap: BeatmapMetadata, direction: SweepDirection) => Promise<void>
  isLoading: boolean
}

function useCardLink(beatmap: BeatmapMetadata) {
  const [copied, setCopied] = useState(false)
  const openBeatmap = () => window.open(beatmapUrl(beatmap), '_blank', 'noreferrer')

  return {
    copied,
    setCopied,
    linkProps: {
      role: 'link',
      tabIndex: 0,
      onClick: openBeatmap,
      onKeyDown: (event: KeyboardEvent<HTMLElement>) => {
        if ((event.target as HTMLElement).closest('a, button')) {
          return
        }
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          openBeatmap()
        }
      },
      onMouseLeave: () => setCopied(false),
      onBlur: (event: FocusEvent<HTMLElement>) => {
        if (!event.currentTarget.contains(event.relatedTarget)) {
          setCopied(false)
        }
      },
    },
  }
}

export function SourceBeatmapCard({ beatmap, onCopy, onPlayPreview, activePreviewSetId, isPreviewPlaying, sweepDirection, sweepPhase, onSweepEnd }: SourceBeatmapCardProps) {
  const { copied, setCopied, linkProps } = useCardLink(beatmap)

  return (
    <article
      className={`${styles['beatmap-card']} ${styles['source-card']} ${styles['clickable-card']}`}
      data-beatmap-card
      data-card-variant="source"
      data-sweep-direction={sweepDirection}
      data-sweep-phase={sweepPhase}
      {...linkProps}
      onAnimationEnd={(event) => {
        if (event.target === event.currentTarget && sweepPhase) {
          onSweepEnd?.()
        }
      }}
    >
      <BeatmapCover
        className={`${styles['source-cover']} ${styles['cover-preview']}`}
        src={beatmap.beatmapset_id === null ? null : cardCoverUrl(beatmap.beatmapset_id)}
        beatmap={beatmap}
        isActive={isPreviewPlaying && activePreviewSetId === beatmap.beatmapset_id}
        onPlayPreview={onPlayPreview}
      />

      <div className={`${styles['beatmap-card-content']} ${styles['source-content']}`}>
        <BeatmapSummary className={styles['source-copy']} beatmap={beatmap} />

        <div className={styles['source-meta']}>
          <CreatorLink beatmap={beatmap} />
          <StatusLabel beatmap={beatmap} date={beatmap.ranked_date ? formatDate(beatmap.ranked_date) : null} />
        </div>
        <div className={styles['source-meta-separator']} aria-hidden="true" />

        <BeatmapStats beatmap={beatmap} variant="source">
          <CardActions beatmap={beatmap} onCopy={onCopy} copied={copied} onCopiedChange={setCopied} />
        </BeatmapStats>
      </div>
    </article>
  )
}

export function ResultBeatmapCard({ beatmap, onCopy, onSearch, isLoading, onPlayPreview, activePreviewSetId, isPreviewPlaying }: ResultBeatmapCardProps) {
  const { copied, setCopied, linkProps } = useCardLink(beatmap)

  return (
    <article className={`${styles['beatmap-card']} ${styles['beatmap-card-result']} ${styles['beatmap-row']} ${styles['clickable-card']}`} data-beatmap-card {...linkProps}>
      <BeatmapCover
        className={styles['cover-preview']}
        src={beatmap.beatmapset_id === null ? null : coverUrl(beatmap.beatmapset_id)}
        lazy
        beatmap={beatmap}
        isActive={isPreviewPlaying && activePreviewSetId === beatmap.beatmapset_id}
        onPlayPreview={onPlayPreview}
      />

      <div className={`${styles['beatmap-card-content']} ${styles['map-content']}`}>
        <div className={styles['map-main']}>
          <div className={styles['result-summary']}>
            <BeatmapSummary className={styles['result-copy']} beatmap={beatmap} />
            <div className={styles['result-meta']}>
              <CreatorLink beatmap={beatmap} />
              <div className={styles['result-meta-slot']}>
                <div className={styles['result-meta-details']}>
                  {'score' in beatmap ? (
                    <span className={styles['match-pill']}>{formatMatch(beatmap.score)} match</span>
                  ) : beatmap.play_count != null ? (
                    <span className={styles['plays-pill']}>{formatCount(beatmap.play_count)} plays</span>
                  ) : null}
                  <StatusLabel beatmap={beatmap} date={beatmap.ranked_date?.slice(0, 4) ?? null} />
                </div>
                <div className={styles['result-meta-actions']}>
                  <CardActions beatmap={beatmap} onCopy={onCopy} onSearch={onSearch} isLoading={isLoading} copied={copied} onCopiedChange={setCopied} />
                </div>
              </div>
            </div>
          </div>
        </div>

        <BeatmapStats beatmap={beatmap} variant="result" />
      </div>
    </article>
  )
}

type BeatmapCoverProps = {
  className: string
  src: string | null
  lazy?: boolean
  beatmap: BeatmapMetadata
  isActive: boolean
  onPlayPreview: (beatmap: BeatmapMetadata) => Promise<void>
}

function BeatmapCover({ className, src, lazy = false, beatmap, isActive, onPlayPreview }: BeatmapCoverProps) {
  const label = src === null ? 'No preview available' : isActive ? 'Pause preview' : 'Play preview'

  return (
    <button
      className={className}
      data-audio-active={isActive || undefined}
      type="button"
      disabled={src === null}
      onClick={(event) => {
        event.stopPropagation()
        void onPlayPreview(beatmap)
      }}
      aria-label={label}
      title={label}
    >
      {src !== null ? (
        <>
          <img
            key={src}
            src={src}
            alt=""
            loading={lazy ? 'lazy' : undefined}
            decoding="async"
            onLoad={(event) => { event.currentTarget.dataset.loaded = 'true' }}
            onError={(event) => { event.currentTarget.hidden = true }}
          />
          <span className={styles['cover-play-overlay']} aria-hidden="true">
            <span className={styles['cover-play-button']}>{isActive ? <Pause weight="fill" /> : <Play weight="fill" />}</span>
          </span>
        </>
      ) : null}
    </button>
  )
}

const statusIcons: Record<string, ReactNode> = {
  ranked: <Star weight="fill" />,
  approved: <Star weight="fill" />,
  qualified: <Star weight="fill" />,
  loved: <Heart weight="fill" />,
  graveyard: <Tombstone />,
}

function Tombstone() {
  return (
    <svg viewBox="0 0 256 256" fill="currentColor">
      <path d="M48 232V96a80 80 0 0 1 160 0v136Z" />
    </svg>
  )
}

function StatusLabel({ beatmap, date }: { beatmap: BeatmapMetadata; date: string | null }) {
  const status = statusLabel(beatmap.status)
  const icon = statusIcons[status]

  return (
    <span className={styles['status-label']}>
      {icon ? <span className={styles['status-icon']} data-status={status} aria-hidden="true">{icon}</span> : null}
      <span>{status}{date ? ` ${date}` : ''}</span>
    </span>
  )
}

function BeatmapSummary({ className, beatmap }: { className: string; beatmap: BeatmapMetadata }) {
  return (
    <div className={className}>
      <div className={styles['map-title']}>{displayTitle(beatmap)}</div>
      <div className={styles['artist-line']}>by {displayArtist(beatmap)}</div>
      <div className={styles['version-line']}>{beatmap.version ?? 'Unknown difficulty'}</div>
    </div>
  )
}

type BeatmapStatsProps = {
  beatmap: BeatmapMetadata
  variant: 'result' | 'source'
  children?: ReactNode
}

function BeatmapStats({ beatmap, variant, children }: BeatmapStatsProps) {
  return (
    <div className={styles['stat-strip']}>
      {variant === 'result' ? <div className={styles['result-stat-separator']} aria-hidden="true" /> : null}
      <div className={`${styles['stat-row']} ${styles['stat-row-main']}`}>
        <Stat className={styles['stat-item']} label={<Star aria-label="Star" />} value={formatFixedNumber(beatmap.stars, 2)} featured />
        <Stat className={styles['stat-item']} label={<Clock aria-label="Length" />} value={formatLength(beatmap.total_length)} featured />
        <Stat className={styles['stat-item']} label={<Metronome aria-label="BPM" />} value={formatNumber(beatmap.bpm, 0)} featured />
      </div>
      <div className={styles['stat-side']}>
        {variant === 'source' ? <div className={styles['source-stat-separator']} aria-hidden="true" /> : null}
        <div className={`${styles['stat-row']} ${styles['stat-row-sub']}`}>
          <Stat className={styles['stat-item']} label="AR" value={formatDifficultyStat(beatmap.ar)} />
          <Stat className={styles['stat-item']} label="CS" value={formatDifficultyStat(beatmap.cs)} />
          <Stat className={styles['stat-item']} label="OD" value={formatDifficultyStat(beatmap.accuracy)} />
          <Stat className={styles['stat-item']} label="HP" value={formatDifficultyStat(beatmap.drain)} />
        </div>
        {children}
      </div>
    </div>
  )
}

type CardActionsProps = {
  beatmap: BeatmapMetadata
  onCopy: (beatmapId: number) => Promise<void>
  onSearch?: (beatmap: BeatmapMetadata, direction: SweepDirection) => Promise<void>
  isLoading?: boolean
  copied: boolean
  onCopiedChange: (copied: boolean) => void
}

function CardActions({ beatmap, onCopy, onSearch, isLoading = false, copied, onCopiedChange }: CardActionsProps) {
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
          <MagnifyingGlass />
          <span className={styles['action-label']}>Search similar</span>
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
        <span className={styles['action-label']}>{copied ? 'Copied' : 'Copy ID'}</span>
      </button>
      <button className={styles['open-action']} type="button" onClick={(event) => handleActionClick(event, () => window.location.assign(`osu://b/${beatmap.beatmap_id}`))} aria-label="Open beatmap in osu!" title="Open in osu!">
        <ArrowSquareOut />
        <span className={styles['action-label']}>osu!</span>
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
