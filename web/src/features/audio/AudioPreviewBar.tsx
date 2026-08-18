import { Pause, Play, Volume2, VolumeX } from 'lucide-react'
import type { BeatmapMetadata } from '../../shared/types'

type AudioPreviewBarProps = {
  beatmap: BeatmapMetadata
  visible: boolean
  isPlaying: boolean
  currentTime: number
  duration: number
  volume: number
  muted: boolean
  onTogglePlay: () => Promise<void>
  onSeek: (value: string) => void
  onToggleMuted: () => void
  onVolumeChange: (value: string) => void
  onPointerDown: () => void
}

export function AudioPreviewBar({
  beatmap,
  visible,
  isPlaying,
  currentTime,
  duration,
  volume,
  muted,
  onTogglePlay,
  onSeek,
  onToggleMuted,
  onVolumeChange,
  onPointerDown,
}: AudioPreviewBarProps) {
  const safeDuration = Number.isFinite(duration) ? Math.max(0, duration) : 0
  const safeCurrentTime = Math.min(safeDuration || currentTime, Math.max(0, currentTime))
  const volumeValue = muted ? 0 : volume

  return (
    <aside
      className={visible ? 'audio-pill is-visible' : 'audio-pill is-hidden'}
      aria-label={`Audio preview player for ${beatmap.title}`}
      onPointerDown={onPointerDown}
      onFocus={onPointerDown}
    >
      <button type="button" className="audio-control-button" onClick={onTogglePlay} aria-label={isPlaying ? 'Pause preview' : 'Play preview'}>
        {isPlaying ? <Pause className="filled-icon" /> : <Play className="filled-icon" />}
      </button>

      <div className="audio-pill-main">
        <input
          className="audio-progress"
          type="range"
          min="0"
          max={safeDuration || 0}
          step="0.1"
          value={safeCurrentTime}
          disabled={!safeDuration}
          onChange={(event) => onSeek(event.target.value)}
          aria-label="Preview progress"
        />
      </div>

      <div className="audio-volume">
        <button type="button" className="audio-control-button" onClick={onToggleMuted} aria-label={muted ? 'Unmute preview' : 'Mute preview'}>
          {muted || volume === 0 ? <VolumeX /> : <Volume2 />}
        </button>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={volumeValue}
          onChange={(event) => onVolumeChange(event.target.value)}
          aria-label="Preview volume"
        />
      </div>
    </aside>
  )
}
