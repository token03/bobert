import { Pause, Play, SpeakerHigh, SpeakerLow, SpeakerX } from '@phosphor-icons/react'
import { useEffect, useRef, useState } from 'react'
import type { CSSProperties, RefObject } from 'react'
import type { BeatmapMetadata } from '../../shared/types'
import styles from './AudioPreviewBar.module.css'

type AudioPreviewBarProps = {
  beatmap: BeatmapMetadata
  audioRef: RefObject<HTMLAudioElement | null>
  visible: boolean
  isPlaying: boolean
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
  audioRef,
  visible,
  isPlaying,
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
  const progressRef = useRef<HTMLInputElement>(null)
  const [localVolume, setLocalVolume] = useState(volume)
  const volumeValue = muted ? 0 : localVolume
  const VolumeIcon = volumeValue === 0 ? SpeakerX : volumeValue < 0.5 ? SpeakerLow : SpeakerHigh
  const progressStyle = { '--range-progress': '0%' } as CSSProperties
  const volumeStyle = { '--range-progress': `${volumeValue * 100}%` } as CSSProperties

  useEffect(() => {
    let frame = 0

    const updateProgress = () => {
      const audio = audioRef.current
      const input = progressRef.current

      if (audio && input) {
        const current = Math.min(safeDuration || audio.currentTime, Math.max(0, audio.currentTime))
        input.value = String(current)
        input.style.setProperty('--range-progress', `${safeDuration ? (current / safeDuration) * 100 : 0}%`)
      }

      if (audio && !audio.paused && !audio.ended) {
        frame = window.requestAnimationFrame(updateProgress)
      }
    }

    updateProgress()
    return () => window.cancelAnimationFrame(frame)
  }, [audioRef, isPlaying, safeDuration])

  return (
    <aside
      className={styles['audio-pill']}
      data-visible={visible}
      aria-label={`Audio preview player for ${beatmap.title}`}
      onPointerDown={onPointerDown}
      onFocus={onPointerDown}
    >
      <button type="button" className={styles['audio-control-button']} data-playing={isPlaying || undefined} onClick={onTogglePlay} aria-label={isPlaying ? 'Pause preview' : 'Play preview'}>
        {isPlaying ? <Pause weight="fill" /> : <Play weight="fill" />}
      </button>

      <div className={styles['audio-pill-main']}>
        <input
          ref={progressRef}
          type="range"
          min="0"
          max={safeDuration || 0}
          step="0.1"
          defaultValue="0"
          style={progressStyle}
          disabled={!safeDuration}
          onInput={(event) => {
            const input = event.currentTarget
            input.style.setProperty('--range-progress', `${safeDuration ? (Number(input.value) / safeDuration) * 100 : 0}%`)
            onSeek(input.value)
          }}
          aria-label="Preview progress"
        />
      </div>

      <div className={styles['audio-volume']}>
        <button type="button" className={styles['audio-control-button']} onClick={onToggleMuted} aria-label={muted ? 'Unmute preview' : 'Mute preview'}>
          <VolumeIcon />
        </button>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={volumeValue}
          style={volumeStyle}
          onChange={(event) => {
            setLocalVolume(Number(event.target.value))
            onVolumeChange(event.target.value)
          }}
          aria-label="Preview volume"
        />
      </div>
    </aside>
  )
}
