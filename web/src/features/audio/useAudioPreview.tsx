import { useEffect, useRef, useState } from 'react'
import { previewUrl } from '../../shared/urls'
import type { BeatmapMetadata } from '../../shared/types'

type UseAudioPreviewOptions = {
  onError: (message: string) => void
}

export function useAudioPreview({ onError }: UseAudioPreviewOptions) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const hideTimerRef = useRef<number | null>(null)
  const [activeBeatmap, setActiveBeatmap] = useState<BeatmapMetadata | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.25)
  const [muted, setMuted] = useState(false)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    return () => {
      if (hideTimerRef.current !== null) {
        window.clearTimeout(hideTimerRef.current)
      }
    }
  }, [])

  function clearHideTimer() {
    if (hideTimerRef.current !== null) {
      window.clearTimeout(hideTimerRef.current)
      hideTimerRef.current = null
    }
  }

  function showTemporarily(forceHideTimer = false) {
    setVisible(true)

    if (isPlaying && !forceHideTimer) {
      clearHideTimer()
      return
    }

    clearHideTimer()

    hideTimerRef.current = window.setTimeout(() => {
      setVisible(false)
      hideTimerRef.current = null
    }, 3000)
  }

  async function playPreview(beatmap: BeatmapMetadata) {
    const beatmapsetId = beatmap.beatmapset_id
    const audio = audioRef.current

    if (!beatmapsetId || !audio) {
      return
    }

    setVisible(true)
    clearHideTimer()

    try {
      if (activeBeatmap?.beatmapset_id === beatmapsetId) {
        if (audio.paused) {
          if (audio.duration && audio.currentTime >= audio.duration) {
            audio.currentTime = 0
          }
          await audio.play()
        } else {
          audio.pause()
        }
        return
      }

      setActiveBeatmap(beatmap)
      setCurrentTime(0)
      setDuration(0)
      audio.src = previewUrl(beatmapsetId)
      audio.currentTime = 0
      audio.volume = volume
      audio.muted = muted
      await audio.play()
    } catch (err) {
      setIsPlaying(false)
      onError(err instanceof Error ? err.message : 'Preview playback failed')
    }
  }

  async function toggleActive() {
    const audio = audioRef.current

    if (!audio || !activeBeatmap) {
      return
    }

    showTemporarily()

    try {
      if (audio.paused) {
        if (audio.duration && audio.currentTime >= audio.duration) {
          audio.currentTime = 0
        }
        await audio.play()
      } else {
        audio.pause()
      }
    } catch (err) {
      setIsPlaying(false)
      onError(err instanceof Error ? err.message : 'Preview playback failed')
    }
  }

  function seek(value: string) {
    const audio = audioRef.current
    const nextTime = Number(value)

    if (!audio || !Number.isFinite(nextTime)) {
      return
    }

    showTemporarily()
    audio.currentTime = nextTime
    setCurrentTime(nextTime)
  }

  function changeVolume(value: string) {
    const audio = audioRef.current
    const nextVolume = Number(value)

    if (!Number.isFinite(nextVolume)) {
      return
    }

    const clampedVolume = Math.min(1, Math.max(0, nextVolume))
    setVolume(clampedVolume)
    setMuted(false)

    if (audio) {
      audio.volume = clampedVolume
      audio.muted = false
    }

    showTemporarily()
  }

  function toggleMuted() {
    const audio = audioRef.current
    const nextMuted = !muted

    setMuted(nextMuted)
    if (audio) {
      audio.muted = nextMuted
    }

    showTemporarily()
  }

  function handleLoadedMetadata() {
    const audio = audioRef.current

    if (!audio) {
      return
    }

    setDuration(Number.isFinite(audio.duration) ? audio.duration : 0)
  }

  function handleTimeUpdate() {
    const audio = audioRef.current

    if (!audio) {
      return
    }

    setCurrentTime(audio.currentTime)
  }

  function handleEnded() {
    const audio = audioRef.current

    setIsPlaying(false)
    setCurrentTime(audio?.duration && Number.isFinite(audio.duration) ? audio.duration : 0)
    showTemporarily()
  }

  return {
    audioElement: (
      <audio
        ref={audioRef}
        preload="none"
        onPlay={() => {
          setIsPlaying(true)
          setVisible(true)
          clearHideTimer()
        }}
        onPause={() => {
          setIsPlaying(false)
          showTemporarily(true)
        }}
        onLoadedMetadata={handleLoadedMetadata}
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
      />
    ),
    activeBeatmap,
    isPlaying,
    currentTime,
    duration,
    volume,
    muted,
    visible,
    playPreview,
    toggleActive,
    seek,
    changeVolume,
    toggleMuted,
    showTemporarily,
  }
}
