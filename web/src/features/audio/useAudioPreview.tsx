import { useEffect, useRef, useState } from 'react'
import { previewUrl } from '../../shared/urls'
import type { BeatmapMetadata } from '../../shared/types'

type UseAudioPreviewOptions = {
  onError: (message: string) => void
}

const initialVolume = 0.25

export function useAudioPreview({ onError }: UseAudioPreviewOptions) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const hideTimerRef = useRef<number | null>(null)
  const [activeBeatmap, setActiveBeatmap] = useState<BeatmapMetadata | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const volumeRef = useRef(initialVolume)
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
      setDuration(0)
      audio.src = previewUrl(beatmapsetId)
      audio.currentTime = 0
      audio.volume = volumeRef.current
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
  }

  function changeVolume(value: string) {
    const audio = audioRef.current
    const nextVolume = Number(value)

    if (!Number.isFinite(nextVolume)) {
      return
    }

    const clampedVolume = Math.min(1, Math.max(0, nextVolume))
    volumeRef.current = clampedVolume
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

  function handleEnded() {
    setIsPlaying(false)
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
        onEnded={handleEnded}
      />
    ),
    activeBeatmap,
    audioRef,
    isPlaying,
    duration,
    volume: initialVolume,
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
