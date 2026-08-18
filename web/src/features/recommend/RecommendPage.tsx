import { useEffect, useRef, useState } from 'react'
import { AudioPreviewBar } from '../audio/AudioPreviewBar'
import { useAudioPreview } from '../audio/useAudioPreview'
import { useTurnstile } from '../turnstile/useTurnstile'
import { fetchDefaultRecommendations } from '../../shared/api'
import { copyText } from '../../shared/copy'
import type { BeatmapMetadata, DefaultRecommendResponse, RecommendResponse } from '../../shared/types'
import { buildRecommendRequest, defaultFilters, normalizeBeatmapInput, parseBeatmapId } from './filters'
import type { RecommendFormValues } from './filters'
import { RecommendForm } from './RecommendForm'
import { readCachedRecommendation, recommendSearchParams, valuesFromRecommendSearch, writeCachedRecommendation } from './recommendHistory'
import { BeatmapCard } from './BeatmapCard'
import { ResultsList } from './ResultsList'
import { useRecommend } from './useRecommend'
import { useRecommendForm } from './useRecommendForm'

type HistoryMode = 'push' | 'replace' | 'none'

export function RecommendPage() {
  const [error, setError] = useState('')
  const [response, setResponse] = useState<RecommendResponse | null>(null)
  const [defaultResponse, setDefaultResponse] = useState<DefaultRecommendResponse | null>(null)
  const [defaultLoading, setDefaultLoading] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const turnstile = useTurnstile()
  const recommend = useRecommend()
  const form = useRecommendForm()
  const audio = useAudioPreview({ onError: setError })
  const defaultRequestId = useRef(0)
  const recommendRequestId = useRef(0)
  const rangeSearchTimeout = useRef<number | null>(null)
  const isLoading = isRunning || recommend.isPending

  function clearRangeSearchTimeout() {
    if (rangeSearchTimeout.current !== null) {
      window.clearTimeout(rangeSearchTimeout.current)
      rangeSearchTimeout.current = null
    }
  }

  useEffect(() => {
    let cancelled = false

    async function loadDefaultRecommendations() {
      const requestId = defaultRequestId.current + 1
      defaultRequestId.current = requestId
      setDefaultLoading(true)

      try {
        const data = await fetchDefaultRecommendations()
        if (!cancelled && defaultRequestId.current === requestId) {
          setDefaultResponse(data)
        }
      } catch (err) {
        if (!cancelled && defaultRequestId.current === requestId) {
          setError(err instanceof Error ? err.message : 'Request failed')
        }
      } finally {
        if (!cancelled && defaultRequestId.current === requestId) {
          setDefaultLoading(false)
        }
      }
    }

    function restoreFromLocation() {
      const values = valuesFromRecommendSearch(window.location.search)

      if (!values) {
        form.reset(defaultFilters)
        setResponse(null)
        setError('')
        void loadDefaultRecommendations()
        return
      }

      form.reset(values)
      defaultRequestId.current += 1
      setDefaultResponse(null)
      setDefaultLoading(false)

      const cached = readCachedRecommendation(values)
      if (cached) {
        setResponse(cached.response)
        setError('')
        return
      }

      setResponse(null)
      form.reset(defaultFilters)
      setError('')
      window.history.replaceState(null, '', window.location.pathname)
      void loadDefaultRecommendations()
    }

    restoreFromLocation()
    window.addEventListener('popstate', restoreFromLocation)

    return () => {
      cancelled = true
      if (rangeSearchTimeout.current !== null) {
        window.clearTimeout(rangeSearchTimeout.current)
        rangeSearchTimeout.current = null
      }
      window.removeEventListener('popstate', restoreFromLocation)
    }
  }, [form])

  function updateRecommendationUrl(values: RecommendFormValues, historyMode: HistoryMode) {
    if (historyMode === 'none') {
      return
    }

    const search = recommendSearchParams(values)
    const nextUrl = `${window.location.pathname}?${search}`
    const currentUrl = `${window.location.pathname}${window.location.search}`

    if (historyMode === 'replace' || currentUrl === nextUrl) {
      window.history.replaceState(null, '', nextUrl)
      return
    }

    window.history.pushState(null, '', nextUrl)
  }

  function scrollToPageTop() {
    window.requestAnimationFrame(() => {
      window.scrollTo({ top: 0, behavior: 'smooth' })
    })
  }

  async function runRecommend(values: RecommendFormValues, historyMode: HistoryMode = 'push', shouldScroll = true) {
    const requestId = recommendRequestId.current + 1
    recommendRequestId.current = requestId
    const normalizedValues = {
      ...values,
      beatmapInput: normalizeBeatmapInput(values.beatmapInput),
    }

    form.setValue('beatmapInput', normalizedValues.beatmapInput, { shouldValidate: false })
    setIsRunning(true)
    setError('')

    const request = buildRecommendRequest(normalizedValues)

    try {
      const turnstileToken = turnstile.enabled ? await turnstile.getToken() : ''
      const data = await recommend.mutateAsync({ ...request, turnstileToken })
      if (recommendRequestId.current !== requestId) {
        return
      }
      setResponse(data)
      writeCachedRecommendation(normalizedValues, data)
      updateRecommendationUrl(normalizedValues, historyMode)
      if (shouldScroll) {
        scrollToPageTop()
      }
    } catch (err) {
      if (recommendRequestId.current === requestId) {
        setError(err instanceof Error ? err.message : 'Request failed')
      }
    } finally {
      if (recommendRequestId.current === requestId) {
        setIsRunning(false)
        turnstile.reset()
      }
    }
  }

  function runAutoRecommend(values: RecommendFormValues) {
    if (!parseBeatmapId(values.beatmapInput)) {
      return
    }

    void runRecommend(values, 'replace', false)
  }

  function scheduleRangeRecommend(values: RecommendFormValues) {
    clearRangeSearchTimeout()

    if (!parseBeatmapId(values.beatmapInput)) {
      return
    }

    rangeSearchTimeout.current = window.setTimeout(() => {
      rangeSearchTimeout.current = null
      runAutoRecommend(values)
    }, 500)
  }

  async function resetRecommendations(values: RecommendFormValues) {
    clearRangeSearchTimeout()

    if (parseBeatmapId(values.beatmapInput)) {
      await runRecommend(values, 'replace', false)
      return
    }

    recommendRequestId.current += 1
    defaultRequestId.current += 1
    const requestId = defaultRequestId.current

    setIsRunning(false)
    setResponse(null)
    setError('')
    setDefaultResponse(null)
    setDefaultLoading(true)
    window.history.replaceState(null, '', window.location.pathname)

    try {
      const data = await fetchDefaultRecommendations()
      if (defaultRequestId.current === requestId) {
        setDefaultResponse(data)
      }
    } catch (err) {
      if (defaultRequestId.current === requestId) {
        setError(err instanceof Error ? err.message : 'Request failed')
      }
    } finally {
      if (defaultRequestId.current === requestId) {
        setDefaultLoading(false)
      }
    }
  }

  async function searchBeatmap(beatmapId: number) {
    const nextValues = { ...form.getValues(), beatmapInput: String(beatmapId) }
    form.reset(nextValues)
    await runRecommend(nextValues)
  }

  const resultBeatmaps = response?.results ?? defaultResponse?.results ?? []
  const showDefaultResults = !response && defaultResponse !== null
  const showLoadingRecommendations = defaultLoading || (!error && !response && !showDefaultResults)
  const recommendForm = (
    <div className="sticky-search-wrap">
      <RecommendForm
        form={form}
        isLoading={isLoading}
        onSubmit={runRecommend}
        onRangeChange={scheduleRangeRecommend}
        onStatusChange={(values) => {
          clearRangeSearchTimeout()
          runAutoRecommend(values)
        }}
        onPasteSearch={(values) => {
          clearRangeSearchTimeout()
          void runRecommend(values)
        }}
        onReset={(values) => {
          void resetRecommendations(values)
        }}
      />
    </div>
  )

  async function copyBeatmapId(beatmapId: number) {
    await copyText(String(beatmapId))
  }

  const resultsList = (beatmaps: BeatmapMetadata[]) => (
    <div className="result-list-wrap" aria-busy={isLoading}>
      <ResultsList
        beatmaps={beatmaps}
        onCopy={copyBeatmapId}
        onSearch={searchBeatmap}
        isLoading={isLoading}
        onPlayPreview={(beatmap: BeatmapMetadata) => audio.playPreview(beatmap)}
        activePreviewSetId={audio.activeBeatmap?.beatmapset_id ?? null}
        isPreviewPlaying={audio.isPlaying}
      />
      {isLoading ? <div className="results-loading-overlay" aria-hidden="true" /> : null}
    </div>
  )

  return (
    <main className="app-shell">
      {turnstile.widget}
      {audio.audioElement}

      <section className="results-panel">
        {error ? <p className="error-text">{error}</p> : null}

        <div className="recommend-layout">
          {response ? <BeatmapCard variant="source" beatmap={response.query.metadata} onCopy={copyBeatmapId} /> : null}
          {recommendForm}
          {response ? (
            response.results.length > 0 ? (
              resultsList(response.results)
            ) : (
              <p className="empty-results">No results found</p>
            )
          ) : showDefaultResults ? (
            resultBeatmaps.length > 0 ? (
              resultsList(resultBeatmaps)
            ) : (
              <p className="empty-results">No results found</p>
            )
          ) : showLoadingRecommendations ? (
            <p className="empty-results loading-recommendations">Loading recommendations</p>
          ) : null}
        </div>
      </section>

      {audio.activeBeatmap ? (
        <AudioPreviewBar
          beatmap={audio.activeBeatmap}
          visible={audio.visible}
          isPlaying={audio.isPlaying}
          currentTime={audio.currentTime}
          duration={audio.duration}
          volume={audio.volume}
          muted={audio.muted}
          onTogglePlay={audio.toggleActive}
          onSeek={audio.seek}
          onToggleMuted={audio.toggleMuted}
          onVolumeChange={audio.changeVolume}
          onPointerDown={audio.showTemporarily}
        />
      ) : null}

      <footer className="site-footer">
        <span>
          made by <a href="https://osu.ppy.sh/users/4881051" target="_blank" rel="noreferrer">tkn</a>
        </span>
      </footer>
    </main>
  )
}
