import { useEffect, useRef, useState } from 'react'
import { keepPreviousData, queryOptions, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useSearch } from '@tanstack/react-router'
import { AudioPreviewBar } from '../audio/AudioPreviewBar'
import { useAudioPreview } from '../audio/useAudioPreview'
import { useTurnstile } from '../turnstile/useTurnstile'
import { fetchBeatmapSummary, fetchDefaultRecommendations, recommendBeatmaps } from '../../shared/api'
import { copyText } from '../../shared/copy'
import type { BeatmapMetadata } from '../../shared/types'
import { cardCoverUrl } from '../../shared/urls'
import { buildRecommendRequest, defaultFilters, normalizeBeatmapInput, parseBeatmapId } from './filters'
import type { RecommendFormValues } from './filters'
import { RecommendForm } from './RecommendForm'
import { BeatmapCard } from './BeatmapCard'
import type { SweepDirection } from './BeatmapCard'
import { ResultsList } from './ResultsList'
import { useRecommendForm } from './useRecommendForm'
import cardStyles from './BeatmapCard.module.css'
import listStyles from './ResultsList.module.css'
import styles from './RecommendPage.module.css'

type HistoryMode = 'push' | 'replace'
type SourceSwap = {
  beatmap: BeatmapMetadata | null
  nextBeatmap: BeatmapMetadata
  direction: SweepDirection
  phase: 'in' | 'out' | 'preloading' | 'waiting'
  requestDone: boolean
}

const loadingCards = Array.from({ length: 50 }, (_, index) => index)
const recommendationStaleTime = 5 * 60_000

function recommendationOptions(values: RecommendFormValues, getToken: () => Promise<string>, resetToken: () => void) {
  return queryOptions({
    queryKey: ['recommendations', values] as const,
    queryFn: async ({ signal }) => {
      const turnstileToken = await getToken()

      try {
        return await recommendBeatmaps({ ...buildRecommendRequest(values), turnstileToken }, signal)
      } finally {
        resetToken()
      }
    },
    enabled: parseBeatmapId(values.beatmap) !== null,
    staleTime: recommendationStaleTime,
    retry: false,
    placeholderData: keepPreviousData,
  })
}

export function RecommendPage() {
  const search = useSearch({ from: '/recommendations' })
  const navigate = useNavigate({ from: '/recommendations' })
  const queryClient = useQueryClient()
  const [uiError, setUiError] = useState('')
  const [sourceSwap, setSourceSwap] = useState<SourceSwap | null>(null)
  const turnstile = useTurnstile()
  const form = useRecommendForm(search)
  const audio = useAudioPreview({ onError: setUiError })
  const rangeSearchTimeout = useRef<number | null>(null)
  const beatmapId = parseBeatmapId(search.beatmap)
  const recommend = useQuery(recommendationOptions(search, turnstile.getToken, turnstile.reset))
  const defaults = useQuery({
    queryKey: ['recommendations', 'default'],
    queryFn: ({ signal }) => fetchDefaultRecommendations(signal),
    enabled: beatmapId === null,
    staleTime: recommendationStaleTime,
  })
  const response = beatmapId === null ? null : recommend.data ?? null
  const isLoading = recommend.isFetching || defaults.isFetching
  const requestError = recommend.error ?? defaults.error
  const error = uiError || (requestError instanceof Error ? requestError.message : requestError ? 'Request failed' : '')

  useEffect(() => {
    form.reset(search)
  }, [form, search])

  useEffect(() => () => {
    if (rangeSearchTimeout.current !== null) {
      window.clearTimeout(rangeSearchTimeout.current)
    }
  }, [])

  function clearRangeSearchTimeout() {
    if (rangeSearchTimeout.current !== null) {
      window.clearTimeout(rangeSearchTimeout.current)
      rangeSearchTimeout.current = null
    }
  }

  function scrollToPageTop() {
    window.requestAnimationFrame(() => {
      window.scrollTo({ top: 0, behavior: 'smooth' })
    })
  }

  async function runRecommend(values: RecommendFormValues, historyMode: HistoryMode = 'push', shouldScroll = true) {
    const normalizedValues = {
      ...values,
      beatmap: normalizeBeatmapInput(values.beatmap),
    }
    const options = recommendationOptions(normalizedValues, turnstile.getToken, turnstile.reset)

    form.setValue('beatmap', normalizedValues.beatmap, { shouldValidate: false })
    setUiError('')
    await queryClient.invalidateQueries({ queryKey: options.queryKey, exact: true, refetchType: 'none' })
    await navigate({ search: normalizedValues, replace: historyMode === 'replace' })

    try {
      await queryClient.fetchQuery(options)
      if (shouldScroll) {
        scrollToPageTop()
      }
    } catch {
      return
    }
  }

  function runAutoRecommend(values: RecommendFormValues) {
    if (!parseBeatmapId(values.beatmap)) {
      return
    }

    void runRecommend(values, 'replace', false)
  }

  function scheduleRangeRecommend(values: RecommendFormValues) {
    clearRangeSearchTimeout()

    if (!parseBeatmapId(values.beatmap)) {
      return
    }

    rangeSearchTimeout.current = window.setTimeout(() => {
      rangeSearchTimeout.current = null
      runAutoRecommend(values)
    }, 500)
  }

  async function resetRecommendations(values: RecommendFormValues) {
    clearRangeSearchTimeout()

    if (parseBeatmapId(values.beatmap)) {
      await runRecommend(values, 'replace', false)
      return
    }

    setUiError('')
    form.reset(defaultFilters)
    await navigate({ search: defaultFilters, replace: true })
  }

  async function swapSourceBeatmap(beatmap: BeatmapMetadata, direction: SweepDirection, request: Promise<void>, preloadCover = true) {
    const currentSource = response?.query.metadata ?? null
    setSourceSwap({
      beatmap: currentSource ?? beatmap,
      nextBeatmap: beatmap,
      direction,
      phase: currentSource ? 'preloading' : 'in',
      requestDone: false,
    })

    void request.then(() => {
      setSourceSwap((swap) => {
        if (!swap || swap.nextBeatmap.beatmap_id !== beatmap.beatmap_id) {
          return swap
        }
        return swap.phase === 'waiting' ? null : { ...swap, requestDone: true }
      })
    })

    if (!currentSource) {
      await request
      return
    }

    if (preloadCover && beatmap.beatmapset_id !== null) {
      await new Promise<void>((resolve) => {
        const image = new Image()
        image.onload = () => resolve()
        image.onerror = () => resolve()
        image.src = cardCoverUrl(beatmap.beatmapset_id!)
      })
    }

    setSourceSwap((swap) => {
      if (!swap || swap.nextBeatmap.beatmap_id !== beatmap.beatmap_id) {
        return swap
      }
      return { ...swap, beatmap: swap.beatmap ?? beatmap, phase: swap.beatmap ? 'out' : 'in' }
    })
    await request
  }

  async function runManualRecommend(values: RecommendFormValues) {
    const nextBeatmapId = parseBeatmapId(values.beatmap)!
    if (response?.query.metadata.beatmap_id === nextBeatmapId) {
      await runRecommend(values)
      return
    }

    let requestDone = false
    const request = runRecommend(values).finally(() => {
      requestDone = true
    })
    const knownBeatmap = [...(response?.results ?? []), ...(defaults.data?.results ?? [])].find((beatmap) => beatmap.beatmap_id === nextBeatmapId)

    try {
      const beatmap = knownBeatmap ?? await queryClient.fetchQuery<BeatmapMetadata>({
        queryKey: ['beatmap', nextBeatmapId],
        queryFn: ({ signal }) => fetchBeatmapSummary(nextBeatmapId, signal),
        staleTime: recommendationStaleTime,
      })
      if (!requestDone) {
        await swapSourceBeatmap(beatmap, 'left', request, false)
        return
      }
    } catch {
      return request
    }
    await request
  }

  async function searchBeatmap(beatmap: BeatmapMetadata, direction: SweepDirection) {
    const nextValues = { ...form.getValues(), beatmap: String(beatmap.beatmap_id) }
    form.reset(nextValues)
    scrollToPageTop()
    await swapSourceBeatmap(beatmap, direction, runRecommend(nextValues, 'push', false))
  }

  function finishSourceSweep() {
    setSourceSwap((swap) => {
      if (!swap) {
        return null
      }
      if (swap.phase === 'out') {
        return { ...swap, beatmap: swap.nextBeatmap, phase: 'in' }
      }
      if (swap.phase === 'in') {
        return swap.requestDone ? null : { ...swap, phase: 'waiting' }
      }
      return swap
    })
  }

  const resultBeatmaps = response?.results ?? defaults.data?.results ?? []
  const showDefaultResults = !response && defaults.data !== undefined
  const showLoadingRecommendations = isLoading || (!error && !response && !showDefaultResults)
  const recommendForm = (
    <div className={styles['sticky-search-wrap']}>
      <RecommendForm
        form={form}
        isLoading={isLoading}
        onSubmit={runManualRecommend}
        onRangeChange={scheduleRangeRecommend}
        onSelectChange={(values) => {
          clearRangeSearchTimeout()
          runAutoRecommend(values)
        }}
        onPasteSearch={(values) => {
          clearRangeSearchTimeout()
          void runManualRecommend(values)
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
    <div className={styles['result-list-wrap']} aria-busy={isLoading}>
      <ResultsList
        beatmaps={beatmaps}
        onCopy={copyBeatmapId}
        onSearch={searchBeatmap}
        isLoading={isLoading}
        onPlayPreview={(beatmap: BeatmapMetadata) => audio.playPreview(beatmap)}
        activePreviewSetId={audio.activeBeatmap?.beatmapset_id ?? null}
        isPreviewPlaying={audio.isPlaying}
      />
      {isLoading ? <div className={styles['results-loading-overlay']} aria-hidden="true" /> : null}
    </div>
  )
  const sourceBeatmap = sourceSwap ? sourceSwap.beatmap : response?.query.metadata
  const sourceSweepPhase = sourceSwap?.phase === 'in' || sourceSwap?.phase === 'out' ? sourceSwap.phase : undefined
  const showSourcePlaceholder = !sourceBeatmap && isLoading && parseBeatmapId(form.getValues('beatmap')) !== null

  return (
    <main className={styles['app-shell']}>
      {turnstile.widget}
      {audio.audioElement}

      <section className={styles['results-panel']}>
        <div className={styles['recommend-layout']}>
          {sourceBeatmap ? (
            <BeatmapCard
              variant="source"
              beatmap={sourceBeatmap}
              onCopy={copyBeatmapId}
              sweepDirection={sourceSweepPhase ? sourceSwap?.direction : undefined}
              sweepPhase={sourceSweepPhase}
              onSweepEnd={finishSourceSweep}
            />
          ) : showSourcePlaceholder ? <div className={`${cardStyles['beatmap-card']} ${cardStyles['source-card']} ${cardStyles['source-card-placeholder']} ${styles['source-card-placeholder']}`} data-card-variant="source" aria-hidden="true" /> : null}
          {recommendForm}
          {response ? (
            response.results.length > 0 ? (
              resultsList(response.results)
            ) : (
              <p className={styles['empty-results']}>No results found</p>
            )
          ) : showDefaultResults ? (
            resultBeatmaps.length > 0 ? (
              resultsList(resultBeatmaps)
            ) : (
              <p className={styles['empty-results']}>No results found</p>
            )
          ) : showLoadingRecommendations ? (
            <div className={`${listStyles['result-list']} ${styles['loading-result-list']}`} role="status" aria-label="Loading recommendations">
              {loadingCards.map((index) => (
                <div className={`${cardStyles['beatmap-card']} ${styles['loading-result-card']}`} key={index} aria-hidden="true">
                  <div className={styles['loading-result-cover']} />
                  <div className={styles['loading-result-content']}>
                    <div className={styles['loading-result-copy']}>
                      <span className={`${styles['loading-result-line']} ${styles['loading-result-title']}`} />
                      <span className={`${styles['loading-result-line']} ${styles['loading-result-artist']}`} />
                      <span className={`${styles['loading-result-line']} ${styles['loading-result-version']}`} />
                    </div>
                    <div className={styles['loading-result-stats']}>
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </section>

      {audio.activeBeatmap ? (
        <AudioPreviewBar
          beatmap={audio.activeBeatmap}
          audioRef={audio.audioRef}
          visible={audio.visible}
          isPlaying={audio.isPlaying}
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

      <footer className={styles['site-footer']}>
        <span>
          made by <a href="https://osu.ppy.sh/users/4881051" target="_blank" rel="noreferrer">tkn</a>
        </span>
      </footer>
    </main>
  )
}
