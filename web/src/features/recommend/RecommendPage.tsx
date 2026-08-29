import { useEffect, useState } from 'react'
import type { CSSProperties } from 'react'
import { keepPreviousData, queryOptions, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useSearch } from '@tanstack/react-router'
import { AudioPreviewBar } from '../audio/AudioPreviewBar'
import { useAudioPreview } from '../audio/useAudioPreview'
import { fetchBeatmapSummary, fetchDefaultRecommendations, recommendBeatmaps } from '../../shared/api'
import { copyText } from '../../shared/copy'
import type { BeatmapMetadata } from '../../shared/types'
import { cardCoverUrl, coverUrl } from '../../shared/urls'
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

const loadingCards = Array.from({ length: 100 }, (_, index) => index)
const staleTime = 5 * 60_000
const coverBatchCount = 16
const coverBatchTimeout = 500

const coverCache = new Map<number, Promise<void>>()

function preloadCover(beatmapsetId: number) {
  let request = coverCache.get(beatmapsetId)
  if (!request) {
    const image = new Image()
    image.src = coverUrl(beatmapsetId)
    request = image.decode().catch(() => new Promise<void>((resolve) => {
      image.onload = () => resolve()
      image.onerror = () => resolve()
    }))
    coverCache.set(beatmapsetId, request)
  }
  return request
}

function useCoversReady(beatmaps: BeatmapMetadata[]) {
  const [batch, setBatch] = useState<{ key: string; ready: boolean }>({ key: '', ready: false })
  const key = beatmaps
    .slice(0, coverBatchCount)
    .flatMap((beatmap) => (beatmap.beatmapset_id === null ? [] : [String(beatmap.beatmapset_id)]))
    .join('|')

  useEffect(() => {
    if (key === '') {
      return
    }

    let cancelled = false
    const reveal = () => {
      if (!cancelled) {
        setBatch({ key, ready: true })
      }
    }
    const timer = setTimeout(reveal, coverBatchTimeout)
    Promise.all(key.split('|').map((id) => preloadCover(Number(id)))).then(() => {
      clearTimeout(timer)
      reveal()
    })

    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [key])

  return key === '' || (batch.key === key && batch.ready)
}

function recommendationOptions(values: RecommendFormValues) {
  return queryOptions({
    queryKey: ['recommendations', values] as const,
    queryFn: ({ signal }) => recommendBeatmaps(buildRecommendRequest(values), signal),
    enabled: parseBeatmapId(values.beatmap) !== null,
    staleTime,
    retry: false,
    placeholderData: keepPreviousData,
  })
}

export function RecommendPage() {
  const search = useSearch({ from: '/recommendations' })
  const navigate = useNavigate({ from: '/recommendations' })
  const queryClient = useQueryClient()
  const [sourceSwap, setSourceSwap] = useState<SourceSwap | null>(null)
  const form = useRecommendForm(search, runManualRecommend)
  const audio = useAudioPreview({ onError: console.error })
  const beatmapId = parseBeatmapId(search.beatmap)
  const recommend = useQuery(recommendationOptions(search))
  const defaults = useQuery({
    queryKey: ['recommendations', 'default'],
    queryFn: ({ signal }) => fetchDefaultRecommendations(signal),
    enabled: beatmapId === null,
    staleTime,
  })
  const response = beatmapId === null ? null : recommend.data ?? null
  const isLoading = recommend.isFetching || defaults.isFetching
  const requestError = recommend.error ?? defaults.error

  useEffect(() => {
    form.reset(search)
  }, [form, search])

  useEffect(() => {
    if (requestError) {
      console.error(requestError)
    }
  }, [requestError])

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
    const options = recommendationOptions(normalizedValues)

    form.setFieldValue('beatmap', normalizedValues.beatmap, { dontValidate: true })
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

  async function resetRecommendations(values: RecommendFormValues) {
    if (parseBeatmapId(values.beatmap)) {
      await runRecommend(values, 'replace', false)
      return
    }

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
        staleTime,
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
    const nextValues = { ...form.state.values, beatmap: String(beatmap.beatmap_id) }
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

  const defaultResponse = beatmapId === null ? defaults.data : undefined
  const resultBeatmaps = response?.results ?? defaultResponse?.results ?? []
  const hasResults = resultBeatmaps.length > 0
  const coversReady = useCoversReady(resultBeatmaps)
  const showDefaultResults = defaultResponse !== undefined
  const showLoadingRecommendations = isLoading || (!requestError && !response && !showDefaultResults) || (hasResults && !coversReady)
  const recommendForm = (
    <div className={styles['sticky-search-wrap']}>
      <RecommendForm
        form={form}
        isLoading={isLoading}
        onRangeChange={runAutoRecommend}
        onSelectChange={runAutoRecommend}
        onPasteSearch={(values) => {
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
  const showSourcePlaceholder = !sourceBeatmap && isLoading && parseBeatmapId(form.getFieldValue('beatmap')) !== null

  return (
    <main className={styles['app-shell']}>
      {audio.audioElement}

      <section className={styles['results-panel']}>
        <div className={styles['recommend-layout']}>
          {sourceBeatmap ? (
            <BeatmapCard
              variant="source"
              beatmap={sourceBeatmap}
              onCopy={copyBeatmapId}
              onPlayPreview={(beatmap: BeatmapMetadata) => audio.playPreview(beatmap)}
              activePreviewSetId={audio.activeBeatmap?.beatmapset_id ?? null}
              isPreviewPlaying={audio.isPlaying}
              sweepDirection={sourceSweepPhase ? sourceSwap?.direction : undefined}
              sweepPhase={sourceSweepPhase}
              onSweepEnd={finishSourceSweep}
            />
          ) : showSourcePlaceholder ? <div className={`${cardStyles['beatmap-card']} ${cardStyles['source-card']} ${cardStyles['source-card-placeholder']} ${styles['source-card-placeholder']}`} data-card-variant="source" aria-hidden="true" /> : null}
          {recommendForm}
          {hasResults && coversReady ? (
            resultsList(resultBeatmaps)
          ) : (response !== null || showDefaultResults) && !hasResults ? (
            <p className={styles['empty-results']}>No results found</p>
          ) : showLoadingRecommendations ? (
            <div className={`${listStyles['result-list']} ${styles['loading-result-list']}`} role="status" aria-label="Loading recommendations">
              {loadingCards.map((index) => (
                <div className={`${cardStyles['beatmap-card']} ${styles['loading-result-card']}`} key={index} style={{ '--i': index } as CSSProperties} aria-hidden="true">
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
