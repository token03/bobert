import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { Tabs } from '@base-ui/react/tabs'
import { DiceFive, GithubLogo, MagnifyingGlass } from '@phosphor-icons/react'
import { keepPreviousData, queryOptions, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useSearch } from '@tanstack/react-router'
import { AudioPreviewBar } from '../audio/AudioPreviewBar'
import { useAudioPreview } from '../audio/useAudioPreview'
import { fetchDefaultRecommendations, recommendBeatmaps } from '../../shared/api'
import { parseBeatmapIds } from '../../shared/beatmapIds'
import { copyText } from '../../shared/copy'
import type { BeatmapMetadata } from '../../shared/types'
import { cardCoverUrl, coverUrl } from '../../shared/urls'
import { buildRecommendRequest, defaultFilters, normalizeBeatmapInput } from './filters'
import type { RecommendFormValues } from './filters'
import { RecommendForm } from './RecommendForm'
import { BeatmapCard } from './BeatmapCard'
import type { SweepDirection } from './BeatmapCard'
import { ResultsList } from './ResultsList'
import { useRecommendForm } from './useRecommendForm'
import { lookupBeatmapSet } from '../search/searchClient'
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
    enabled: parseBeatmapIds(values.beatmap) !== null,
    staleTime,
    retry: false,
    placeholderData: keepPreviousData,
  })
}

export function RecommendPage() {
  const search = useSearch({ from: '/recommendations' })
  const navigate = useNavigate({ from: '/recommendations' })
  const queryClient = useQueryClient()
  const resultsRef = useRef<HTMLDivElement>(null)
  const [outgoingResults, setOutgoingResults] = useState<BeatmapMetadata[] | null>(null)
  const [resultsHeight, setResultsHeight] = useState(0)
  const [isRolling, setIsRolling] = useState(false)
  const [sourceSwap, setSourceSwap] = useState<SourceSwap | null>(null)
  const [sourceView, setSourceView] = useState<{ beatmapId: number | null; direction: SweepDirection; animate: boolean }>({ beatmapId: null, direction: 'left', animate: false })
  const form = useRecommendForm(search, runManualRecommend)
  const audio = useAudioPreview({ onError: console.error })
  const beatmapIds = parseBeatmapIds(search.beatmap)
  const recommend = useQuery(recommendationOptions(search))
  const defaults = useQuery({
    queryKey: ['recommendations', 'default'],
    queryFn: ({ signal }) => fetchDefaultRecommendations(signal),
    enabled: beatmapIds === null,
    staleTime,
  })
  const response = beatmapIds === null ? null : recommend.data ?? null
  const sourceBeatmaps = response?.sources.map((source) => source.metadata) ?? []
  const selectedSource = sourceBeatmaps.find((beatmap) => beatmap.beatmap_id === sourceView.beatmapId) ?? sourceBeatmaps[0] ?? null
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
    if (hasResults && coversReady && !isLoading) {
      setResultsHeight(resultsRef.current?.getBoundingClientRect().height ?? 0)
      setOutgoingResults(resultBeatmaps)
    }

    const normalizedValues = {
      ...values,
      beatmap: normalizeBeatmapInput(values.beatmap),
    }
    const options = recommendationOptions(normalizedValues)

    form.setFieldValue('beatmap', normalizedValues.beatmap, { dontValidate: true })
    await queryClient.invalidateQueries({ queryKey: options.queryKey, exact: true, refetchType: 'none' })
    await navigate({ search: normalizedValues, replace: historyMode === 'replace' })

    try {
      const data = await queryClient.fetchQuery(options)
      if (shouldScroll) {
        scrollToPageTop()
      }
      return data
    } catch {
      return null
    }
  }

  function runAutoRecommend(values: RecommendFormValues) {
    if (!parseBeatmapIds(values.beatmap)) {
      return
    }

    void runRecommend(values, 'replace', false)
  }

  async function resetRecommendations(values: RecommendFormValues) {
    if (parseBeatmapIds(values.beatmap)) {
      await runRecommend(values, 'replace', false)
      return
    }

    form.reset(defaultFilters)
    await navigate({ search: defaultFilters, replace: true })
  }

  async function updateBeatmaps(values: RecommendFormValues) {
    setSourceSwap(null)
    if (!parseBeatmapIds(values.beatmap)) {
      return
    }
    await runRecommend(values, 'push', false)
  }

  async function swapSourceBeatmap(beatmap: BeatmapMetadata, direction: SweepDirection, request: Promise<unknown>, preloadCover = true) {
    const currentSource = selectedSource
    setSourceView((view) => ({ ...view, beatmapId: null, animate: false }))
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
    const nextBeatmapIds = parseBeatmapIds(values.beatmap)!
    if (nextBeatmapIds.length !== 1) {
      await runRecommend(values)
      return
    }
    const nextBeatmapId = nextBeatmapIds[0]
    if (response?.sources.length === 1 && response.sources[0].beatmap_id === nextBeatmapId) {
      await runRecommend(values)
      return
    }

    const request = runRecommend(values)
    const knownBeatmap = [...(response?.results ?? []), ...(defaults.data?.results ?? [])].find((beatmap) => beatmap.beatmap_id === nextBeatmapId)
    if (knownBeatmap) {
      await swapSourceBeatmap(knownBeatmap, 'left', request, false)
      return
    }

    void lookupBeatmapSet(nextBeatmapId).then((setId) => {
      if (setId !== null) new Image().src = cardCoverUrl(setId)
    }).catch(() => {})

    const data = await request
    const nextSource = data?.sources.find((source) => source.beatmap_id === nextBeatmapId)?.metadata ?? data?.sources[0]?.metadata ?? null
    if (nextSource && nextSource.beatmap_id !== selectedSource?.beatmap_id) {
      await swapSourceBeatmap(nextSource, 'left', Promise.resolve(), true)
    }
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

  const defaultResponse = beatmapIds === null ? defaults.data : undefined
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
        onBeatmapsChange={(values) => {
          void updateBeatmaps(values)
        }}
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
    <ResultsList
      beatmaps={beatmaps}
      onCopy={copyBeatmapId}
      onSearch={searchBeatmap}
      isLoading={isLoading}
      onPlayPreview={(beatmap: BeatmapMetadata) => audio.playPreview(beatmap)}
      activePreviewSetId={audio.activeBeatmap?.beatmapset_id ?? null}
      isPreviewPlaying={audio.isPlaying}
    />
  )
  const sourceBeatmap = sourceSwap ? sourceSwap.beatmap : selectedSource
  const sourceSweepPhase = sourceSwap?.phase === 'in' || sourceSwap?.phase === 'out' ? sourceSwap.phase : undefined
  const showSourcePlaceholder = !sourceBeatmap && isLoading && parseBeatmapIds(form.getFieldValue('beatmap')) !== null

  return (
    <main className={styles['app-shell']}>
      {audio.audioElement}

      <section className={styles['results-panel']}>
        <div className={styles['recommend-layout']}>
          {!isRolling && sourceSwap && sourceBeatmap ? (
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
          ) : !isRolling && sourceBeatmap ? (
            <SourceBeatmapPager
              beatmaps={sourceBeatmaps}
              beatmap={sourceBeatmap}
              direction={sourceView.direction}
              animate={sourceView.animate}
              onSelect={(beatmapId, direction) => setSourceView({ beatmapId, direction, animate: true })}
              onCopy={copyBeatmapId}
              onPlayPreview={(beatmap) => audio.playPreview(beatmap)}
              activePreviewSetId={audio.activeBeatmap?.beatmapset_id ?? null}
              isPreviewPlaying={audio.isPlaying}
            />
          ) : !isRolling && showSourcePlaceholder ? <div className={`${cardStyles['beatmap-card']} ${cardStyles['source-card']} ${cardStyles['source-card-placeholder']} ${styles['source-card-placeholder']}`} data-card-variant="source" aria-hidden="true" /> : isRolling || beatmapIds === null ? (
            <header className={styles['intro']}>
              <h1>bobert</h1>
              <p>to find similar beatmaps, enter a beatmap id or use <span className={styles['intro-action']}><MagnifyingGlass aria-hidden="true" /> search similar</span> on a map below. have fun!</p>
              <div className={styles['intro-actions']}>
                <button
                  type="button"
                  disabled={isLoading || isRolling || !hasResults}
                  onClick={() => {
                    setIsRolling(true)
                    const beatmap = resultBeatmaps[Math.floor(Math.random() * resultBeatmaps.length)]
                    void searchBeatmap(beatmap, 'left')
                  }}
                >
                  <DiceFive
                    className={styles['pick-dice']}
                    data-rolling={isRolling || undefined}
                    aria-hidden="true"
                    onAnimationEnd={() => setIsRolling(false)}
                  />
                  pick for me
                </button>
                <a href="https://github.com/token03/bobert" target="_blank" rel="noreferrer" aria-label="view source on github" title="view source on github">
                  <GithubLogo aria-hidden="true" />
                  view source
                </a>
              </div>
            </header>
          ) : null}
          {recommendForm}
          <div
            ref={resultsRef}
            className={`${styles['result-list-wrap']}${outgoingResults ? ` ${styles['results-exiting']}` : ''}`}
            aria-busy={showLoadingRecommendations || outgoingResults !== null}
            inert={outgoingResults !== null}
            style={outgoingResults || showLoadingRecommendations ? { minHeight: resultsHeight } : undefined}
            onAnimationEnd={(event) => {
              if (event.target === event.currentTarget && outgoingResults) {
                setOutgoingResults(null)
              }
            }}
          >
            {outgoingResults ? (
              resultsList(outgoingResults)
            ) : showLoadingRecommendations ? (
              <div className={`${listStyles['result-list']} ${styles['loading-result-list']}`} role="status" aria-label="Loading recommendations">
                {loadingCards.map((index) => (
                  <div className={`${cardStyles['beatmap-card']} ${cardStyles['beatmap-card-result']} ${cardStyles['beatmap-row']} ${styles['loading-result-card']}`} key={index} style={{ '--i': index } as CSSProperties} aria-hidden="true">
                    <div className={`${cardStyles['cover-preview']} ${styles['loading-result-cover']}`} />
                    <div className={`${cardStyles['beatmap-card-content']} ${cardStyles['map-content']}`}>
                      <div className={cardStyles['map-main']}>
                        <div className={cardStyles['result-summary']}>
                          <div className={`${cardStyles['result-copy']} ${styles['loading-result-copy']}`}>
                            <span className={`${styles['loading-result-line']} ${styles['loading-result-title']}`} />
                            <span className={`${styles['loading-result-line']} ${styles['loading-result-artist']}`} />
                            <span className={`${styles['loading-result-line']} ${styles['loading-result-version']}`} />
                          </div>
                          <div className={`${cardStyles['result-meta']} ${styles['loading-result-meta']}`}>
                            <span className={`${styles['loading-result-line']} ${styles['loading-result-creator']}`} />
                            <span className={`${styles['loading-result-line']} ${styles['loading-result-status']}`} />
                          </div>
                        </div>
                      </div>
                      <div className={cardStyles['stat-strip']}>
                        <div className={cardStyles['result-stat-separator']} aria-hidden="true" />
                        <div className={`${cardStyles['stat-row']} ${cardStyles['stat-row-main']}`}>
                          {Array.from({ length: 3 }, (_, stat) => (
                            <div key={stat} className={cardStyles['stat-item']}>
                              <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-icon']}`} />
                              <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-main']}`} />
                            </div>
                          ))}
                        </div>
                        <div className={cardStyles['stat-side']}>
                          <div className={`${cardStyles['stat-row']} ${cardStyles['stat-row-sub']}`}>
                            {Array.from({ length: 4 }, (_, stat) => (
                              <div key={stat} className={cardStyles['stat-item']}>
                                <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-label']}`} />
                                <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-sub']}`} />
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : hasResults ? (
              resultsList(resultBeatmaps)
            ) : requestError ? (
              <p className={styles['empty-results']} role="alert">Could not load recommendations. Please try again.</p>
            ) : (response !== null || showDefaultResults) ? (
              <p className={styles['empty-results']}>No results found</p>
            ) : null}
          </div>
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

type SourceBeatmapPagerProps = {
  beatmaps: BeatmapMetadata[]
  beatmap: BeatmapMetadata
  direction: SweepDirection
  animate: boolean
  onSelect: (beatmapId: number, direction: SweepDirection) => void
  onCopy: (beatmapId: number) => Promise<void>
  onPlayPreview: (beatmap: BeatmapMetadata) => Promise<void>
  activePreviewSetId: number | null
  isPreviewPlaying: boolean
}

function SourceBeatmapPager({ beatmaps, beatmap, direction, animate, onSelect, onCopy, onPlayPreview, activePreviewSetId, isPreviewPlaying }: SourceBeatmapPagerProps) {
  const card = (
    <BeatmapCard
      key={beatmap.beatmap_id}
      variant="source"
      beatmap={beatmap}
      onCopy={onCopy}
      onPlayPreview={onPlayPreview}
      activePreviewSetId={activePreviewSetId}
      isPreviewPlaying={isPreviewPlaying}
      sweepDirection={direction}
      sweepPhase={animate && beatmaps.length > 1 ? 'in' : undefined}
    />
  )

  if (beatmaps.length < 2) {
    return card
  }

  const selectedIndex = beatmaps.findIndex((source) => source.beatmap_id === beatmap.beatmap_id)
  return (
    <Tabs.Root className={styles['source-pager']} value={beatmap.beatmap_id} onValueChange={(beatmapId) => {
      const nextIndex = beatmaps.findIndex((source) => source.beatmap_id === beatmapId)
      onSelect(beatmapId as number, nextIndex < selectedIndex ? 'left' : 'right')
    }}>
      <Tabs.Panel className={styles['source-panel']} value={beatmap.beatmap_id}>{card}</Tabs.Panel>
      <Tabs.List className={styles['source-tabs']} activateOnFocus aria-label="Source beatmaps">
        {beatmaps.map((source, index) => (
          <Tabs.Tab
            className={styles['source-tab']}
            key={source.beatmap_id}
            value={source.beatmap_id}
            aria-label={`Source ${index + 1} of ${beatmaps.length}: ${source.title ?? `Beatmap ${source.beatmap_id}`}`}
            title={source.title ?? `Beatmap ${source.beatmap_id}`}
          />
        ))}
      </Tabs.List>
    </Tabs.Root>
  )
}
