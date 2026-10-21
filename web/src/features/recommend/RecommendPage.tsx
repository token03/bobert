import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { Tabs } from '@base-ui/react/tabs'
import { ClipboardText, CursorText, DiceFive, GithubLogo, MagnifyingGlass, WarningCircle } from '@phosphor-icons/react'
import { queryOptions, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useSearch } from '@tanstack/react-router'
import { AudioPreviewBar } from '../audio/AudioPreviewBar'
import { useAudioPreview } from '../audio/useAudioPreview'
import { fetchDefaultRecommendations, recommendBeatmaps, UnavailableBeatmapError } from '../../shared/api'
import { parseBeatmapIds } from '../../shared/beatmapIds'
import { copyText } from '../../shared/copy'
import type { BeatmapMetadata } from '../../shared/types'
import { cardCoverUrl } from '../../shared/urls'
import { modes, modeLabels } from '../../shared/types'
import { buildRecommendRequest, clearFilters, defaultSearch } from './filters'
import type { RecommendSearch } from './filters'
import { RecommendForm } from './RecommendForm'
import { SourceBeatmapCard } from './BeatmapCard'
import type { SweepDirection } from './BeatmapCard'
import { ResultsList } from './ResultsList'
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

function recommendationOptions(search: RecommendSearch) {
  return queryOptions({
    queryKey: ['recommendations', search] as const,
    queryFn: ({ signal }) => recommendBeatmaps(buildRecommendRequest(search), signal),
    enabled: parseBeatmapIds(search.beatmap) !== null,
    staleTime,
    retry: false,
    placeholderData: (previous, query) => (query?.queryKey[1] as RecommendSearch | undefined)?.mode === search.mode ? previous : undefined,
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
  const [shownSource, setShownSource] = useState<BeatmapMetadata | null>(null)
  const [leavingSource, setLeavingSource] = useState<BeatmapMetadata | null>(null)
  const audio = useAudioPreview({ onError: console.error })
  const beatmapIds = parseBeatmapIds(search.beatmap)
  const recommend = useQuery(recommendationOptions(search))
  const defaults = useQuery({
    queryKey: ['recommendations', 'default', search.mode],
    queryFn: ({ signal }) => fetchDefaultRecommendations(search.mode, signal),
    enabled: beatmapIds === null,
    staleTime,
  })
  const response = beatmapIds === null ? null : recommend.data ?? null
  const sourceBeatmaps = response?.sources.map((source) => source.metadata) ?? []
  const selectedSource = sourceBeatmaps.find((beatmap) => beatmap.beatmap_id === sourceView.beatmapId) ?? sourceBeatmaps[0] ?? null
  const isLoading = recommend.isFetching || defaults.isFetching
  const requestError = beatmapIds === null ? defaults.error : recommend.error
  const unavailableBeatmapId = requestError instanceof UnavailableBeatmapError ? requestError.beatmapId : null

  useEffect(() => {
    const inferredMode = response?.sources[0]?.metadata.mode
    if (!response || recommend.isPlaceholderData || !inferredMode || inferredMode === search.mode) return
    const values = { ...search, mode: inferredMode }
    queryClient.setQueryData(recommendationOptions(values).queryKey, response)
    void navigate({ search: values, replace: true })
  }, [response, recommend.isPlaceholderData, search, queryClient, navigate])

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

  function exitResults() {
    if (hasResults && !isLoading) {
      setResultsHeight(resultsRef.current?.getBoundingClientRect().height ?? 0)
      setOutgoingResults(resultBeatmaps)
    }
  }

  async function runRecommend(values: RecommendSearch, historyMode: HistoryMode = 'push', shouldScroll = true) {
    exitResults()

    const normalizedValues = {
      ...values,
      beatmap: parseBeatmapIds(values.beatmap)?.join(',') ?? '',
    }
    const options = recommendationOptions(normalizedValues)

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

  function runAutoRecommend(values: RecommendSearch) {
    if (!parseBeatmapIds(values.beatmap)) {
      void navigate({ search: values, replace: true })
      return
    }

    void runRecommend(values, 'replace', false)
  }

  async function resetRecommendations(values: RecommendSearch) {
    if (parseBeatmapIds(values.beatmap)) {
      await runRecommend(values, 'replace', false)
      return
    }

    await navigate({ search: { ...defaultSearch, mode: values.mode }, replace: true })
  }

  async function updateBeatmaps(values: RecommendSearch) {
    setSourceSwap(null)
    if (!parseBeatmapIds(values.beatmap)) {
      exitResults()
      await navigate({ search: values })
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

  async function runManualRecommend(values: RecommendSearch) {
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

    void lookupBeatmapSet(nextBeatmapId, values.mode).then((setId) => {
      if (setId !== null) new Image().src = cardCoverUrl(setId)
    }).catch(() => {})

    const data = await request
    const nextSource = data?.sources.find((source) => source.beatmap_id === nextBeatmapId)?.metadata ?? data?.sources[0]?.metadata ?? null
    if (nextSource && nextSource.beatmap_id !== selectedSource?.beatmap_id) {
      await swapSourceBeatmap(nextSource, 'left', Promise.resolve(), true)
    }
  }

  async function searchBeatmap(beatmap: BeatmapMetadata, direction: SweepDirection) {
    const nextValues = { ...search, beatmap: String(beatmap.beatmap_id) }
    if (beatmap.title) {
      queryClient.setQueryData<string | null>(['beatmap-title', search.mode, beatmap.beatmap_id], (title) => title ?? beatmap.title)
    }
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
  const showDefaultResults = defaultResponse !== undefined
  const showLoadingRecommendations = isLoading || (!requestError && !response && !showDefaultResults)
  const recommendForm = (
    <div className={styles['sticky-search-wrap']}>
      <RecommendForm
        key={search.mode}
        search={search}
        sources={sourceBeatmaps}
        unavailableBeatmapId={unavailableBeatmapId}
        isLoading={isLoading}
        onFiltersChange={runAutoRecommend}
        onBeatmapsChange={(values) => {
          void updateBeatmaps(values)
        }}
        onSubmit={(values) => {
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
  if (sourceBeatmap !== shownSource) {
    setShownSource(sourceBeatmap)
    setLeavingSource(!sourceBeatmap && beatmapIds === null && !isRolling ? shownSource : null)
  }
  const sourceSweepPhase = sourceSwap?.phase === 'in' || sourceSwap?.phase === 'out' ? sourceSwap.phase : undefined
  const showSourcePlaceholder = !sourceBeatmap && isLoading && beatmapIds !== null

  return (
    <main className={styles['app-shell']}>
      {audio.audioElement}

      <section className={styles['results-panel']}>
        <div className={styles['recommend-layout']}>
          {leavingSource && beatmapIds === null ? (
            <SourceBeatmapCard
              beatmap={leavingSource}
              onCopy={copyBeatmapId}
              onPlayPreview={(beatmap: BeatmapMetadata) => audio.playPreview(beatmap)}
              activePreviewSetId={audio.activeBeatmap?.beatmapset_id ?? null}
              isPreviewPlaying={audio.isPlaying}
              sweepDirection="left"
              sweepPhase="out"
              onSweepEnd={() => {
                setLeavingSource(null)
              }}
            />
          ) : !isRolling && sourceSwap && sourceBeatmap ? (
            <SourceBeatmapCard
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
          ) : !isRolling && showSourcePlaceholder ? (
            <div className={`${cardStyles['beatmap-card']} ${cardStyles['source-card']} ${styles['source-card-placeholder']}`} data-card-variant="source" aria-hidden="true">
              <div className={`${cardStyles['cover-preview']} ${cardStyles['source-cover']} ${styles['loading-result-cover']}`} />
              <div className={`${cardStyles['beatmap-card-content']} ${cardStyles['source-content']}`}>
                <div className={`${cardStyles['source-copy']} ${styles['loading-result-copy']}`}>
                  <span className={`${styles['loading-result-line']} ${styles['loading-result-title']}`} />
                  <span className={`${styles['loading-result-line']} ${styles['loading-result-artist']}`} />
                  <span className={`${styles['loading-result-line']} ${styles['loading-result-version']}`} />
                </div>
                <div className={`${cardStyles['source-meta']} ${styles['loading-result-meta']}`}>
                  <span className={`${styles['loading-result-line']} ${styles['loading-result-creator']}`} />
                  <span className={`${styles['loading-result-line']} ${styles['loading-result-status']}`} />
                </div>
                <div className={cardStyles['source-meta-separator']} />
                <div className={cardStyles['stat-strip']}>
                  <div className={`${cardStyles['stat-row']} ${cardStyles['stat-row-main']}`}>
                    {Array.from({ length: 3 }, (_, stat) => (
                      <div key={stat} className={cardStyles['stat-item']}>
                        <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-icon']}`} />
                        <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-main']}`} />
                      </div>
                    ))}
                  </div>
                  <div className={cardStyles['stat-side']}>
                    <div className={cardStyles['source-stat-separator']} />
                    <div className={`${cardStyles['stat-row']} ${cardStyles['stat-row-sub']}`}>
                      {Array.from({ length: 4 }, (_, stat) => (
                        <div key={stat} className={cardStyles['stat-item']}>
                          <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-label']}`} />
                          <span className={`${styles['loading-result-stat']} ${styles['loading-result-stat-main']}`} />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : isRolling || beatmapIds === null ? (
            <header className={styles['intro']}>
              <div className={styles['intro-copy']}>
                <div className={styles['intro-lead']}>
                  <span>to find similar maps:</span>
                </div>
                <span className={styles['intro-steps']}>
                  <span><span className={styles['intro-action']}><CursorText aria-hidden="true" /> search</span> for a map</span>
                  <span><span className={styles['intro-action']}><ClipboardText aria-hidden="true" /> paste</span> a link/id anywhere</span>
                  <span><span className={styles['intro-action']}><MagnifyingGlass aria-hidden="true" /> search similar</span> on a map below</span>
                </span>
                <span className={styles['intro-outro']}>have fun!</span>
              </div>
              <div className={styles['intro-actions']}>
                <div className={styles['mode-controls']} role="group" aria-label="Game mode">
                  {modes.map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      aria-pressed={search.mode === mode}
                      onClick={() => {
                        setSourceSwap(null)
                        setOutgoingResults(null)
                        void navigate({ search: { ...defaultSearch, mode } })
                      }}
                    >{modeLabels[mode]}</button>
                  ))}
                </div>
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
            ) : requestError ? (
              <section className={styles['empty-results']} role="alert">
                <WarningCircle aria-hidden="true" />
                <h2>{unavailableBeatmapId === null ? 'Couldn’t load recommendations' : 'Beatmap unavailable'}</h2>
                <p>{unavailableBeatmapId === null ? requestError.message : `Beatmap #${unavailableBeatmapId} couldn’t be downloaded. Try another ID or link.`}</p>
                <div className={styles['state-actions']}>
                  <button type="button" onClick={() => { void (beatmapIds === null ? defaults.refetch() : recommend.refetch()) }}>Try again</button>
                  <button type="button" onClick={() => { void updateBeatmaps({ ...search, beatmap: unavailableBeatmapId === null ? '' : (beatmapIds ?? []).filter((id) => id !== unavailableBeatmapId).join(',') }) }}>{unavailableBeatmapId === null ? 'Clear search' : 'Remove map'}</button>
                </div>
              </section>
            ) : hasResults ? (
              resultsList(resultBeatmaps)
            ) : (response !== null || showDefaultResults) ? (
              <section className={styles['empty-results']} role="status">
                <MagnifyingGlass aria-hidden="true" />
                <h2>No matching beatmaps</h2>
                <p>Try widening your filters or searching a different map.</p>
                <div className={styles['state-actions']}>
                  <button type="button" onClick={() => { void resetRecommendations(clearFilters(search)) }}>Clear filters</button>
                  <button type="button" onClick={() => { void updateBeatmaps({ ...defaultSearch, mode: search.mode }) }}>Choose another map</button>
                </div>
              </section>
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
    <SourceBeatmapCard
      key={beatmap.beatmap_id}
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
