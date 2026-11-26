import { useEffect, useRef, useState } from 'react'
import { Tabs } from '@base-ui/react/tabs'
import { ClipboardText, CursorText, GithubLogo, MagnifyingGlass, SignIn, SignOut, WarningCircle } from '@phosphor-icons/react'
import { queryOptions, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate, useSearch } from '@tanstack/react-router'
import { AudioPreviewBar } from '../audio/AudioPreviewBar'
import { useAudioPreview } from '../audio/useAudioPreview'
import { fetchDefaultRecommendations, fetchMyGroups, fetchSession, logout, recommendBeatmaps, UnavailableBeatmapError } from '../../shared/api'
import { parseBeatmapIds } from '../../shared/beatmapIds'
import { copyText } from '../../shared/copy'
import type { BeatmapMetadata } from '../../shared/types'
import { avatarUrl, userUrl } from '../../shared/urls'
import { modes, modeLabels } from '../../shared/types'
import { buildRecommendRequest, clearFilters, defaultSearch } from './filters'
import type { RecommendSearch } from './filters'
import { PlayGroupTabs } from './PlayGroupTabs'
import { RecommendForm } from './RecommendForm'
import { SourceBeatmapCard } from './BeatmapCard'
import { LoadingResultsList, ResultsList } from './ResultsList'
import cardStyles from './BeatmapCard.module.css'
import listStyles from './ResultsList.module.css'
import styles from './RecommendPage.module.css'

type HistoryMode = 'push' | 'replace'

const staleTime = 5 * 60_000

function recommendationOptions(search: RecommendSearch) {
  return queryOptions({
    queryKey: ['recommendations', search] as const,
    queryFn: ({ signal }) => recommendBeatmaps(buildRecommendRequest(search, parseBeatmapIds(search.beatmap)!), signal),
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
  const [selectedSourceId, setSelectedSourceId] = useState<number | null>(null)
  const audio = useAudioPreview({ onError: console.error })
  const beatmapIds = parseBeatmapIds(search.beatmap)
  const recommend = useQuery(recommendationOptions(search))
  const session = useQuery({ queryKey: ['session'], queryFn: fetchSession, staleTime: Infinity })
  const groups = useQuery({
    queryKey: ['groups', search.mode],
    queryFn: ({ signal }) => fetchMyGroups(search.mode, signal),
    enabled: Boolean(session.data),
    staleTime,
    retry: false,
  })
  const playSources = new Map(groups.data?.sources.map((beatmap) => [beatmap.beatmap_id, beatmap]))
  const playGroups = (groups.data?.groups ?? []).map(({ label, beatmap_ids: ids }) => ({ label, beatmaps: ids.map((id) => playSources.get(id)!) }))
  const group = playGroups.at(Math.min(search.group, playGroups.length - 1))
  const groupIds = group?.beatmaps.map((beatmap) => beatmap.beatmap_id)
  const playedSets = new Set([...playSources.values()].map((beatmap) => beatmap.beatmapset_id))
  const awaitingGroups = session.isPending || (Boolean(session.data) && groups.isPending)
  const personal = useQuery({
    queryKey: ['recommendations', 'group', search, groupIds] as const,
    queryFn: ({ signal }) => recommendBeatmaps(buildRecommendRequest(search, groupIds!), signal),
    enabled: beatmapIds === null && group !== undefined,
    staleTime,
    retry: false,
  })
  const defaults = useQuery({
    queryKey: ['recommendations', 'default', search.mode],
    queryFn: ({ signal }) => fetchDefaultRecommendations(search.mode, signal),
    enabled: beatmapIds === null && !awaitingGroups && group === undefined,
    staleTime,
  })
  const browse = group ? personal : defaults
  const browseResults = group
    ? personal.data?.results.filter((beatmap) => !playedSets.has(beatmap.beatmapset_id))
    : defaults.data?.results
  const response = beatmapIds === null ? null : recommend.data ?? null
  const sourceBeatmaps = response?.sources.map((source) => source.metadata) ?? []
  const selectedSource = sourceBeatmaps.find((beatmap) => beatmap.beatmap_id === selectedSourceId) ?? sourceBeatmaps[0] ?? null
  const isLoading = recommend.isFetching || browse.isFetching
  const requestError = beatmapIds === null ? browse.error : recommend.error
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

    await navigate({ search: { ...defaultSearch, mode: values.mode, group: values.group }, replace: true })
  }

  function selectGroup(index: number) {
    exitResults()
    void navigate({ search: { ...search, group: index }, replace: true })
  }

  async function signOut() {
    await logout()
    queryClient.setQueryData(['session'], null)
    queryClient.removeQueries({ queryKey: ['groups'] })
  }

  async function updateBeatmaps(values: RecommendSearch) {
    if (!parseBeatmapIds(values.beatmap)) {
      exitResults()
      await navigate({ search: values })
      return
    }
    await runRecommend(values, 'push', false)
  }

  async function searchBeatmap(beatmap: BeatmapMetadata) {
    const nextValues = { ...search, beatmap: String(beatmap.beatmap_id) }
    if (beatmap.title) {
      queryClient.setQueryData<string | null>(['beatmap-title', search.mode, beatmap.beatmap_id], (title) => title ?? beatmap.title)
    }
    setSelectedSourceId(null)
    scrollToPageTop()
    await runRecommend(nextValues, 'push', false)
  }

  const defaultResults = beatmapIds === null ? browseResults : undefined
  const resultBeatmaps = response?.results ?? defaultResults ?? []
  const hasResults = resultBeatmaps.length > 0
  const showDefaultResults = defaultResults !== undefined
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
  const showSourcePlaceholder = !selectedSource && isLoading && beatmapIds !== null

  return (
    <main className={styles['app-shell']}>
      {audio.audioElement}

      <section className={styles['results-panel']}>
        <div className={styles['recommend-layout']}>
          {selectedSource ? (
            <SourceBeatmapPager
              beatmaps={sourceBeatmaps}
              beatmap={selectedSource}
              onSelect={setSelectedSourceId}
              onCopy={copyBeatmapId}
              onPlayPreview={(beatmap) => audio.playPreview(beatmap)}
              activePreviewSetId={audio.activeBeatmap?.beatmapset_id ?? null}
              isPreviewPlaying={audio.isPlaying}
            />
          ) : showSourcePlaceholder ? (
            <div className={`${cardStyles['beatmap-card']} ${cardStyles['source-card']} ${styles['source-card-placeholder']}`} data-card-variant="source" aria-hidden="true">
              <div className={`${cardStyles['cover-preview']} ${cardStyles['source-cover']} ${listStyles['loading-result-cover']}`} />
              <div className={`${cardStyles['beatmap-card-content']} ${cardStyles['source-content']}`}>
                <div className={`${cardStyles['source-copy']} ${listStyles['loading-result-copy']}`}>
                  <span className={`${listStyles['loading-result-line']} ${listStyles['loading-result-title']}`} />
                  <span className={`${listStyles['loading-result-line']} ${listStyles['loading-result-artist']}`} />
                  <span className={`${listStyles['loading-result-line']} ${listStyles['loading-result-version']}`} />
                </div>
                <div className={`${cardStyles['source-meta']} ${listStyles['loading-result-meta']}`}>
                  <span className={`${listStyles['loading-result-line']} ${listStyles['loading-result-creator']}`} />
                  <span className={`${listStyles['loading-result-line']} ${listStyles['loading-result-status']}`} />
                </div>
                <div className={cardStyles['source-meta-separator']} />
                <div className={cardStyles['stat-strip']}>
                  <div className={`${cardStyles['stat-row']} ${cardStyles['stat-row-main']}`}>
                    {Array.from({ length: 3 }, (_, stat) => (
                      <div key={stat} className={cardStyles['stat-item']}>
                        <span className={`${listStyles['loading-result-stat']} ${listStyles['loading-result-stat-icon']}`} />
                        <span className={`${listStyles['loading-result-stat']} ${listStyles['loading-result-stat-main']}`} />
                      </div>
                    ))}
                  </div>
                  <div className={cardStyles['stat-side']}>
                    <div className={cardStyles['source-stat-separator']} />
                    <div className={`${cardStyles['stat-row']} ${cardStyles['stat-row-sub']}`}>
                      {Array.from({ length: 4 }, (_, stat) => (
                        <div key={stat} className={cardStyles['stat-item']}>
                          <span className={`${listStyles['loading-result-stat']} ${listStyles['loading-result-stat-label']}`} />
                          <span className={`${listStyles['loading-result-stat']} ${listStyles['loading-result-stat-main']}`} />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : beatmapIds === null ? (
            <header className={styles['intro']}>
              {group ? (
                <PlayGroupTabs groups={playGroups} selected={playGroups.indexOf(group)} onSelect={selectGroup} />
              ) : (
                <div className={styles['intro-copy']}>
                  <div className={styles['intro-lead']}>
                    <span>to find similar maps:</span>
                  </div>
                  <span className={styles['intro-steps']}>
                    <span><span className={styles['intro-action']}><CursorText aria-hidden="true" /> search</span> for a map</span>
                    <span><span className={styles['intro-action']}><ClipboardText aria-hidden="true" /> paste</span> a link/id anywhere</span>
                    <span><span className={styles['intro-action']}><MagnifyingGlass aria-hidden="true" /> click</span> on any map below</span>
                  </span>
                  <span className={styles['intro-outro']}>have fun!</span>
                </div>
              )}
              <div className={styles['intro-actions']}>
                <div className={styles['mode-controls']} role="group" aria-label="Game mode">
                  {modes.map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      aria-pressed={search.mode === mode}
                      onClick={() => {
                        setOutgoingResults(null)
                        void navigate({ search: { ...defaultSearch, mode } })
                      }}
                    >{modeLabels[mode]}</button>
                  ))}
                </div>
                {session.data ? (
                  <>
                    <a href={userUrl(session.data.user_id)} target="_blank" rel="noreferrer">
                      <img className={styles['avatar']} src={avatarUrl(session.data.user_id)} alt="" />
                      {session.data.username}
                    </a>
                    <button type="button" onClick={() => { void signOut() }}>
                      <SignOut aria-hidden="true" />
                      log out
                    </button>
                  </>
                ) : session.data === null ? (
                  <a href={`/api/auth/login?next=${encodeURIComponent(window.location.pathname + window.location.search)}`}>
                    <SignIn aria-hidden="true" />
                    log in
                  </a>
                ) : null}
                {session.data ? null : (
                  <a href="https://github.com/token03/bobert" target="_blank" rel="noreferrer" aria-label="view source on github" title="view source on github">
                    <GithubLogo aria-hidden="true" />
                    view source
                  </a>
                )}
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
              <LoadingResultsList />
            ) : requestError ? (
              <section className={styles['empty-results']} role="alert">
                <WarningCircle aria-hidden="true" />
                <h2>{unavailableBeatmapId === null ? 'Couldn’t load recommendations' : 'Beatmap unavailable'}</h2>
                <p>{unavailableBeatmapId === null ? requestError.message : `Beatmap #${unavailableBeatmapId} couldn’t be downloaded. Try another ID or link.`}</p>
                <div className={styles['state-actions']}>
                  <button type="button" onClick={() => { void (beatmapIds === null ? browse.refetch() : recommend.refetch()) }}>Try again</button>
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
                  <button type="button" onClick={() => { void updateBeatmaps({ ...defaultSearch, mode: search.mode, group: search.group }) }}>Choose another map</button>
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
  onSelect: (beatmapId: number) => void
  onCopy: (beatmapId: number) => Promise<void>
  onPlayPreview: (beatmap: BeatmapMetadata) => Promise<void>
  activePreviewSetId: number | null
  isPreviewPlaying: boolean
}

function SourceBeatmapPager({ beatmaps, beatmap, onSelect, onCopy, onPlayPreview, activePreviewSetId, isPreviewPlaying }: SourceBeatmapPagerProps) {
  const card = (
    <SourceBeatmapCard
      key={beatmap.beatmap_id}
      beatmap={beatmap}
      onCopy={onCopy}
      onPlayPreview={onPlayPreview}
      activePreviewSetId={activePreviewSetId}
      isPreviewPlaying={isPreviewPlaying}
    />
  )

  if (beatmaps.length < 2) {
    return card
  }

  return (
    <Tabs.Root className={styles['source-pager']} value={beatmap.beatmap_id} onValueChange={(beatmapId) => onSelect(beatmapId as number)}>
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
