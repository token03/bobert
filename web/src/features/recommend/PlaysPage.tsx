import { useState } from 'react'
import { ArrowClockwise, ClockCounterClockwise, MagnifyingGlass, Trophy, WarningCircle } from '@phosphor-icons/react'
import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate } from '@tanstack/react-router'
import { AudioPreviewBar } from '../audio/AudioPreviewBar'
import { useAudioPreview } from '../audio/useAudioPreview'
import { fetchPlays, groupPlays, recommendBeatmaps } from '../../shared/api'
import type { PlayKind } from '../../shared/api'
import { copyText } from '../../shared/copy'
import { formatFixedNumber, formatNumber } from '../../shared/format'
import type { BeatmapMetadata } from '../../shared/types'
import { modeLabels, modes } from '../../shared/types'
import { coverUrl, userUrl } from '../../shared/urls'
import { buildRecommendRequest, clearFilters, defaultSearch } from './filters'
import { defaultPlaysSearch, groupSummary, modCombo, readExcludedPlays, writeExcludedPlays } from './plays'
import type { Play, PlaysSearch } from './plays'
import { FilterControls } from './RecommendForm'
import { LoadingResultsList, ResultsList } from './ResultsList'
import formStyles from './RecommendForm.module.css'
import pageStyles from './RecommendPage.module.css'
import styles from './PlaysPage.module.css'

const staleTime = 5 * 60_000
const kindRoutes = { best: '/top/$userId', recent: '/recent/$userId' } as const

type PlaysPageProps = {
  kind: PlayKind
  userId: number
  search: PlaysSearch
}

export function PlaysPage({ kind, userId, search }: PlaysPageProps) {
  const navigate = useNavigate()
  const audio = useAudioPreview({ onError: console.error })
  const [excluded, setExcluded] = useState(() => readExcludedPlays(userId))
  const { mode } = search
  const plays = useQuery({
    queryKey: ['plays', userId, kind, mode],
    queryFn: ({ signal }) => fetchPlays(userId, kind, mode, signal),
    staleTime,
    retry: false,
  })
  const playsById = new Map<number, Play>()
  for (const play of plays.data?.plays ?? []) {
    if (!playsById.has(play.beatmap_id)) playsById.set(play.beatmap_id, play)
  }
  const groupedPlays = (plays.data?.plays ?? []).map(({ beatmap_id, mods }) => ({ beatmap_id, mods }))
  const grouping = useQuery({
    queryKey: ['groups', mode, groupedPlays],
    queryFn: ({ signal }) => groupPlays(mode, groupedPlays, signal),
    enabled: groupedPlays.length > 0,
    staleTime: Infinity,
  })
  const sources = new Map(grouping.data?.sources.map((beatmap) => [beatmap.beatmap_id, beatmap]))
  const groups = (grouping.data?.groups ?? []).map(({ label, beatmap_ids: ids }) => {
    const beatmaps = ids.map((beatmapId) => sources.get(beatmapId)!)
    return { ids, beatmaps, summary: groupSummary(beatmaps), name: label ?? 'mixed' }
  })
  const groupIndex = Math.min(search.group, Math.max(groups.length - 1, 0))
  const group = groups.at(groupIndex)
  const active = group?.ids.filter((beatmapId) => !excluded.has(beatmapId)) ?? []
  const recommendations = useQuery({
    queryKey: ['recommendations', 'plays', search, active],
    queryFn: ({ signal }) => recommendBeatmaps(buildRecommendRequest(search, active), signal),
    enabled: active.length > 0,
    staleTime,
    retry: false,
    placeholderData: (previous) => previous,
  })
  const playedSets = new Set([...sources.values()].map((beatmap) => beatmap.beatmapset_id))
  playedSets.delete(null)
  const results = (recommendations.data?.results ?? []).filter((beatmap) => !playedSets.has(beatmap.beatmapset_id))
  const error = plays.error ?? grouping.error
  const user = plays.data?.user

  function updateSearch(next: PlaysSearch) {
    void navigate({ to: kindRoutes[kind], params: { userId }, search: next, replace: true })
  }

  function setExcludedPlays(next: Set<number>) {
    writeExcludedPlays(userId, next)
    setExcluded(next)
  }

  function toggle(beatmapId: number) {
    const next = new Set(excluded)
    if (!next.delete(beatmapId)) next.add(beatmapId)
    setExcludedPlays(next)
  }

  async function searchBeatmap(beatmap: BeatmapMetadata) {
    await navigate({ to: '/recommendations', search: { ...defaultSearch, mode, beatmap: String(beatmap.beatmap_id) } })
  }

  return (
    <main className={pageStyles['app-shell']}>
      {audio.audioElement}

      <section className={pageStyles['results-panel']}>
        <header className={pageStyles['intro']}>
          <div className={pageStyles['intro-copy']}>
            <div className={pageStyles['intro-lead']}>
              <span>maps like your:</span>
            </div>
            <span className={pageStyles['intro-steps']}>
              <span>
                <Link className={`${pageStyles['intro-action']} ${styles['kind']}`} to={kindRoutes.best} params={{ userId }} search={{ ...defaultPlaysSearch, mode }}>
                  <Trophy aria-hidden="true" /> top plays
                </Link> your best 100
              </span>
              <span>
                <Link className={`${pageStyles['intro-action']} ${styles['kind']}`} to={kindRoutes.recent} params={{ userId }} search={{ ...defaultPlaysSearch, mode }}>
                  <ClockCounterClockwise aria-hidden="true" /> recent plays
                </Link> last 24 hours
              </span>
              <span>click a cover to leave it out</span>
            </span>
            <span className={pageStyles['intro-outro']}>
              {grouping.data?.missing.length ? `${grouping.data.missing.length} not indexed yet` : null}
            </span>
          </div>
          <div className={pageStyles['intro-actions']}>
            <div className={pageStyles['mode-controls']} role="group" aria-label="Game mode">
              {modes.map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={mode === value}
                  onClick={() => updateSearch({ ...defaultPlaysSearch, mode: value })}
                >{modeLabels[value]}</button>
              ))}
            </div>
            <button type="button" disabled={plays.isFetching} onClick={() => { void plays.refetch() }}>
              <ArrowClockwise aria-hidden="true" />
              refresh
            </button>
            <a href={userUrl(userId)} target="_blank" rel="noreferrer">
              {user?.avatar_url ? <img className={styles['avatar']} src={user.avatar_url} alt="" /> : null}
              {user?.username ?? `user ${userId}`}
            </a>
          </div>
        </header>

        {error ? (
          <section className={pageStyles['empty-results']} role="alert">
            <WarningCircle aria-hidden="true" />
            <h2>Couldn’t load plays</h2>
            <p>{error.message}</p>
            <div className={pageStyles['state-actions']}>
              <button type="button" onClick={() => { void (plays.error ? plays.refetch() : grouping.refetch()) }}>Try again</button>
            </div>
          </section>
        ) : plays.data && playsById.size === 0 ? (
          <section className={pageStyles['empty-results']} role="status">
            <ClockCounterClockwise aria-hidden="true" />
            <h2>No {kind === 'best' ? 'top' : 'recent'} plays</h2>
            <p>{kind === 'recent' ? 'osu! keeps recent plays for 24 hours. Play a few maps and refresh.' : `No ${modeLabels[mode]} plays yet.`}</p>
          </section>
        ) : grouping.data && groups.length === 0 ? (
          <section className={pageStyles['empty-results']} role="status">
            <MagnifyingGlass aria-hidden="true" />
            <h2>No indexed plays yet</h2>
            <p>These plays aren’t in the recommendation index yet. Try another game mode or player.</p>
          </section>
        ) : (
          <>
            <div className={pageStyles['sticky-search-wrap']}>
              <div className={styles['group-tabs']} role="group" aria-label="Play groups">
                {groups.length ? groups.map(({ ids, name }, index) => (
                  <button
                    key={index}
                    type="button"
                    className={styles['group-tab']}
                    aria-pressed={index === groupIndex}
                    onClick={() => updateSearch({ ...search, group: index })}
                  >
                    <span>{name}</span>
                    <span className={styles['group-tab-count']}>{ids.length}</span>
                  </button>
                )) : Array.from({ length: 3 }, (_, index) => <span key={index} className={styles['group-tab-placeholder']} />)}
              </div>
              <div className={formStyles['control-panel']}>
                <div className={formStyles['primary-controls']}>
                  <div className={`${formStyles['beatmap-field']} ${styles['group-summary']}`}>
                    {group ? (
                      <>
                        <strong>{active.length === group.ids.length ? `${group.ids.length} plays` : `${active.length} of ${group.ids.length} plays`}</strong>
                        <span>{formatFixedNumber(group.summary.stars, 2)}★</span>
                        <span>{formatNumber(group.summary.bpm, 0)} bpm</span>
                        {active.length < group.ids.length ? (
                          <button type="button" className={styles['restore']} onClick={() => setExcludedPlays(new Set([...excluded].filter((beatmapId) => !group.ids.includes(beatmapId))))}>
                            include all
                          </button>
                        ) : null}
                      </>
                    ) : null}
                  </div>
                  <div className={formStyles['search-separator']} aria-hidden="true" />
                  <FilterControls
                    search={search}
                    onFiltersChange={(next) => updateSearch({ ...search, ...next })}
                    onReset={() => updateSearch({ ...clearFilters(search), group: search.group })}
                  />
                </div>
              </div>
            </div>

            <div className={styles['group-head']}>
              <div className={styles['group-covers']}>
                {group ? group.beatmaps.map((beatmap) => {
                  const play = playsById.get(beatmap.beatmap_id)!
                  return (
                    <button
                      key={beatmap.beatmap_id}
                      type="button"
                      className={styles['cover-toggle']}
                      aria-pressed={!excluded.has(beatmap.beatmap_id)}
                      aria-label={`${excluded.has(beatmap.beatmap_id) ? 'Include' : 'Leave out'} ${beatmap.title} [${beatmap.version}]`}
                      title={`${beatmap.title} [${beatmap.version}] · ${modCombo(play.mods)}${play.pp === null ? '' : ` · ${Math.round(play.pp)}pp`}`}
                      onClick={() => toggle(beatmap.beatmap_id)}
                    >
                      {beatmap.beatmapset_id === null ? null : <img src={coverUrl(beatmap.beatmapset_id)} alt="" />}
                    </button>
                  )
                }) : Array.from({ length: 12 }, (_, index) => <span key={index} className={styles['cover-placeholder']} />)}
              </div>
            </div>

            {group && active.length === 0 ? (
              <section className={pageStyles['empty-results']} role="status">
                <MagnifyingGlass aria-hidden="true" />
                <h2>Every play here is left out</h2>
                <p>Click a cover to bring it back.</p>
              </section>
            ) : recommendations.error ? (
              <section className={pageStyles['empty-results']} role="alert">
                <WarningCircle aria-hidden="true" />
                <h2>Couldn’t load recommendations</h2>
                <p>{recommendations.error.message}</p>
                <div className={pageStyles['state-actions']}>
                  <button type="button" onClick={() => { void recommendations.refetch() }}>Try again</button>
                </div>
              </section>
            ) : !recommendations.data ? (
              <LoadingResultsList count={12} />
            ) : results.length ? (
              <div className={recommendations.isPlaceholderData ? styles['results-stale'] : undefined}>
                <ResultsList
                  beatmaps={results}
                  onCopy={(beatmapId) => copyText(String(beatmapId))}
                  onSearch={searchBeatmap}
                  isLoading={recommendations.isFetching}
                  onPlayPreview={(beatmap) => audio.playPreview(beatmap)}
                  activePreviewSetId={audio.activeBeatmap?.beatmapset_id ?? null}
                  isPreviewPlaying={audio.isPlaying}
                />
              </div>
            ) : (
              <section className={pageStyles['empty-results']} role="status">
                <MagnifyingGlass aria-hidden="true" />
                <h2>No matching beatmaps</h2>
                <p>Try widening your filters.</p>
                <div className={pageStyles['state-actions']}>
                  <button type="button" onClick={() => updateSearch({ ...clearFilters(search), group: search.group })}>Clear filters</button>
                </div>
              </section>
            )}
          </>
        )}
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
    </main>
  )
}
