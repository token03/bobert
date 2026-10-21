import { useEffect, useEffectEvent, useRef, useState } from 'react'
import { Popover } from '@base-ui/react/popover'
import { useQuery } from '@tanstack/react-query'
import { ArrowCounterClockwise, X, XCircle } from '@phosphor-icons/react'
import { modeFromUrl, parseBeatmapIds } from '../../shared/beatmapIds'
import type { BeatmapMetadata, Mode } from '../../shared/types'
import { BeatmapSearch } from '../search/BeatmapSearch'
import { lookupBeatmapTitle } from '../search/searchClient'
import { clearFilters, maxBeatmaps } from './filters'
import type { RecommendSearch } from './filters'
import { DateRangeFilter } from './DateRangeFilter'
import { KeyFilter, RangeFilter } from './RangeFilter'
import { rangeFilters } from './rangeFilters'
import { StatusFilter } from './StatusFilter'
import styles from './RecommendForm.module.css'

type RecommendFormProps = {
  search: RecommendSearch
  sources: BeatmapMetadata[]
  unavailableBeatmapId: number | null
  isLoading: boolean
  onFiltersChange: (search: RecommendSearch) => void
  onBeatmapsChange: (search: RecommendSearch) => void
  onSubmit: (search: RecommendSearch) => void
  onReset: (search: RecommendSearch) => void
}

export function RecommendForm({ search, sources, unavailableBeatmapId, isLoading, onFiltersChange, onBeatmapsChange, onSubmit, onReset }: RecommendFormProps) {
  const [resetAnimation, setResetAnimation] = useState(0)
  const [beatmapDraft, setBeatmapDraft] = useState('')
  const [beatmapInputError, setBeatmapInputError] = useState<string | null>(null)
  const beatmapTokensRef = useRef<HTMLSpanElement>(null)
  const beatmapIds = parseBeatmapIds(search.beatmap) ?? []

  function withBeatmaps(ids: number[]): RecommendSearch {
    return { ...search, beatmap: ids.join(',') }
  }

  function addBeatmaps(value: string, append: boolean) {
    const pasted = parseBeatmapIds(value)
    if (!pasted) {
      return null
    }
    const mode = modeFromUrl(value) ?? search.mode
    const ids = [...new Set([...(append && mode === search.mode ? beatmapIds : []), ...pasted])]
    if (ids.length > maxBeatmaps) {
      setBeatmapInputError(`Use up to ${maxBeatmaps} beatmaps.`)
      return false
    }
    setBeatmapDraft('')
    setBeatmapInputError(null)
    window.requestAnimationFrame(() => {
      const tokens = beatmapTokensRef.current
      if (tokens) {
        tokens.scrollLeft = tokens.scrollWidth
      }
    })
    return { ...withBeatmaps(ids), mode }
  }

  function searchPastedBeatmaps(value: string, append: boolean) {
    const next = addBeatmaps(value, append)
    if (next) {
      onSubmit(next)
    }
    return next !== null
  }

  function removeBeatmaps(ids: number[]) {
    setBeatmapInputError(null)
    onBeatmapsChange(withBeatmaps(ids))
  }

  function submit() {
    if (!beatmapDraft.trim()) {
      if (beatmapIds.length) {
        onSubmit(search)
      } else {
        onReset(search)
      }
      return
    }
    const next = addBeatmaps(beatmapDraft, true)
    if (next === null) {
      setBeatmapInputError('Choose a difficulty from search, or enter a beatmap ID or link.')
    } else if (next) {
      onSubmit(next)
    }
  }

  function resetForm() {
    setResetAnimation((animation) => animation + 1)
    setBeatmapDraft('')
    setBeatmapInputError(null)
    onReset(clearFilters(search))
  }

  const handleWindowPaste = useEffectEvent((value: string) => searchPastedBeatmaps(value, false))

  useEffect(() => {
    function pasteSearch(event: ClipboardEvent) {
      const target = event.target

      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        (target instanceof HTMLElement && target.isContentEditable) ||
        window.getSelection()?.toString()
      ) {
        return
      }

      if (handleWindowPaste(event.clipboardData?.getData('text') ?? '')) {
        event.preventDefault()
      }
    }

    window.addEventListener('paste', pasteSearch)
    return () => window.removeEventListener('paste', pasteSearch)
  }, [])

  return (
    <form
      className={styles['control-panel']}
      onSubmit={(event) => {
        event.preventDefault()
        submit()
      }}
    >
      <div className={styles['primary-controls']}>
        <div className={`${styles.field} ${styles['beatmap-field']} ${styles['search-field']}`}>
          <label className={styles['sr-only']} htmlFor="beatmap">Search</label>
          <span className={styles['input-with-status']}>
            <span className={styles['search-pill']} data-search-anchor data-error={Boolean(beatmapInputError)}>
              {beatmapIds.length ? (
                <span ref={beatmapTokensRef} className={styles['beatmap-tokens']} role="list" aria-label="Source beatmaps">
                  {beatmapIds.map((beatmapId) => (
                    <BeatmapToken
                      key={beatmapId}
                      beatmapId={beatmapId}
                      mode={search.mode}
                      source={sources.find((source) => source.beatmap_id === beatmapId)}
                      unavailable={beatmapId === unavailableBeatmapId}
                      onRemove={() => removeBeatmaps(beatmapIds.filter((id) => id !== beatmapId))}
                    />
                  ))}
                </span>
              ) : null}
              <span className={styles['search-entry']}>
                <BeatmapSearch
                  mode={search.mode}
                  query={beatmapDraft}
                  hasSelection={beatmapIds.length > 0}
                  isLoading={isLoading}
                  buttonClassName={`${styles['primary-button']} ${styles['search-button']}`}
                  spinnerClassName={styles['spinner-icon']}
                  onQuery={(value) => {
                    setBeatmapDraft(value)
                    setBeatmapInputError(null)
                  }}
                  onSelect={(value) => { searchPastedBeatmaps(value, true) }}
                  onRemoveLast={() => {
                    if (beatmapIds.length) {
                      removeBeatmaps(beatmapIds.slice(0, -1))
                    }
                  }}
                />
              </span>
            </span>
            {beatmapInputError ? (
              <>
                <span id="beatmap-error" className={styles['sr-only']}>{beatmapInputError}</span>
                <FieldErrorPopover label={beatmapInputError} />
              </>
            ) : null}
          </span>
        </div>

        <div className={styles['search-separator']} aria-hidden="true" />

        <div className={styles['filter-controls']}>
          {rangeFilters.filter((config) => config.key !== 'ar' || search.mode === 'std' || search.mode === 'catch').filter((config) => config.key !== 'cs' || search.mode !== 'taiko').map((config) => {
            if (config.key === 'cs' && search.mode === 'mania') {
              return <KeyFilter key={config.key} min={search.minCs} max={search.maxCs} onValueCommit={(minCs, maxCs) => onFiltersChange({ ...search, minCs, maxCs })} />
            }
            return (
            <RangeFilter
              key={config.key}
              config={config}
              mode={search.mode}
              min={search[config.params[0]]}
              max={search[config.params[1]]}
              onValueCommit={(min, max) => onFiltersChange({ ...search, [config.params[0]]: min, [config.params[1]]: max })}
            />
            )
          })}

          <DateRangeFilter
            min={search.minDate}
            max={search.maxDate}
            onValueCommit={(minDate, maxDate) => onFiltersChange({ ...search, minDate, maxDate })}
          />

          <StatusFilter value={search.status} onValueCommit={(status) => onFiltersChange({ ...search, status })} />

          <button className={styles['ghost-button']} data-filter="reset" type="button" onClick={resetForm}>
            <ArrowCounterClockwise key={resetAnimation} data-reset-animate={resetAnimation > 0 || undefined} />
            <span className={styles['sr-only']}>Reset</span>
          </button>
        </div>
      </div>
    </form>
  )
}

type BeatmapTokenProps = {
  mode: Mode
  beatmapId: number
  source: BeatmapMetadata | undefined
  unavailable: boolean
  onRemove: () => void
}

function BeatmapToken({ beatmapId, mode, source, unavailable, onRemove }: BeatmapTokenProps) {
  const catalog = useQuery({
    queryKey: ['beatmap-title', mode, beatmapId],
    queryFn: () => lookupBeatmapTitle(beatmapId, mode),
    staleTime: Infinity,
    retry: false,
  })
  const text = catalog.data ?? source?.title ?? `#${beatmapId}`

  return (
    <span className={styles['beatmap-token']} role="listitem" data-unavailable={unavailable || undefined} title={unavailable ? `Beatmap ${beatmapId} isn't available` : `Beatmap ${beatmapId}`}>
      <span className={styles['beatmap-token-label']}>{text}</span>
      <button type="button" onClick={onRemove} aria-label={`Remove ${text}`}>
        <X />
      </button>
    </span>
  )
}

function FieldErrorPopover({ label }: { label: string }) {
  return (
    <Popover.Root defaultOpen defaultTriggerId="beatmap-error-trigger">
      <Popover.Trigger id="beatmap-error-trigger" className={styles['field-error-icon']} type="button" openOnHover delay={0} aria-label={label}>
        <XCircle />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Positioner className={styles['field-error-positioner']} side="top" align="end" sideOffset={8}>
          <Popover.Popup className={styles['field-error-popover']} initialFocus={false} role="alert">
            <Popover.Description>{label}</Popover.Description>
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  )
}
