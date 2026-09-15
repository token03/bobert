import { useCallback, useEffect, useRef, useState } from 'react'
import { Combobox } from '@base-ui/react/combobox'
import { ArrowLeft, CaretRight, Star } from '@phosphor-icons/react'
import type { SearchSet } from './search'
import { searchBeatmaps, startSearch } from './searchClient'
import { parseBeatmapIds } from './filters'
import { formatDifficultyStat, formatLength, formatNumber } from '../../shared/format'
import styles from './BeatmapSearch.module.css'

type Props = {
  query: string
  hasSelection: boolean
  onQuery: (value: string) => void
  onSelect: (value: string) => void
  onRemoveLast: () => void
}

const statusLabel = (status: number) => (status === 2 ? 'Ranked' : status === 1 ? 'Loved' : 'Unranked')

function median(values: number[]): number | null {
  if (!values.length) return null
  const sorted = [...values].sort((left, right) => left - right)
  const middle = sorted.length >> 1
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

export function BeatmapSearch({ query, hasSelection, onQuery, onSelect, onRemoveLast }: Props) {
  const [anchor, setAnchor] = useState<Element | null>(null)
  const [found, setFound] = useState<SearchSet[]>([])
  const [searchStatus, setSearchStatus] = useState('')
  const [open, setOpen] = useState(false)
  const [highlighted, setHighlighted] = useState<string | null>(null)
  const [selectedSet, setSelectedSet] = useState<SearchSet | null>(null)
  const inputElement = useRef<HTMLInputElement | null>(null)
  const setIndex = useRef(0)
  const highlightIndex = useRef(-1)
  const highlightTarget = useRef<number | null>(null)
  const inputRef = useCallback((node: HTMLInputElement | null) => {
    setAnchor(node?.parentElement ?? null)
    inputElement.current = node
  }, [])

  useEffect(() => {
    const timer = setTimeout(startSearch, 400)
    return () => clearTimeout(timer)
  }, [])
  useEffect(() => {
    if (query.trim().length < 2 || parseBeatmapIds(query)) return
    let active = true
    const loading = setTimeout(() => setSearchStatus('Searching…'), 200)
    const timer = setTimeout(() => {
      void searchBeatmaps(query).then((results) => {
        if (!active) return
        clearTimeout(loading)
        setFound(results)
        setSearchStatus(results.length ? '' : 'No matching beatmaps')
      }).catch((error: Error) => {
        if (!active) return
        clearTimeout(loading)
        setFound([])
        setSearchStatus(error.message)
      })
    }, 30)
    return () => { active = false; clearTimeout(timer); clearTimeout(loading) }
  }, [query])

  const visible = query.trim().length >= 2 && !parseBeatmapIds(query)
  const results = visible ? found : []
  const status = visible && !results.length ? searchStatus : ''
  const diffs = selectedSet?.diffs ?? []
  const bpm = median(diffs.map((diff) => diff.bpm).filter((value) => value > 0))
  const length = median(diffs.map((diff) => diff.length).filter((value) => value > 0))

  const openSet = (set: SearchSet) => {
    const index = results.findIndex((candidate) => candidate.id === set.id)
    setIndex.current = index < 0 ? 0 : index
    highlightTarget.current = 0
    setSelectedSet(set)
    setOpen(true)
  }

  const back = () => {
    highlightTarget.current = setIndex.current
    setSelectedSet(null)
  }

  useEffect(() => {
    if (highlightTarget.current === null) return
    const to = highlightTarget.current
    highlightTarget.current = null
    const from = Math.max(highlightIndex.current, 0)
    const key = to > from ? 'ArrowDown' : 'ArrowUp'
    for (let step = 0; step < Math.abs(to - from); step++) {
      inputElement.current?.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }))
    }
  }, [selectedSet])

  return (
    <Combobox.Root<string>
      items={selectedSet ? diffs.map((diff) => `map:${diff.id}`) : results.map((set) => (set.diffs.length === 1 ? `map:${set.diffs[0].id}` : `set:${set.id}`))}
      filter={null}
      autoHighlight={'always' as unknown as boolean}
      value={null}
      inputValue={query}
      onInputValueChange={(value, details) => {
        if (details.reason === 'item-press' || details.reason === 'escape-key') return
        const forcedClear = details.reason === 'input-clear' && !details.event.isTrusted
        if (forcedClear) return
        setSelectedSet(null)
        setHighlighted(null)
        onQuery(value)
      }}
      open={open && visible && Boolean(results.length || selectedSet || status)}
      onOpenChange={(nextOpen, details) => {
        if (!nextOpen && selectedSet && details.reason === 'escape-key') {
          details.cancel()
          back()
          return
        }
        if (!nextOpen) setSelectedSet(null)
        setOpen(nextOpen)
      }}
      onItemHighlighted={(value, details) => {
        highlightIndex.current = details.index
        setHighlighted(value ?? null)
      }}
      onValueChange={(value, details) => {
        if (value?.startsWith('set:')) {
          details.cancel()
          const set = results.find((candidate) => candidate.id === Number(value.slice(4)))
          if (set) openSet(set)
        } else if (value?.startsWith('map:')) {
          onSelect(value.slice(4))
          setSelectedSet(null)
          setOpen(false)
        }
      }}
    >
      <Combobox.Input
        id="beatmap"
        ref={inputRef}
        placeholder={hasSelection ? '' : 'Artist, title, mapper, ID or link'}
        onFocus={startSearch}
        onKeyDown={(event) => {
          if (event.key === 'Backspace' && !query) onRemoveLast()
          if (event.key === 'ArrowLeft' && selectedSet) {
            event.preventDefault()
            back()
          }
          if (event.key === 'ArrowRight' && open && !selectedSet && highlighted?.startsWith('set:')) {
            const set = results.find((candidate) => candidate.id === Number(highlighted.slice(4)))
            if (set) {
              event.preventDefault()
              openSet(set)
            }
          }
          if (event.key === 'Enter' && parseBeatmapIds(query)) {
            event.preventDefault()
            onSelect(query)
          } else if (event.key === 'Enter' && query.trim()) event.preventDefault()
        }}
        onPaste={(event) => {
          const text = event.clipboardData.getData('text')
          if (parseBeatmapIds(text)) {
            event.preventDefault()
            onSelect(text)
          }
        }}
      />
      <Combobox.Portal>
        <Combobox.Positioner anchor={anchor} positionMethod="fixed" sideOffset={6} align="start" className={styles.positioner}>
          <Combobox.Popup className={styles.popup}>
            {selectedSet && (
              <div className={styles.detailHeading}>
                <button type="button" onClick={() => { back(); inputElement.current?.focus() }} aria-label="Back to beatmapsets"><ArrowLeft /></button>
                <span className={styles.cover} style={{ backgroundImage: `url(https://assets.ppy.sh/beatmaps/${selectedSet.id}/covers/list.jpg)` }} />
                <span className={styles.info}><strong>{selectedSet.title}</strong><small>{selectedSet.artist} · {selectedSet.creator}</small></span>
                <span className={styles.meta}>
                  <span data-status={selectedSet.status}>{statusLabel(selectedSet.status)}</span>
                  <small>{diffs.length} difficulties</small>
                  {bpm !== null && length !== null ? <small>{formatNumber(bpm, 0)} BPM · {formatLength(length)}</small> : null}
                </span>
              </div>
            )}
            {!selectedSet && status && <div className={styles.status} role="status">{status}</div>}
            <Combobox.List>
              {selectedSet ? diffs.map((diff) => (
                <Combobox.Item key={diff.id} value={`map:${diff.id}`} className={styles.item}>
                  <span className={styles.version}>{diff.version}</span>
                  <span className={styles.difficulty}>
                    {([['AR', diff.ar], ['CS', diff.cs], ['OD', diff.od], ['HP', diff.hp]] as const).map(([label, value]) => (
                      <span key={label}><span className={styles.difficultyLabel}>{label}</span>{formatDifficultyStat(value)}</span>
                    ))}
                  </span>
                  <span className={styles.stats}>
                    <span className={styles.stars} aria-label={`${diff.stars?.toFixed(2) ?? 'Unknown'} stars`}><Star aria-hidden="true" />{diff.stars?.toFixed(2) ?? '—'}</span>
                  </span>
                </Combobox.Item>
              )) : results.map((set) => (
                <Combobox.Item key={set.id} value={set.diffs.length === 1 ? `map:${set.diffs[0].id}` : `set:${set.id}`} className={styles.set}>
                  <span className={styles.cover} style={{ backgroundImage: `url(https://assets.ppy.sh/beatmaps/${set.id}/covers/list.jpg)` }} />
                  <span className={styles.info}>
                    <strong>{set.title}</strong>
                    <small>{set.artist} · {set.creator}{set.year && <span className={styles.year}> · {set.year}</span>}</small>
                  </span>
                  <span className={styles.meta}>
                    <span data-status={set.status}>{statusLabel(set.status)}</span>
                    <small>{set.diffs.length} {set.diffs.length === 1 ? 'difficulty' : 'difficulties'}</small>
                  </span>
                  <CaretRight className={styles.chevron} aria-hidden="true" data-single={set.diffs.length === 1} />
                </Combobox.Item>
              ))}
            </Combobox.List>
          </Combobox.Popup>
        </Combobox.Positioner>
      </Combobox.Portal>
    </Combobox.Root>
  )
}
