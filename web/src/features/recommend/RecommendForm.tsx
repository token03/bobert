import { useEffect, useEffectEvent, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Popover } from '@base-ui/react/popover'
import { Select } from '@base-ui/react/select'
import { useStore } from '@tanstack/react-form'
import { ArrowCounterClockwise, CaretDown, Check, Clock, MagnifyingGlass, Metronome, Spinner, Star, Tag, X, XCircle } from '@phosphor-icons/react'
import { defaultFilters, maxBeatmaps, parseBeatmapIds } from './filters'
import type { RecommendFormValues } from './filters'
import { DateRangeFilter } from './DateRangeFilter'
import { BeatmapSearch } from './BeatmapSearch'
import { RangeFilter } from './RangeFilter'
import type { RangeFilterConfig } from './RangeFilter'
import type { useRecommendForm } from './useRecommendForm'
import styles from './RecommendForm.module.css'

type RecommendFormProps = {
  form: ReturnType<typeof useRecommendForm>
  isLoading: boolean
  onRangeChange: (values: RecommendFormValues) => void
  onSelectChange: (values: RecommendFormValues) => void
  onBeatmapsChange: (values: RecommendFormValues) => void
  onPasteSearch: (values: RecommendFormValues) => void
  onReset: (values: RecommendFormValues) => void
}

type RangeFieldName = 'minSr' | 'maxSr' | 'minAr' | 'maxAr' | 'minCs' | 'maxCs' | 'minLength' | 'maxLength' | 'minBpm' | 'maxBpm'

const decimalAria = (unit: string) => (value: number, overflow: 'lower' | 'higher' | null) => `${value.toFixed(1)} ${unit}${overflow ? ` or ${overflow}` : ''}`
const integerAria = (unit: string) => (value: number, overflow: 'lower' | 'higher' | null) => `${value} ${unit}${overflow ? ` or ${overflow}` : ''}`

function formatDuration(value: number) {
  const hours = Math.floor(value / 3600)
  const minutes = Math.floor(value % 3600 / 60)
  const seconds = String(value % 60).padStart(2, '0')
  return hours > 0 ? `${hours}:${String(minutes).padStart(2, '0')}:${seconds}` : `${minutes}:${seconds}`
}

function formatDurationAria(value: number, overflow: 'lower' | 'higher' | null) {
  const minutes = Math.floor(value / 60)
  const seconds = value % 60
  const parts = [minutes ? `${minutes} minute${minutes === 1 ? '' : 's'}` : '', seconds ? `${seconds} second${seconds === 1 ? '' : 's'}` : ''].filter(Boolean)
  return `${parts.join(' ') || '0 seconds'}${overflow ? ` or ${overflow === 'higher' ? 'longer' : 'shorter'}` : ''}`
}

const lengthSliderMax = 200
const lengthMax = 20 * 60
const lengthScale = Math.log1p(lengthMax / 60)

function durationToSlider(value: number) {
  return lengthSliderMax * Math.log1p(value / 60) / lengthScale
}

function sliderToDuration(value: number) {
  return Math.round(60 * Math.expm1(value / lengthSliderMax * lengthScale))
}

const rangeFilters: ReadonlyArray<RangeFilterConfig & { minField: RangeFieldName; maxField: RangeFieldName }> = [
  {
    key: 'stars',
    label: 'Star rating',
    defaultLabel: 'Stars',
    icon: <Star />,
    minField: 'minSr',
    maxField: 'maxSr',
    min: 0,
    max: 12,
    step: 0.1,
    largeStep: 0.5,
    overflowMax: true,
    formatValue: (value) => value.toFixed(1),
    formatAriaValue: decimalAria('stars'),
  },
  {
    key: 'ar',
    label: 'Approach rate',
    triggerLabel: 'AR',
    defaultLabel: '—',
    minField: 'minAr',
    maxField: 'maxAr',
    min: 0,
    max: 10,
    step: 0.1,
    largeStep: 0.5,
    formatValue: (value) => value.toFixed(1),
    formatAriaValue: decimalAria('approach rate'),
  },
  {
    key: 'cs',
    label: 'Circle size',
    triggerLabel: 'CS',
    defaultLabel: '—',
    minField: 'minCs',
    maxField: 'maxCs',
    min: 0,
    max: 10,
    step: 0.1,
    largeStep: 0.5,
    formatValue: (value) => value.toFixed(1),
    formatAriaValue: decimalAria('circle size'),
  },
  {
    key: 'bpm',
    label: 'BPM',
    defaultLabel: 'BPM',
    icon: <Metronome />,
    minField: 'minBpm',
    maxField: 'maxBpm',
    min: 100,
    max: 300,
    step: 5,
    largeStep: 25,
    overflowMin: true,
    overflowMax: true,
    formatValue: String,
    formatAriaValue: integerAria('beats per minute'),
  },
  {
    key: 'length',
    label: 'Length',
    defaultLabel: 'Length',
    icon: <Clock />,
    minField: 'minLength',
    maxField: 'maxLength',
    min: 0,
    max: lengthMax,
    step: 5,
    largeStep: 30,
    sliderMin: 0,
    sliderMax: lengthSliderMax,
    sliderStep: 1,
    sliderLargeStep: 6,
    overflowMax: true,
    formatValue: formatDuration,
    formatAriaValue: formatDurationAria,
    toSliderValue: durationToSlider,
    fromSliderValue: sliderToDuration,
  },
]

const statusOptions = [
  { value: null, label: 'Any' },
  { value: 'ranked', label: 'Ranked' },
  { value: 'loved', label: 'Loved' },
  { value: 'unranked', label: 'Unranked' },
] as const

export function RecommendForm({ form, isLoading, onRangeChange, onSelectChange, onBeatmapsChange, onPasteSearch, onReset }: RecommendFormProps) {
  const [resetAnimation, setResetAnimation] = useState(0)
  const [beatmapDraft, setBeatmapDraft] = useState('')
  const [beatmapInputError, setBeatmapInputError] = useState<string | null>(null)
  const beatmapTokensRef = useRef<HTMLSpanElement>(null)
  const { values, isSubmitting } = useStore(form.store, (state) => ({
    values: state.values,
    isSubmitting: state.isSubmitting,
  }))
  const submitDisabled = isLoading || isSubmitting

  function addBeatmaps(value: string, append: boolean) {
    const pasted = parseBeatmapIds(value)
    if (!pasted) {
      return null
    }
    const current = append ? parseBeatmapIds(form.getFieldValue('beatmap')) ?? [] : []
    const ids = [...new Set([...current, ...pasted])]
    if (ids.length > maxBeatmaps) {
      setBeatmapInputError(`Use up to ${maxBeatmaps} beatmaps.`)
      return false
    }
    const beatmap = ids.join(',')
    const nextValues = { ...form.state.values, beatmap }
    form.setFieldValue('beatmap', beatmap, { dontValidate: true })
    setBeatmapDraft('')
    setBeatmapInputError(null)
    window.requestAnimationFrame(() => {
      const tokens = beatmapTokensRef.current
      if (tokens) {
        tokens.scrollLeft = tokens.scrollWidth
      }
    })
    return nextValues
  }

  function searchPastedBeatmaps(value: string, append: boolean) {
    const nextValues = addBeatmaps(value, append)
    if (nextValues) {
      onPasteSearch(nextValues)
    }
    return nextValues !== null
  }

  function commitBeatmapDraft() {
    if (!beatmapDraft.trim()) {
      return true
    }
    const result = addBeatmaps(beatmapDraft, true)
    if (result === null) {
      setBeatmapInputError('Choose a difficulty from search, or enter a beatmap ID or link.')
      return false
    }
    return result !== false
  }

  function removeBeatmap(beatmapId: number) {
    const ids = (parseBeatmapIds(form.getFieldValue('beatmap')) ?? []).filter((id) => id !== beatmapId)
    const beatmap = ids.join(',')
    const nextValues = { ...form.state.values, beatmap }
    form.setFieldValue('beatmap', beatmap, { dontValidate: true })
    setBeatmapInputError(null)
    onBeatmapsChange(nextValues)
  }

  function removeLastBeatmap() {
    const ids = parseBeatmapIds(form.getFieldValue('beatmap')) ?? []
    if (!ids.length) {
      return
    }
    const beatmap = ids.slice(0, -1).join(',')
    const nextValues = { ...form.state.values, beatmap }
    form.setFieldValue('beatmap', beatmap, { dontValidate: true })
    setBeatmapInputError(null)
    onBeatmapsChange(nextValues)
  }

  function updateBeatmapDraft(value: string) {
    setBeatmapDraft(value)
    if (beatmapInputError) {
      setBeatmapInputError(null)
    }
  }

  function resetForm() {
    setResetAnimation((animation) => animation + 1)
    setBeatmapDraft('')
    setBeatmapInputError(null)
    const nextValues = { ...defaultFilters, beatmap: form.getFieldValue('beatmap') }
    form.reset(nextValues)
    onReset(nextValues)
  }

  function updateRange(minField: RangeFieldName, maxField: RangeFieldName, min: string, max: string) {
    form.setFieldValue(minField, min)
    form.setFieldValue(maxField, max)
    onRangeChange({ ...form.state.values, [minField]: min, [maxField]: max })
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
        if (commitBeatmapDraft()) {
          if (!form.getFieldValue('beatmap').trim()) {
            onReset(form.state.values)
          } else {
            void form.handleSubmit()
          }
        }
      }}
    >
      <div className={styles['primary-controls']}>
        <div className={`${styles.field} ${styles['beatmap-field']} ${styles['search-field']}`}>
          <label className={styles['sr-only']} htmlFor="beatmap">Search</label>
          <form.Field name="beatmap">
            {(field) => {
              const beatmapError = beatmapInputError ?? field.state.meta.errors[0]?.message
              const beatmapIds = parseBeatmapIds(field.state.value) ?? []

              return (
                <span className={styles['input-with-status']}>
                  <span className={styles['search-pill']} data-error={Boolean(beatmapError)}>
                    {beatmapIds.length ? (
                      <span ref={beatmapTokensRef} className={styles['beatmap-tokens']} role="list" aria-label="Source beatmaps">
                        {beatmapIds.map((beatmapId) => (
                          <span className={styles['beatmap-token']} role="listitem" key={beatmapId}>
                            <span>{beatmapId}</span>
                            <button type="button" onClick={() => removeBeatmap(beatmapId)} aria-label={`Remove beatmap ${beatmapId}`}>
                              <X />
                            </button>
                          </span>
                        ))}
                      </span>
                    ) : null}
                    <BeatmapSearch
                      query={beatmapDraft}
                      hasSelection={beatmapIds.length > 0}
                      onQuery={updateBeatmapDraft}
                      onSelect={(value) => { searchPastedBeatmaps(value, true) }}
                      onRemoveLast={removeLastBeatmap}
                    />
                    <button className={`${styles['primary-button']} ${styles['search-button']}`} type="submit" disabled={submitDisabled} aria-label="Recommend">
                      {submitDisabled ? <Spinner className={styles['spinner-icon']} /> : <MagnifyingGlass />}
                      <span className={styles['sr-only']}>Recommend</span>
                    </button>
                  </span>
                  {beatmapError ? (
                    <>
                      <span id="beatmap-error" className={styles['sr-only']}>{beatmapError}</span>
                      <FieldErrorPopover label={beatmapError} />
                    </>
                  ) : null}
                </span>
              )
            }}
          </form.Field>
        </div>

        <div className={styles['search-separator']} aria-hidden="true" />

        <div className={styles['filter-controls']}>
          {rangeFilters.map(({ minField, maxField, ...config }) => (
            <RangeFilter
              key={config.key}
              config={config}
              minValue={values[minField] as string}
              maxValue={values[maxField] as string}
              onValueCommit={(min, max) => updateRange(minField, maxField, min, max)}
            />
          ))}

          <DateRangeFilter
            minValue={values.minDate}
            maxValue={values.maxDate}
            onValueCommit={(minDate, maxDate) => {
              form.setFieldValue('minDate', minDate)
              form.setFieldValue('maxDate', maxDate)
              onSelectChange({ ...form.state.values, minDate, maxDate })
            }}
          />

          <FilterSelect
            filterKey="status"
            label="Status"
            defaultLabel="Status"
            icon={<Tag />}
            value={values.status}
            options={statusOptions}
            onValueChange={(status) => {
              form.setFieldValue('status', status)
              onSelectChange({ ...form.state.values, status })
            }}
          />

          <button className={styles['ghost-button']} data-filter="reset" type="button" onClick={resetForm}>
            <ArrowCounterClockwise key={resetAnimation} data-reset-animate={resetAnimation > 0 || undefined} />
            <span className={styles['sr-only']}>Reset</span>
          </button>
        </div>
      </div>
    </form>
  )
}

type FilterSelectProps = {
  filterKey: string
  label: string
  defaultLabel: string
  icon: ReactNode
  value: string
  options: ReadonlyArray<{ value: string | null; label: string }>
  onValueChange: (value: string) => void
}

function FilterSelect({ filterKey, label, defaultLabel, icon, value, options, onValueChange }: FilterSelectProps) {
  return (
    <Select.Root
      items={options}
      value={value || null}
      onValueChange={(nextValue) => onValueChange(nextValue ?? '')}
    >
      <Select.Trigger
        className={`${styles.field} ${styles['select-field']}`}
        data-active={value || undefined}
        data-filter={filterKey}
        aria-label={label}
      >
        <span aria-hidden="true">{icon}</span>
        <Select.Value className={styles['select-value']}>
          {value ? options.find((option) => option.value === value)?.label : defaultLabel}
        </Select.Value>
        <Select.Icon className={styles['select-icon']}>
          <CaretDown />
        </Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Positioner
          className={styles['select-positioner']}
          sideOffset={3}
          align="start"
          alignItemWithTrigger={false}
        >
          <Select.Popup className={styles['select-popup']}>
            <Select.List className={styles['select-list']}>
              {options.map((option) => (
                <Select.Item
                  className={styles['select-item']}
                  key={option.value ?? 'empty'}
                  value={option.value}
                >
                  <Select.ItemText>{option.label}</Select.ItemText>
                  <Select.ItemIndicator className={styles['select-item-indicator']}>
                    <Check />
                  </Select.ItemIndicator>
                </Select.Item>
              ))}
            </Select.List>
          </Select.Popup>
        </Select.Positioner>
      </Select.Portal>
    </Select.Root>
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
