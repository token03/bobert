import { useEffect, useEffectEvent } from 'react'
import type { ReactNode } from 'react'
import { Select } from '@base-ui/react/select'
import { Tooltip } from '@base-ui/react/tooltip'
import { useStore } from '@tanstack/react-form'
import { CalendarDays, Check, ChevronDown, Clock, Loader, Metronome, RotateCcw, Search, Star, Tag, XCircle } from 'lucide-react'
import { defaultFilters, normalizeBeatmapInput, parseBeatmapId } from './filters'
import type { RecommendFormValues } from './filters'
import { RangeFields } from './RangeFields'
import type { useRecommendForm } from './useRecommendForm'
import styles from './RecommendForm.module.css'

type RecommendFormProps = {
  form: ReturnType<typeof useRecommendForm>
  isLoading: boolean
  onRangeChange: (values: RecommendFormValues) => void
  onSelectChange: (values: RecommendFormValues) => void
  onPasteSearch: (values: RecommendFormValues) => void
  onReset: (values: RecommendFormValues) => void
}

type RangeFieldName = 'minSr' | 'maxSr' | 'minLength' | 'maxLength' | 'minBpm' | 'maxBpm'

const dateWindowOptions = [
  { value: null, label: 'All time' },
  { value: 'last_week', label: 'Last week' },
  { value: 'last_month', label: 'Last month' },
  { value: 'last_3_months', label: 'Last 3 months' },
  { value: 'last_6_months', label: 'Last 6 months' },
  { value: 'last_year', label: 'Last year' },
  { value: 'last_2_years', label: 'Last 2 years' },
  { value: 'last_5_years', label: 'Last 5 years' },
] as const

const statusOptions = [
  { value: null, label: 'Any' },
  { value: 'ranked', label: 'Ranked' },
  { value: 'loved', label: 'Loved' },
  { value: 'unranked', label: 'Unranked' },
] as const

export function RecommendForm({ form, isLoading, onRangeChange, onSelectChange, onPasteSearch, onReset }: RecommendFormProps) {
  const { values, isSubmitting } = useStore(form.store, (state) => ({
    values: state.values,
    isSubmitting: state.isSubmitting,
  }))
  const submitDisabled = isLoading || isSubmitting

  function resetForm() {
    const nextValues = { ...defaultFilters, beatmap: form.getFieldValue('beatmap') }
    form.reset(nextValues)
    onReset(nextValues)
  }

  function updateRange(field: RangeFieldName, value: string) {
    form.setFieldValue(field, value)
    onRangeChange({ ...form.state.values, [field]: value })
  }

  function searchPastedBeatmap(value: string) {
    if (!parseBeatmapId(value)) {
      return false
    }

    const beatmap = normalizeBeatmapInput(value)
    const nextValues = { ...form.state.values, beatmap }
    form.setFieldValue('beatmap', beatmap, { dontValidate: true })
    onPasteSearch(nextValues)
    return true
  }

  const handleWindowPaste = useEffectEvent((value: string) => searchPastedBeatmap(value))

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
        void form.handleSubmit()
      }}
    >
      <div className={styles['primary-controls']}>
        <div className={`${styles.field} ${styles['beatmap-field']} ${styles['search-field']}`}>
          <label className={styles['sr-only']} htmlFor="beatmap">Search</label>
          <form.Field name="beatmap">
            {(field) => {
              const beatmapError = field.state.meta.errors[0]?.message

              return (
                <span className={styles['input-with-status']}>
                  <span className={styles['search-pill']}>
                    <input
                      required
                      id="beatmap"
                      name={field.name}
                      value={field.state.value}
                      aria-invalid={beatmapError ? 'true' : 'false'}
                      aria-describedby={beatmapError ? 'beatmap-error' : undefined}
                      data-error={Boolean(beatmapError)}
                      onChange={(event) => field.handleChange(event.target.value)}
                      onPaste={(event) => {
                        if (searchPastedBeatmap(event.clipboardData.getData('text'))) {
                          event.preventDefault()
                        }
                      }}
                      onBlur={(event) => {
                        field.handleBlur()
                        form.setFieldValue('beatmap', normalizeBeatmapInput(event.target.value), { dontValidate: true })
                      }}
                      placeholder="1872396 or https://osu.ppy.sh/beatmaps/1872396"
                    />
                    <button className={`${styles['primary-button']} ${styles['search-button']}`} type="submit" disabled={submitDisabled} aria-label="Recommend">
                      {submitDisabled ? <Loader className={styles['spinner-icon']} /> : <Search />}
                      <span className={styles['sr-only']}>Recommend</span>
                    </button>
                  </span>
                  {beatmapError ? (
                    <>
                      <span id="beatmap-error" className={styles['sr-only']}>{beatmapError}</span>
                      <FieldErrorIcon label={beatmapError} />
                    </>
                  ) : null}
                </span>
              )
            }}
          </form.Field>
        </div>

        <div className={styles['filter-controls']}>
          <RangeFields label="Star" icon={<Star strokeWidth={3} />} min={values.minSr} max={values.maxSr} setMin={(value) => updateRange('minSr', value)} setMax={(value) => updateRange('maxSr', value)} />
          <RangeFields label="BPM" icon={<Metronome strokeWidth={3} />} min={values.minBpm} max={values.maxBpm} setMin={(value) => updateRange('minBpm', value)} setMax={(value) => updateRange('maxBpm', value)} />
          <RangeFields label="Length" icon={<Clock strokeWidth={3} />} min={values.minLength} max={values.maxLength} setMin={(value) => updateRange('minLength', value)} setMax={(value) => updateRange('maxLength', value)} />

          <FilterSelect
            label="Date window"
            icon={<CalendarDays strokeWidth={3} />}
            value={values.dateWindow === 'all_time' ? '' : values.dateWindow}
            options={dateWindowOptions}
            onValueChange={(value) => {
              const dateWindow = value as RecommendFormValues['dateWindow']
              form.setFieldValue('dateWindow', dateWindow)
              onSelectChange({ ...form.state.values, dateWindow })
            }}
          />

          <FilterSelect
            label="Status"
            icon={<Tag strokeWidth={3} />}
            value={values.status}
            options={statusOptions}
            onValueChange={(status) => {
              form.setFieldValue('status', status)
              onSelectChange({ ...form.state.values, status })
            }}
          />

          <button className={styles['ghost-button']} type="button" onClick={resetForm}>
            <RotateCcw />
            <span className={styles['sr-only']}>Reset</span>
          </button>
        </div>
      </div>
    </form>
  )
}

type FilterSelectProps = {
  label: string
  icon: ReactNode
  value: string
  options: ReadonlyArray<{ value: string | null; label: string }>
  onValueChange: (value: string) => void
}

function FilterSelect({ label, icon, value, options, onValueChange }: FilterSelectProps) {
  return (
    <Select.Root
      items={options}
      value={value || null}
      highlightItemOnHover={false}
      onValueChange={(nextValue) => onValueChange(nextValue ?? '')}
    >
      <Select.Trigger className={`${styles.field} ${styles['select-field']}`} aria-label={label}>
        <span aria-hidden="true">{icon}</span>
        <Select.Value className={styles['select-value']} />
        <Select.Icon className={styles['select-icon']}>
          <ChevronDown />
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

function FieldErrorIcon({ label }: { label: string }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger className={styles['field-error-icon']} type="button" delay={0} aria-label={label}>
        <XCircle />
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Positioner className={styles['field-tooltip-positioner']} side="top" align="end" sideOffset={8}>
          <Tooltip.Popup className={styles['field-tooltip']}>{label}</Tooltip.Popup>
        </Tooltip.Positioner>
      </Tooltip.Portal>
    </Tooltip.Root>
  )
}
