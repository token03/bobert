import { useEffect, useEffectEvent } from 'react'
import { Tooltip } from '@base-ui/react/tooltip'
import { useStore } from '@tanstack/react-form'
import { CalendarDays, Clock, Loader, Metronome, RotateCcw, Search, Star, Tag, XCircle } from 'lucide-react'
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

          <label className={`${styles.field} ${styles['select-field']}`}>
            <span aria-hidden="true">
              <CalendarDays strokeWidth={3} />
            </span>
            <select
              aria-label="Date window"
              value={values.dateWindow}
              onChange={(event) => {
                const dateWindow = event.target.value as RecommendFormValues['dateWindow']
                form.setFieldValue('dateWindow', dateWindow)
                onSelectChange({ ...form.state.values, dateWindow })
              }}
            >
              <option value="">All time</option>
              <option value="last_week">Last week</option>
              <option value="last_month">Last month</option>
              <option value="last_3_months">Last 3 months</option>
              <option value="last_6_months">Last 6 months</option>
              <option value="last_year">Last year</option>
              <option value="last_2_years">Last 2 years</option>
              <option value="last_5_years">Last 5 years</option>
            </select>
          </label>

          <label className={`${styles.field} ${styles['select-field']}`}>
            <span aria-hidden="true">
              <Tag strokeWidth={3} />
            </span>
            <select
              aria-label="Status"
              value={values.status}
              onChange={(event) => {
                form.setFieldValue('status', event.target.value)
                onSelectChange({ ...form.state.values, status: event.target.value })
              }}
            >
              <option value="">Any</option>
              <option value="ranked">Ranked</option>
              <option value="loved">Loved</option>
              <option value="unranked">Unranked</option>
            </select>
          </label>

          <button className={styles['ghost-button']} type="button" onClick={resetForm}>
            <RotateCcw />
            <span className={styles['sr-only']}>Reset</span>
          </button>
        </div>
      </div>
    </form>
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
