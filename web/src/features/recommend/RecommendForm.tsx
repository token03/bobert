import { useEffect } from 'react'
import { CalendarDays, Clock, Loader, Metronome, RotateCcw, Search, Star, Tag, XCircle } from 'lucide-react'
import type { UseFormReturn } from 'react-hook-form'
import { defaultFilters, normalizeBeatmapInput, parseBeatmapId } from './filters'
import type { RecommendFormValues } from './filters'
import { RangeFields } from './RangeFields'

type RecommendFormProps = {
  form: UseFormReturn<RecommendFormValues>
  isLoading: boolean
  onSubmit: (values: RecommendFormValues) => Promise<void>
  onRangeChange: (values: RecommendFormValues) => void
  onSelectChange: (values: RecommendFormValues) => void
  onPasteSearch: (values: RecommendFormValues) => void
  onReset: (values: RecommendFormValues) => void
}

type RangeFieldName = 'minSr' | 'maxSr' | 'minLength' | 'maxLength' | 'minBpm' | 'maxBpm'

export function RecommendForm({ form, isLoading, onSubmit, onRangeChange, onSelectChange, onPasteSearch, onReset }: RecommendFormProps) {
  const values = form.watch()
  const beatmapError = form.formState.errors.beatmapInput?.message
  const submitDisabled = isLoading || form.formState.isSubmitting
  const beatmapInput = form.register('beatmapInput')
  const status = form.register('status')
  const dateWindow = form.register('dateWindow')

  function resetForm() {
    const nextValues = { ...defaultFilters, beatmapInput: form.getValues('beatmapInput') }
    form.reset(nextValues)
    onReset(nextValues)
  }

  function updateRange(field: RangeFieldName, value: string) {
    form.setValue(field, value)
    onRangeChange({ ...form.getValues(), [field]: value })
  }

  function searchPastedBeatmap(value: string) {
    if (!parseBeatmapId(value)) {
      return false
    }

    const beatmapInput = normalizeBeatmapInput(value)
    const nextValues = { ...form.getValues(), beatmapInput }
    form.clearErrors('beatmapInput')
    form.setValue('beatmapInput', beatmapInput, { shouldValidate: false })
    onPasteSearch(nextValues)
    return true
  }

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

      if (searchPastedBeatmap(event.clipboardData?.getData('text') ?? '')) {
        event.preventDefault()
      }
    }

    window.addEventListener('paste', pasteSearch)
    return () => window.removeEventListener('paste', pasteSearch)
  })

  return (
    <>
      <form className="control-panel" onSubmit={form.handleSubmit(onSubmit)}>
        <div className="primary-controls">
          <label className="field beatmap-field search-field">
            <span className="sr-only">Search</span>
            <span className="input-with-status">
              <span className="search-pill">
                <input
                  required
                  aria-invalid={beatmapError ? 'true' : 'false'}
                  aria-describedby={beatmapError ? 'beatmap-error' : undefined}
                  className={beatmapError ? 'has-field-error' : undefined}
                  {...beatmapInput}
                  onChange={(event) => {
                    form.clearErrors('beatmapInput')
                    beatmapInput.onChange(event)
                  }}
                  onPaste={(event) => {
                    if (searchPastedBeatmap(event.clipboardData.getData('text'))) {
                      event.preventDefault()
                    }
                  }}
                  onBlur={(event) => {
                    beatmapInput.onBlur(event)
                    form.setValue('beatmapInput', normalizeBeatmapInput(event.target.value), { shouldValidate: false })
                  }}
                  placeholder="1872396 or https://osu.ppy.sh/beatmaps/1872396"
                />
                <button className="primary-button search-button" type="submit" disabled={submitDisabled} aria-label="Recommend">
                  {submitDisabled ? <Loader className="spinner-icon" /> : <Search />}
                  <span className="sr-only">Recommend</span>
                </button>
              </span>
              {beatmapError ? <FieldErrorIcon id="beatmap-error" label="Invalid beatmap ID" /> : null}
            </span>
          </label>

          <div className="filter-controls">
            <RangeFields label="Star" icon={<Star strokeWidth={3} />} min={values.minSr} max={values.maxSr} setMin={(value) => updateRange('minSr', value)} setMax={(value) => updateRange('maxSr', value)} />
            <RangeFields label="BPM" icon={<Metronome strokeWidth={3} />} min={values.minBpm} max={values.maxBpm} setMin={(value) => updateRange('minBpm', value)} setMax={(value) => updateRange('maxBpm', value)} />
            <RangeFields label="Length" icon={<Clock strokeWidth={3} />} min={values.minLength} max={values.maxLength} setMin={(value) => updateRange('minLength', value)} setMax={(value) => updateRange('maxLength', value)} />

            <label className="field select-field date-field">
              <span aria-hidden="true">
                <CalendarDays strokeWidth={3} />
              </span>
              <select
                aria-label="Date window"
                {...dateWindow}
                onChange={(event) => {
                  dateWindow.onChange(event)
                  onSelectChange({ ...form.getValues(), dateWindow: event.target.value })
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

            <label className="field select-field status-field">
              <span aria-hidden="true">
                <Tag strokeWidth={3} />
              </span>
              <select
                aria-label="Status"
                {...status}
                onChange={(event) => {
                  status.onChange(event)
                  onSelectChange({ ...form.getValues(), status: event.target.value })
                }}
              >
                <option value="">Any</option>
                <option value="ranked">Ranked</option>
                <option value="loved">Loved</option>
                <option value="unranked">Unranked</option>
              </select>
            </label>

            <button className="ghost-button" type="button" onClick={resetForm}>
              <RotateCcw />
              <span className="sr-only">Reset</span>
            </button>
          </div>
        </div>

      </form>
    </>
  )
}

function FieldErrorIcon({ id, label }: { id: string; label: string }) {
  return (
    <span className="field-error-icon" tabIndex={0} aria-label={label}>
      <XCircle />
      <span id={id} className="field-tooltip" role="tooltip">
        {label}
      </span>
    </span>
  )
}
