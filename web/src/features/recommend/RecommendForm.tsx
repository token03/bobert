import { useEffect } from 'react'
import { Clock, Loader, Metronome, RotateCcw, Search, Star, Tag, XCircle } from 'lucide-react'
import type { UseFormReturn } from 'react-hook-form'
import { defaultFilters, normalizeBeatmapInput, parseBeatmapId } from './filters'
import type { RecommendFormValues } from './filters'
import { RangeFields } from './RangeFields'

type RecommendFormProps = {
  form: UseFormReturn<RecommendFormValues>
  isLoading: boolean
  onSubmit: (values: RecommendFormValues) => Promise<void>
  onRangeChange: (values: RecommendFormValues) => void
  onStatusChange: (values: RecommendFormValues) => void
  onPasteSearch: (values: RecommendFormValues) => void
  onReset: (values: RecommendFormValues) => void
}

type RangeFieldName = 'minSr' | 'maxSr' | 'minLength' | 'maxLength' | 'minBpm' | 'maxBpm'

export function RecommendForm({ form, isLoading, onSubmit, onRangeChange, onStatusChange, onPasteSearch, onReset }: RecommendFormProps) {
  const values = form.watch()
  const beatmapError = form.formState.errors.beatmapInput?.message
  const submitDisabled = isLoading || form.formState.isSubmitting
  const beatmapInput = form.register('beatmapInput')
  const status = form.register('status')

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

            <label className="field status-field">
              <span aria-hidden="true">
                <Tag strokeWidth={3} />
              </span>
              <select
                {...status}
                onChange={(event) => {
                  status.onChange(event)
                  onStatusChange({ ...form.getValues(), status: event.target.value })
                }}
              >
                <option value="">Any</option>
                <option value="-2">Graveyard</option>
                <option value="-1">WIP</option>
                <option value="0">Pending</option>
                <option value="1">Ranked</option>
                <option value="2">Approved</option>
                <option value="3">Qualified</option>
                <option value="4">Loved</option>
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
