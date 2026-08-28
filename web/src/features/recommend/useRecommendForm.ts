import { useForm } from '@tanstack/react-form'
import { recommendFormSchema } from './filters'
import type { RecommendFormValues } from './filters'

export function useRecommendForm(defaultValues: RecommendFormValues, onSubmit: (values: RecommendFormValues) => Promise<void>) {
  return useForm({
    defaultValues,
    validators: {
      onSubmit: recommendFormSchema,
    },
    onSubmit: ({ value }) => onSubmit(value),
  })
}
