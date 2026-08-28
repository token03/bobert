import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { recommendFormSchema } from './filters'
import type { RecommendFormValues } from './filters'

export function useRecommendForm(defaultValues: RecommendFormValues) {
  return useForm<RecommendFormValues>({
    resolver: zodResolver(recommendFormSchema),
    defaultValues,
  })
}
