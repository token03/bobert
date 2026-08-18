import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { defaultFilters, recommendFormSchema } from './filters'
import type { RecommendFormValues } from './filters'

export function useRecommendForm() {
  return useForm<RecommendFormValues>({
    resolver: zodResolver(recommendFormSchema),
    defaultValues: defaultFilters,
  })
}
