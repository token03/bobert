import { createFileRoute, stripSearchParams } from '@tanstack/react-router'
import { RecommendPage } from '../features/recommend/RecommendPage'
import { defaultFilters, recommendSearchSchema } from '../features/recommend/filters'

export const Route = createFileRoute('/recommendations')({
  validateSearch: recommendSearchSchema,
  search: {
    middlewares: [stripSearchParams(defaultFilters)],
  },
  component: RecommendPage,
})
