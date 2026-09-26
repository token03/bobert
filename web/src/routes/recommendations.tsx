import { createFileRoute, stripSearchParams } from '@tanstack/react-router'
import { RecommendPage } from '../features/recommend/RecommendPage'
import { defaultSearch, recommendSearchSchema } from '../features/recommend/filters'

export const Route = createFileRoute('/recommendations')({
  validateSearch: recommendSearchSchema,
  search: {
    middlewares: [stripSearchParams(defaultSearch)],
  },
  component: RecommendPage,
})
