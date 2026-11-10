import { createFileRoute, stripSearchParams } from '@tanstack/react-router'
import { PlaysPage } from '../features/recommend/PlaysPage'
import { defaultPlaysSearch, playsParams, playsSearchSchema } from '../features/recommend/plays'

export const Route = createFileRoute('/top/$userId')({
  params: playsParams,
  validateSearch: playsSearchSchema,
  search: {
    middlewares: [stripSearchParams(defaultPlaysSearch)],
  },
  component: function TopPlaysRoute() {
    const { userId } = Route.useParams()
    return <PlaysPage key={userId} kind="best" userId={userId} search={Route.useSearch()} />
  },
})
