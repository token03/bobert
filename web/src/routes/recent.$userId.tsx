import { createFileRoute, stripSearchParams } from '@tanstack/react-router'
import { PlaysPage } from '../features/recommend/PlaysPage'
import { defaultPlaysSearch, playsParams, playsSearchSchema } from '../features/recommend/plays'

export const Route = createFileRoute('/recent/$userId')({
  params: playsParams,
  validateSearch: playsSearchSchema,
  search: {
    middlewares: [stripSearchParams(defaultPlaysSearch)],
  },
  component: function RecentPlaysRoute() {
    const { userId } = Route.useParams()
    return <PlaysPage key={userId} kind="recent" userId={userId} search={Route.useSearch()} />
  },
})
