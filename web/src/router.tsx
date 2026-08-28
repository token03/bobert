import { createRootRoute, createRoute, createRouter, redirect, stringifySearchWith, stripSearchParams } from '@tanstack/react-router'
import App from './App'
import { RecommendPage } from './features/recommend/RecommendPage'
import { defaultFilters, recommendSearchSchema } from './features/recommend/filters'

const rootRoute = createRootRoute({ component: App })

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  validateSearch: recommendSearchSchema,
  beforeLoad: ({ search }) => {
    throw redirect({ to: '/recommendations', search, replace: true })
  },
})

const recommendationsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/recommendations',
  validateSearch: recommendSearchSchema,
  search: {
    middlewares: [stripSearchParams(defaultFilters)],
  },
  component: RecommendPage,
})

const routeTree = rootRoute.addChildren([indexRoute, recommendationsRoute])

export const router = createRouter({
  routeTree,
  stringifySearch: stringifySearchWith(JSON.stringify),
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
