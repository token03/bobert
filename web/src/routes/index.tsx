import { createFileRoute, redirect } from '@tanstack/react-router'
import { recommendSearchSchema } from '../features/recommend/filters'

export const Route = createFileRoute('/')({
  validateSearch: recommendSearchSchema,
  beforeLoad: ({ search }) => {
    throw redirect({ to: '/recommendations', search, replace: true })
  },
})
