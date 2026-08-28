import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { experimental_createQueryPersister } from '@tanstack/query-persist-client-core'
import { RouterProvider } from '@tanstack/react-router'
import './index.css'
import { router } from './router'

const persister = experimental_createQueryPersister({
  storage: window.sessionStorage,
  prefix: 'bobert:query',
  maxAge: 24 * 60 * 60_000,
  refetchOnRestore: false,
})

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      gcTime: 24 * 60 * 60_000,
      persister: persister.persisterFn,
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
)
