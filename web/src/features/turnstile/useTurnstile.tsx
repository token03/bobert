import { useCallback, useMemo, useRef } from 'react'
import { Turnstile, type TurnstileInstance } from '@marsidev/react-turnstile'

export const turnstileSiteKey = import.meta.env.VITE_TURNSTILE_SITE_KEY ?? ''

export function useTurnstile() {
  const turnstileRef = useRef<TurnstileInstance | null>(null)
  const tokenRef = useRef<string | null>(null)
  const enabled = Boolean(turnstileSiteKey)

  const clearToken = useCallback(() => {
    tokenRef.current = null
  }, [])

  const getToken = useCallback(async () => {
    if (!enabled) {
      return ''
    }

    const existingToken = tokenRef.current ?? turnstileRef.current?.getResponse()
    if (existingToken) {
      tokenRef.current = null
      return existingToken
    }

    const nextToken = await turnstileRef.current?.getResponsePromise()
    if (!nextToken) {
      throw new Error('Turnstile is not ready. Please try again.')
    }

    tokenRef.current = null
    return nextToken
  }, [enabled])

  const reset = useCallback(() => {
    tokenRef.current = null
    turnstileRef.current?.reset()
  }, [])

  const widget = useMemo(() => {
    if (!enabled) {
      return null
    }

    return (
      <Turnstile
        ref={turnstileRef}
        siteKey={turnstileSiteKey}
        options={{
          size: 'invisible',
          execution: 'render',
          refreshExpired: 'auto',
          retry: 'auto',
          responseField: false,
        }}
        onSuccess={(token) => {
          tokenRef.current = token
        }}
        onExpire={clearToken}
        onError={clearToken}
        onTimeout={clearToken}
      />
    )
  }, [clearToken, enabled])

  return {
    enabled,
    getToken,
    reset,
    widget,
  }
}
