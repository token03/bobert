interface Env {
  BOBERT_API_URL: string
}

interface PagesContext {
  request: Request
  env: Env
}

export async function onRequest(context: PagesContext) {
  const backendUrl = context.env.BOBERT_API_URL

  if (!backendUrl) {
    return Response.json({ detail: 'BOBERT_API_URL is not configured' }, { status: 500 })
  }

  const target = new URL('/recommend', backendUrl)
  const headers = new Headers(context.request.headers)

  headers.set('host', target.host)

  const response = await fetch(target, {
    method: context.request.method,
    headers,
    body: context.request.method === 'GET' || context.request.method === 'HEAD' ? undefined : context.request.body,
  })

  return new Response(response.body, {
    status: response.status,
    headers: response.headers,
  })
}
