const internal = /^\/api\/(players|users|beatmaps)\//
const limited = new Set(["/api/recommend", "/api/me/groups"])

function rateLimited(retryAfter) {
  return Response.json(
    { detail: "Too many requests" },
    { status: 429, headers: { "Retry-After": String(retryAfter) } }
  )
}

export default {
  async fetch(request, env) {
    const incoming = new URL(request.url)

    if (internal.test(incoming.pathname)) {
      return Response.json({ detail: "Not Found" }, { status: 404 })
    }

    const ip = request.headers.get("CF-Connecting-IP")
    if (ip && limited.has(incoming.pathname)) {
      if (!(await env.API_BURST.limit({ key: ip })).success) {
        return rateLimited(10)
      }
      if (!(await env.API_SUSTAINED.limit({ key: ip })).success) {
        return rateLimited(60)
      }
    }

    const target = new URL(
      `http://localhost:8000${incoming.pathname}${incoming.search}`
    )

    const init = {
      method: request.method,
      headers: request.headers,
      redirect: "manual",
    }

    if (request.method !== "GET" && request.method !== "HEAD") {
      init.body = request.body
    }

    const started = Date.now()
    const upstream = await env.API.fetch(new Request(target, init))
    const response = new Response(upstream.body, upstream)
    const user = response.headers.get("X-Bobert-User")
    response.headers.delete("X-Bobert-User")
    const separator = user?.indexOf(":") ?? -1
    console.log({
      method: request.method,
      path: incoming.pathname,
      status: response.status,
      ms: Date.now() - started,
      ip: request.headers.get("CF-Connecting-IP"),
      country: request.cf?.country,
      user_id: separator > 0 ? Number(user.slice(0, separator)) : null,
      username: separator > 0 ? user.slice(separator + 1) : null,
      request_id: response.headers.get("X-Request-ID"),
    })
    return response
  },
}
