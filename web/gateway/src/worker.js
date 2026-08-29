function rateLimited(retryAfter, reason) {
  return Response.json(
    { detail: "Too many recommendation requests" },
    {
      status: 429,
      headers: {
        "Retry-After": String(retryAfter),
        "X-Bobert-RateLimit": reason,
      },
    }
  )
}

export default {
  async fetch(request, env) {
    const incoming = new URL(request.url)

    if (request.method === "POST" && incoming.pathname === "/api/recommend") {
      const ip =
        request.headers.get("CF-Connecting-IP") ??
        request.headers.get("X-Real-IP")

      if (ip) {
        const burst = await env.RECOMMEND_BURST.limit({ key: `recommend:${ip}` })
        if (!burst.success) {
          return rateLimited(10, "burst")
        }

        const sustained = await env.RECOMMEND_SUSTAINED.limit({
          key: `recommend:${ip}`,
        })
        if (!sustained.success) {
          return rateLimited(60, "sustained")
        }
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

    return env.API.fetch(new Request(target, init))
  },
}
