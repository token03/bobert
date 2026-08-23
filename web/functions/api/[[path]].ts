interface Env {
  GATEWAY: {
    fetch(request: Request): Promise<Response>
  }
}

interface PagesContext {
  request: Request
  env: Env
}

export function onRequest(context: PagesContext) {
  return context.env.GATEWAY.fetch(context.request)
}
