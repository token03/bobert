FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-default-groups --group serve --no-install-project --no-cache

COPY core ./core
COPY server ./server

RUN chmod -R a+rX core server

RUN useradd --uid 10001 --create-home bobert && mkdir -p /app/cache && chown bobert:bobert /app/cache

USER bobert

CMD ["python", "-m", "server.app"]
