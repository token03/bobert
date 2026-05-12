FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1

COPY requirements.txt ./

RUN uv pip install --system --no-cache -r requirements.txt

COPY core ./core
COPY config.yaml ./

COPY backend ./backend

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
