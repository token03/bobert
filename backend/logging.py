from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from inspect import iscoroutinefunction, signature
from typing import Any, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


def configure_logging() -> None:
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if os.getenv("APP_ENV") == "production"
        else structlog.dev.ConsoleRenderer()
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
        force=True,
    )

    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str):
    return structlog.get_logger(name)


def request_client_ip(request: Request) -> str:
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@contextmanager
def timed(event: str, **fields: Any):
    log = get_logger("bobert.timing")
    start = time.perf_counter()
    try:
        yield
    except Exception:
        log.exception(
            f"{event}.error",
            **fields,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        raise
    else:
        log.info(
            event,
            **fields,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )


@asynccontextmanager
async def async_timed(event: str, **fields: Any):
    log = get_logger("bobert.timing")
    start = time.perf_counter()
    try:
        yield
    except Exception:
        log.exception(
            f"{event}.error",
            **fields,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        raise
    else:
        log.info(
            event,
            **fields,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )


def timed_call(event: str, fields: tuple[str, ...] = ()):
    def decorator(func: Callable[..., Any]):
        func_signature = signature(func)

        def log_fields(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
            bound = func_signature.bind_partial(*args, **kwargs)
            return {
                name: bound.arguments[name]
                for name in fields
                if name in bound.arguments
            }

        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any):
                async with async_timed(event, **log_fields(args, kwargs)):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with timed(event, **log_fields(args, kwargs)):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request_client_ip(request),
        )

        log = get_logger("bobert.api")
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "request.error",
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )
            structlog.contextvars.clear_contextvars()
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(duration_ms)
        log.info(
            "request.complete",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        structlog.contextvars.clear_contextvars()
        return response
