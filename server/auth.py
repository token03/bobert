import base64
import hmac
import json
import os
import time
from http.cookies import SimpleCookie
from typing import Any

SESSION_COOKIE = "bobert_session"
STATE_COOKIE = "bobert_oauth"
SESSION_SECONDS = 30 * 24 * 3600
STATE_SECONDS = 600
SECRET = os.environ["SESSION_SECRET"].encode()
if not SECRET:
    raise RuntimeError("SESSION_SECRET is required")


def _encode(data: bytes) -> bytes:
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def _signature(body: bytes) -> bytes:
    return _encode(hmac.digest(SECRET, body, "sha256"))


def sign(payload: dict[str, Any], seconds: int) -> str:
    claims = {**payload, "exp": int(time.time()) + seconds}
    body = _encode(json.dumps(claims, separators=(",", ":")).encode())
    return f"{body.decode()}.{_signature(body).decode()}"


def verify(token: str | None) -> dict[str, Any] | None:
    if not token or token.count(".") != 1:
        return None
    body, signature = token.encode().split(b".")
    if not hmac.compare_digest(signature, _signature(body)):
        return None
    claims = json.loads(base64.urlsafe_b64decode(body + b"=" * (-len(body) % 4)))
    if claims.pop("exp") <= time.time():
        return None
    return claims


def cookie_session(header: str | None) -> dict[str, Any] | None:
    if not header:
        return None
    morsel = SimpleCookie(header).get(SESSION_COOKIE)
    return verify(morsel.value) if morsel else None
