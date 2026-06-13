from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from rag.config import DEFAULT_RETRY_ATTEMPTS, DEFAULT_RETRY_BACKOFF_SECONDS


class RetryExhausted(RuntimeError):
    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class ComponentFallback(RuntimeError):
    def __init__(self, component: str, reason: str, error: str, attempts: int) -> None:
        super().__init__(error)
        self.component = component
        self.reason = reason
        self.error = error
        self.attempts = attempts


def call_with_retries(
    operation,
    *,
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
):
    attempts = max(1, attempts)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation(), attempt
        except ComponentFallback:
            raise
        except Exception as exc:  # noqa: BLE001 - component adapters normalize failure in trace.
            last_exc = exc
            if attempt < attempts and backoff_seconds > 0:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    message = str(last_exc) if last_exc else "operation failed"
    raise RetryExhausted(message, attempts) from last_exc


def request_json(
    method: str,
    url: str,
    body: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    ok_statuses: tuple[int, ...] = (200,),
) -> dict:
    request_headers = {
        "Content-Type": "application/json",
        **(headers or {}),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            status = getattr(response, "status", 200)
            if status not in ok_statuses:
                raise RuntimeError(f"API request failed for {method} {url}: HTTP {status}")
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed for {method} {url}: HTTP {exc.code} {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed for {method} {url}: {format_transport_error(url, exc.reason)}") from exc


def post_json(url: str, body: dict[str, object], headers: dict[str, str] | None = None) -> dict:
    # During Phase 1, tests patch run_pipeline.request_json through the shim.
    # Keep that patch point alive while the transport helper lives in rag.http.
    shim = sys.modules.get("run_pipeline")
    request_json_func = getattr(shim, "request_json", request_json) if shim is not None else request_json
    return request_json_func("POST", url, body=body, headers=headers)


def format_transport_error(url: str, reason: object) -> str:
    message = str(reason)
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme == "https"
        and parsed.hostname
        and parsed.hostname.endswith(".cloud.qdrant.io")
        and parsed.port is None
    ):
        message = (
            f"{message}. Qdrant Cloud REST endpoints require port 6333; "
            f"set QDRANT_URL to https://{parsed.hostname}:6333"
        )
    elif "UNEXPECTED_EOF_WHILE_READING" in message:
        message = (
            f"{message}. This usually means the URL scheme or port does not match the service TLS mode. "
            "For local Qdrant use http://localhost:6333; for Qdrant Cloud use https://<cluster>.cloud.qdrant.io:6333."
        )
    return message
