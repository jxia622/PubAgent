from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    text: str
    headers: dict[str, str]

    def json(self) -> dict[str, Any]:
        import json

        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP request failed with status {self.status_code}")


class RetryHttpClient:
    def __init__(self, *, timeout: float = 20.0, max_retries: int = 3, backoff_seconds: float = 0.5) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def get_text(self, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> str:
        response = self._get(url, params=params, headers=headers)
        return response.text

    def get_json(self, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
        response = self._get(url, params=params, headers=headers)
        return response.json()

    def _get(self, url: str, *, params: dict[str, Any] | None, headers: dict[str, str] | None) -> HttpResponse:
        last_error: Exception | None = None
        full_url = url
        if params:
            full_url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(full_url, headers=headers or {})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return HttpResponse(
                        status_code=response.getcode(),
                        text=response.read().decode("utf-8", errors="replace"),
                        headers=dict(response.headers.items()),
                    )
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 429 or exc.code >= 500:
                    retry_after = exc.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else self.backoff_seconds * (2**attempt)
                    time.sleep(delay)
                    continue
                raise
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
                    continue
                break
        if last_error:
            raise last_error
        raise RuntimeError(f"GET failed after retries: {full_url}")
