from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class JsonCacheStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._data: dict[str, Any] = {}
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}

    @staticmethod
    def make_key(namespace: str, payload: str) -> str:
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, namespace: str, payload: str) -> Any | None:
        key = self.make_key(namespace, payload)
        record = self._data.get(key)
        if not record:
            return None
        return record.get("value")

    def set(self, namespace: str, payload: str, value: Any) -> None:
        key = self.make_key(namespace, payload)
        self._data[key] = {"created_at": time.time(), "value": value}
        self.flush()

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

