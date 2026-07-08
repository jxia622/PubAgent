from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


def _post_json(url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {body}") from exc


def _parse_json_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


class LLMClient:
    def complete_json(self, *, system: str, user: dict[str, Any]) -> dict[str, Any] | None:
        raise NotImplementedError


class NoopLLMClient(LLMClient):
    def complete_json(self, *, system: str, user: dict[str, Any]) -> dict[str, Any] | None:
        return None


class OpenAIJsonClient(LLMClient):
    def __init__(self, *, api_key: str | None = None, model: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.timeout = timeout

    def complete_json(self, *, system: str, user: dict[str, Any]) -> dict[str, Any] | None:
        if not self.api_key:
            return None
        try:
            payload = _post_json(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                payload={
                    "model": self.model,
                    "instructions": system + "\nReturn only valid JSON.",
                    "input": "Return JSON for this request:\n" + json.dumps(user),
                    "text": {"format": {"type": "json_object"}},
                },
                timeout=self.timeout,
            )
        except Exception:
            return None
        text = payload.get("output_text")
        if not text:
            parts = []
            for item in payload.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"}:
                        parts.append(content.get("text", ""))
            text = "\n".join(parts)
        return _parse_json_text(text) if text else None


class AnthropicJsonClient(LLMClient):
    def __init__(self, *, api_key: str | None = None, model: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        self.timeout = timeout

    def complete_json(self, *, system: str, user: dict[str, Any]) -> dict[str, Any] | None:
        if not self.api_key:
            return None
        try:
            payload = _post_json(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                payload={
                    "model": self.model,
                    "max_tokens": 1600,
                    "system": system + "\nReturn only valid JSON.",
                    "messages": [{"role": "user", "content": "Return JSON for this request:\n" + json.dumps(user)}],
                },
                timeout=self.timeout,
            )
        except Exception:
            return None
        text = "\n".join(part.get("text", "") for part in payload.get("content", []) if part.get("type") == "text")
        return _parse_json_text(text) if text else None


def default_llm_client() -> LLMClient:
    provider = os.getenv("RESEARCH_AGENT_LLM_PROVIDER") or os.getenv("LITERATURE_AGENT_LLM_PROVIDER", "")
    provider = provider.lower()
    if provider == "none":
        return NoopLLMClient()
    if provider == "anthropic":
        return AnthropicJsonClient()
    if provider == "openai":
        return OpenAIJsonClient()
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIJsonClient()
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicJsonClient()
    return NoopLLMClient()
