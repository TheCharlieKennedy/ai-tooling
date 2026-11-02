# Copyright 2026 ai-tooling contributors
# SPDX-License-Identifier: Apache-2.0
"""Backend provider implementations."""

from __future__ import annotations

import logging
import shutil
import subprocess  # nosec B404
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None  # type: ignore[assignment]

try:
    import openai as _openai_sdk
except ImportError:
    _openai_sdk = None  # type: ignore[assignment]


_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 16.0
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 529})


class ProviderUnavailableError(Exception):
    """Raised when a call is attempted but the backend is not usable."""


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and status in _TRANSIENT_STATUS_CODES:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    cls_name = type(exc).__name__.lower()
    return any(k in cls_name for k in ("timeout", "ratelimit", "rate_limit", "connection"))


def _retry_with_backoff(fn: Any, *, label: str = "LLM") -> Any:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == _MAX_RETRIES - 1:
                raise
            delay = min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)
            logger.warning(
                "%s transient error (attempt %d/%d), retrying in %.1fs: %s",
                label,
                attempt + 1,
                _MAX_RETRIES,
                delay,
                exc,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


class _BaseProvider(ABC):
    _model: str
    _client: Any | None
    _timeout_seconds: int
    _label: str

    @property
    def available(self) -> bool:
        return self._client is not None

    def call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        if not self.available:
            raise ProviderUnavailableError(self._unavailable_message())
        return _retry_with_backoff(
            lambda: self._call_backend(prompt, max_tokens),
            label=self._label,
        )

    @abstractmethod
    def _call_backend(self, prompt: str, max_tokens: int) -> str: ...

    @abstractmethod
    def _unavailable_message(self) -> str: ...


class AnthropicProvider(_BaseProvider):
    _label = "anthropic"

    def __init__(self, *, model: str, api_key: str, base_url: str | None, timeout_seconds: int) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        if _anthropic_sdk is None:
            self._client = None
        elif not api_key:
            self._client = None
        else:
            kwargs: dict[str, Any] = {"api_key": api_key, "timeout": float(timeout_seconds)}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = _anthropic_sdk.Anthropic(**kwargs)

    def _call_backend(self, prompt: str, max_tokens: int) -> str:
        assert self._client is not None  # noqa: S101
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        block = response.content[0]
        if not hasattr(block, "text"):
            raise TypeError(f"Expected TextBlock, got {type(block).__name__}")
        return block.text

    def _unavailable_message(self) -> str:
        return "anthropic SDK not installed or no API key configured"


class OpenAICompatibleProvider(_BaseProvider):
    _label = "openai"

    def __init__(self, *, model: str, api_key: str, base_url: str | None, timeout_seconds: int) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        if _openai_sdk is None:
            self._client = None
        elif not api_key:
            self._client = None
        else:
            kwargs: dict[str, Any] = {"api_key": api_key, "timeout": float(timeout_seconds)}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = _openai_sdk.OpenAI(**kwargs)

    def _call_backend(self, prompt: str, max_tokens: int) -> str:
        assert self._client is not None  # noqa: S101
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if content is None:
            raise TypeError("OpenAI response contained no content")
        return content

    def _unavailable_message(self) -> str:
        return "openai SDK not installed or no API key configured"


class ClaudeCliProvider(_BaseProvider):
    _label = "claude-cli"

    def __init__(self, *, timeout_seconds: int) -> None:
        self._model = "claude-cli"
        self._timeout_seconds = timeout_seconds
        self._cli_path: str | None = shutil.which("claude")
        self._client: Any | None = self._cli_path

    @property
    def available(self) -> bool:
        return self._cli_path is not None

    def _call_backend(self, prompt: str, max_tokens: int) -> str:
        assert self._cli_path is not None  # noqa: S101
        cmd = [self._cli_path, "-p", "--output-format", "text", "--max-turns", "1", "--allowedTools", ""]
        try:
            result = subprocess.run(  # nosec B603
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProviderUnavailableError(f"claude CLI timed out after {self._timeout_seconds}s") from exc
        if result.returncode != 0:
            raise ProviderUnavailableError(f"claude CLI exited {result.returncode}: {result.stderr.strip()[:200]}")
        return result.stdout.strip()

    def _unavailable_message(self) -> str:
        return "claude CLI not found on PATH"


__all__ = [
    "AnthropicProvider",
    "ClaudeCliProvider",
    "OpenAICompatibleProvider",
    "ProviderUnavailableError",
]
