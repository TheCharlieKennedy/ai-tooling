# Copyright 2026 ai-tooling contributors
# SPDX-License-Identifier: Apache-2.0
"""Core protocol — the contract any caller depends on."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LlmClient(Protocol):
    """Pluggable LLM backend. Any provider implementation must satisfy this contract."""

    @property
    def available(self) -> bool:
        """Whether this backend is configured and ready to make calls."""
        ...

    def call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        """Send *prompt* to the LLM and return the response text."""
        ...


__all__ = ["LlmClient"]
