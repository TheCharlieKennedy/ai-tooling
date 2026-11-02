# Copyright 2026 ai-tooling contributors
# SPDX-License-Identifier: Apache-2.0
"""Prompted workflow — system prompt + user message."""

from __future__ import annotations

from ai_tooling.protocols import LlmClient


def run_prompted(
    client: LlmClient,
    system_prompt: str,
    user_message: str,
    *,
    max_tokens: int = 1024,
) -> str:
    """Combine *system_prompt* and *user_message* and send to *client*."""
    combined = f"System: {system_prompt}\n\nUser: {user_message}"
    return client.call(combined, max_tokens=max_tokens)


__all__ = ["run_prompted"]
