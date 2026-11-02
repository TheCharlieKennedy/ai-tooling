# Copyright 2026 ai-tooling contributors
# SPDX-License-Identifier: Apache-2.0
"""Session factory — the single entry point for any caller."""

from __future__ import annotations

import logging
import os

from ai_tooling.config import ProviderConfig
from ai_tooling.providers import AnthropicProvider, ClaudeCliProvider, OpenAICompatibleProvider
from ai_tooling.protocols import LlmClient

logger = logging.getLogger(__name__)


def create_session(config: ProviderConfig | None = None) -> LlmClient:
    """Create an LLM session from a ProviderConfig.

    When *config* is None, uses default ProviderConfig values with env-var
    overrides applied. The returned object satisfies the LlmClient protocol.
    """
    if config is None:
        config = ProviderConfig()
    resolved = config.resolve()

    if resolved.provider == "claude-cli":
        return ClaudeCliProvider(timeout_seconds=resolved.timeout_seconds)

    api_key = os.environ.get(resolved.api_key_env_var, "").strip()
    if not api_key:
        logger.debug(
            "No API key found (%s not set); LLM calls will be unavailable",
            resolved.api_key_env_var,
        )

    if resolved.provider == "openai":
        return OpenAICompatibleProvider(
            model=resolved.model,
            api_key=api_key,
            base_url=resolved.base_url,
            timeout_seconds=resolved.timeout_seconds,
        )

    return AnthropicProvider(
        model=resolved.model,
        api_key=api_key,
        base_url=resolved.base_url,
        timeout_seconds=resolved.timeout_seconds,
    )


__all__ = ["create_session"]
