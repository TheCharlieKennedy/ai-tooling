# Copyright 2026 ai-tooling contributors
# SPDX-License-Identifier: Apache-2.0
"""Provider configuration model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider: Literal["anthropic", "openai", "claude-cli"] = "anthropic"
    model: str = "claude-opus-4-5"
    base_url: str | None = None
    api_key_env_var: str = "ANTHROPIC_API_KEY"
    timeout_seconds: int = 120

    def resolve(self) -> ProviderConfig:
        """Return a copy with env-var overrides applied."""
        import os

        overrides: dict[str, object] = {}
        if (p := os.environ.get("AI_TOOLING_PROVIDER")):
            overrides["provider"] = p
        if (m := os.environ.get("AI_TOOLING_MODEL")):
            overrides["model"] = m
        if (u := os.environ.get("AI_TOOLING_BASE_URL")):
            overrides["base_url"] = u
        return self.model_copy(update=overrides) if overrides else self


__all__ = ["ProviderConfig"]
