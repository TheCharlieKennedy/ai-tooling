# Copyright 2026 ai-tooling contributors
# SPDX-License-Identifier: Apache-2.0
"""Chat workflow — multi-turn conversation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_tooling.protocols import LlmClient


@dataclass
class ChatSession:
    """Maintains conversation history for multi-turn exchanges."""

    client: LlmClient
    history: list[dict[str, str]] = field(default_factory=list)

    def send(self, message: str, *, max_tokens: int = 1024) -> str:
        """Send *message*, append both turns to history, return response."""
        self.history.append({"role": "user", "content": message})
        # Build a single prompt from history for providers that don't support
        # multi-turn natively via the LlmClient.call() interface.
        prompt = _render_history(self.history)
        response = self.client.call(prompt, max_tokens=max_tokens)
        self.history.append({"role": "assistant", "content": response})
        return response

    def reset(self) -> None:
        self.history.clear()


def _render_history(history: list[dict[str, str]]) -> str:
    parts = []
    for turn in history:
        role = turn["role"].capitalize()
        parts.append(f"{role}: {turn['content']}")
    return "\n\n".join(parts)


__all__ = ["ChatSession"]
