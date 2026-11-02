# Copyright 2026 ai-tooling contributors
# SPDX-License-Identifier: Apache-2.0
"""Tools workflow — structured tool use.

Provides a lightweight framework for describing tools and dispatching
LLM-driven tool calls. Provider-level tool use APIs (e.g. Anthropic's
native tool_use content blocks) can be wired in via a ToolRunner subclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolDefinition:
    """Describes a tool the LLM can call."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]


class ToolRunner:
    """Runs a prompt against a set of registered tools.

    Base implementation serializes tool definitions into the prompt and
    parses a JSON tool-call response. Subclass and override _execute_call()
    for provider-native tool use.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call the handler for *tool_name* with *arguments*."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name!r}")
        return self._tools[tool_name].handler(**arguments)

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


__all__ = ["ToolDefinition", "ToolRunner"]
