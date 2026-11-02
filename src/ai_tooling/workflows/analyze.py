# Copyright 2026 ai-tooling contributors
# SPDX-License-Identifier: Apache-2.0
"""Analyze workflow — single prompt + evidence bundle → str.

This is the nfr-review workflow. A caller provides a prompt describing what
to look for and a serialized evidence bundle; the LLM returns an analysis.
"""

from __future__ import annotations

from ai_tooling.protocols import LlmClient
from ai_tooling.response import serialize_bundle


def run_analysis(
    client: LlmClient,
    prompt: str,
    evidence: list[dict],
    *,
    max_tokens: int = 1024,
    max_bundle_bytes: int = 8192,
) -> str:
    """Send *prompt* and serialized *evidence* to *client* and return the response."""
    bundle = serialize_bundle(evidence, max_bytes=max_bundle_bytes)
    combined = prompt + "\n\n" + bundle
    return client.call(combined, max_tokens=max_tokens)


__all__ = ["run_analysis"]
