# Copyright 2026 ai-tooling contributors
# SPDX-License-Identifier: Apache-2.0
"""Utilities for working with LLM responses."""

from __future__ import annotations

import json
import re


def extract_json(text: str, *, expect: str = "object") -> dict | list | None:
    """Extract a JSON object or array from an LLM response.

    Uses a 3-stage strategy: direct parse, markdown fence, bare delimiter.
    Returns None if no valid JSON of the expected type can be found.

    *expect*: "object" for dict, "array" for list, "any" for either.
    """
    _ok_types: tuple[type, ...] = (
        (dict,) if expect == "object" else (list,) if expect == "array" else (dict, list)
    )

    def _check(val: object) -> dict | list | None:
        return val if isinstance(val, _ok_types) else None  # type: ignore[return-value]

    stripped = text.strip()

    try:
        if (result := _check(json.loads(stripped))) is not None:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", stripped, re.DOTALL)
    if fence_match:
        try:
            if (result := _check(json.loads(fence_match.group(1)))) is not None:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    open_char = "{" if expect == "object" else "[" if expect == "array" else None
    if open_char:
        close_char = "}" if open_char == "{" else "]"
        try:
            start = stripped.index(open_char)
            end = stripped.rindex(close_char) + 1
            if (result := _check(json.loads(stripped[start:end]))) is not None:
                return result
        except (ValueError, json.JSONDecodeError):
            pass
    else:
        for oc, cc in ("{}", "[]"):
            try:
                start = stripped.index(oc)
                end = stripped.rindex(cc) + 1
                if (result := _check(json.loads(stripped[start:end]))) is not None:
                    return result
            except (ValueError, json.JSONDecodeError):
                pass

    return None


def serialize_bundle(items: list[dict], max_bytes: int = 8192) -> str:
    """Serialize a list of evidence dicts to JSON, truncating to fit *max_bytes*.

    Items are dropped from the end until the serialized output fits.
    """
    items = list(items)
    while True:
        serialized = json.dumps(items, separators=(",", ":"))
        if len(serialized.encode()) <= max_bytes:
            return serialized
        if not items:
            return "[]"
        items.pop()


__all__ = ["extract_json", "serialize_bundle"]
