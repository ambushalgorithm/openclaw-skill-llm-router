"""Shared types and normalization helpers for openclaw-skill-llm-router.

This module defines the minimal shapes we care about for the router
result and the backend response, and provides a `normalize_response`
function that produces the skill's public output schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass
class RouterResult:
    provider: str
    model: str
    backend: str | None = None
    raw: Dict[str, Any] | None = None


def coerce_router_result(data: Mapping[str, Any]) -> RouterResult:
    provider = str(data.get("provider", "")).strip()
    model = str(data.get("model", "")).strip()
    backend = data.get("backend")

    if not provider or not model:
        raise ValueError("Router result must include non-empty 'provider' and 'model'.")

    return RouterResult(
        provider=provider,
        model=model,
        backend=str(backend) if backend is not None else None,
        raw=dict(data),
    )


def normalize_response(*, router_result: Mapping[str, Any] | RouterResult, backend_response: Mapping[str, Any], category: str) -> Dict[str, Any]:
    """Produce the normalized JSON response for the skill.

    The backend adapter is expected to return at least:

        {
          "content": "...",                  # assistant text
          "role": "assistant"                # optional, defaults to "assistant"
        }

    Additional backend-specific fields are passed through under
    `raw.backend`.
    """

    if not isinstance(router_result, RouterResult):
        router_result = coerce_router_result(router_result)

    content = backend_response.get("content")
    if not isinstance(content, str):
        raise ValueError("backend_response['content'] must be a string.")

    role = backend_response.get("role") or "assistant"

    raw_backend = dict(backend_response)
    # Remove top-level fields we promote so callers don't see duplicates.
    raw_backend.pop("content", None)
    raw_backend.pop("role", None)

    return {
        "provider": router_result.provider,
        "model": router_result.model,
        "category": category,
        "response": {
            "role": role,
            "content": content,
        },
        "raw": {
            "router": router_result.raw or {},
            "backend": raw_backend,
        },
    }
