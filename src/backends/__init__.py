"""Backend adapter registry for openclaw-skill-llm-router.

This module exposes a single function:

    get_backend_for_router_result(router_result) -> Backend

A Backend is any object with a `run(router_result, messages, options)`
method that returns a dict containing at least `{"content": "..."}`.

To add support for a new backend, implement a small adapter and register
it here. Keep adapters focused and stateless.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .. import types as _types
from .claude_cli import ClaudeCliBackend


class Backend:
    """Abstract backend interface.

    Concrete backends should implement `run`.
    """

    def run(self, *, router_result: Mapping[str, Any], messages: list[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class NotImplementedBackend(Backend):
    def run(self, *, router_result: Mapping[str, Any], messages: list[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
        rr = _types.coerce_router_result(router_result)
        raise RuntimeError(
            f"No backend adapter implemented for provider={rr.provider!r}, backend={rr.backend!r}."
        )


def get_backend_for_router_result(router_result: Mapping[str, Any]) -> Backend:
    """Return an appropriate Backend instance for the given router result.

    Dispatch is based on the router's `provider` / `backend` fields.
    For now we support anthropic via the Claude CLI and fall back to a
    "not implemented" backend otherwise.
    """

    rr = _types.coerce_router_result(router_result)
    backend_key = (rr.provider or "").lower()
    backend_hint = (rr.backend or "").lower() if rr.backend is not None else ""

    # Anthropic via Claude CLI
    if backend_key == "anthropic" or backend_hint == "claude-cli":
        return ClaudeCliBackend()

    # TODO: add more backends (openai-cli, http, etc.) as needed.
    return NotImplementedBackend()
