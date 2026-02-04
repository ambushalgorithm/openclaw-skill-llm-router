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

    For now, this is a stub that always returns NotImplementedBackend.
    Actual instances should dispatch based on fields like `provider`
    and/or `backend` once concrete adapters are added.
    """

    # TODO: wire up real dispatch logic once we know the router's schema.
    return NotImplementedBackend()
