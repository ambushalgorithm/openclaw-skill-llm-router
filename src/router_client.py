"""Internal router client.

Instead of shelling out to an external `llm-router` repo, this module
calls the embedded router logic in :mod:`router_core`.
"""

from __future__ import annotations

from typing import Any, Dict

from . import router_core


class RouterError(RuntimeError):
    pass


def route(category: str, *, variant: str | None = None, estimated_cost_usd: float | None = None, prompt: str | None = None) -> Dict[str, Any]:
    """Route a task using the embedded OpenClaw-mode router logic."""

    task: Dict[str, Any] = {
        "mode": "OpenClaw",
        "category": category,
        "variant": variant or "Default",
        "prompt": prompt or "",  # may contain explicit Model=... hints
        "meta": {},
    }
    if estimated_cost_usd is not None:
        task["meta"]["estimated_cost_usd"] = float(estimated_cost_usd)

    result = router_core.route_openclaw(task)

    if result.get("status") != "ok":
        # Surface routing errors as exceptions so callers can decide how
        # to handle them (fallback, user-facing message, etc.).
        raise RouterError(str(result))

    return result
