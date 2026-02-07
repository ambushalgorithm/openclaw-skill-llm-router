#!/usr/bin/env python3
"""Entry point for the openclaw-skill-llm-router.

This is a thin CLI wrapper that:
- Reads a single JSON request from stdin (or an optional file/arg in future).
- Routes to the appropriate LLM using capability-aware router v2 (with legacy fallback).
- Dispatches to the appropriate LLM backend adapter.
- Writes a normalized JSON response to stdout.

The detailed behavior and configuration are documented in SKILL.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional

from . import router_client
from . import router_core
from . import types
from .backends import get_backend_for_router_result
from .openclaw_import import import_openclaw_usage
from .quota_tracker import get_ollama_status, QuotaTracker, reload_ollama_config

# Router v2 imports
from .router_v2 import Router as RouterV2, RoutingRequest as RoutingRequestV2


def read_request() -> Dict[str, Any]:
    """Read a single JSON object from stdin.

    In the future, this could be extended to support an explicit
    `--input` argument or streaming, but for now stdin keeps the
    contract simple for OpenClaw.
    """

    raw = sys.stdin.read()
    if not raw.strip():
        raise SystemExit("No input provided on stdin; expected JSON request.")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON input: {e}") from e
    return data


def _get_view_mode(args: list[str]) -> str:
    """Extract view mode from command line args."""
    if "--real" in args:
        return "real"
    if "--normalized" in args:
        return "normalized"
    return "combined"  # default


def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for English text."""
    if not text:
        return 0
    return len(text) // 4


def _build_prompt_from_messages(messages: list[Dict[str, Any]]) -> str:
    """Build a single prompt string from chat messages."""
    prompt_parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if content:
            prompt_parts.append(f"{role}: {content}")
    return " ".join(prompt_parts)


def _route_with_v2(
    category: str,
    prompt: str,
    messages: list[Dict[str, Any]],
    estimated_cost_usd: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Try to route using router v2 (capability-aware).

    Returns a router result dict compatible with legacy format, or None if v2 fails.
    """
    try:
        # Initialize v2 router (loads policy, catalog, etc.)
        router = RouterV2()

        # Estimate tokens from prompt
        estimated_tokens = _estimate_tokens(prompt)

        # Build routing request
        request = RoutingRequestV2(
            prompt=prompt,
            category_hint=category,
            estimated_tokens=estimated_tokens,
            max_cost_per_1k=estimated_cost_usd * 1000 if estimated_cost_usd else None,
        )

        # Route via v2
        result = router.route(request)

        # Map v2 result to legacy-compatible dict
        # Backends expect: provider, model_id (full format for Ollama, short for Anthropic)
        router_result = {
            "provider": result.provider,
            "model_id": result.model_id,  # e.g., "ollama/kimi-k2.5" or "anthropic/claude-sonnet-4-5"
            "model_name": result.model_name,
            "tier": result.tier,
            "confidence": result.confidence,
            # Include v2 metadata for debugging
            "_v2": {
                "capabilities": sorted(result.capabilities),
                "signals": result.classification_signals,
                "reason": result.reason,
                "candidates_considered": result.candidates_considered,
                "quota_limited": result.quota_limited,
                "policy_applied": result.policy_applied,
            },
        }

        return router_result

    except Exception as e:
        # V2 routing failed - return None to trigger legacy fallback
        # In debug mode, we could log this
        if os.getenv("LLM_ROUTER_DEBUG"):
            print(f"[DEBUG] V2 routing failed: {e}", file=sys.stderr)
        return None


def _route_with_legacy(
    category: str,
    prompt: str,
    estimated_cost_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Route using legacy router (router_client)."""
    return router_client.route(
        category=category,
        estimated_cost_usd=estimated_cost_usd,
        prompt=prompt,
    )


def route_with_fallback(
    category: str,
    prompt: str,
    messages: list[Dict[str, Any]],
    estimated_cost_usd: Optional[float] = None,
    prefer_v2: bool = True,
) -> Dict[str, Any]:
    """Route a request, preferring v2 with fallback to legacy.

    Args:
        category: The routing category (e.g., "Coding", "Brain")
        prompt: The combined prompt text
        messages: Original message list for token estimation
        estimated_cost_usd: Optional cost hint
        prefer_v2: If True, try v2 first; if False, use legacy only

    Returns:
        Router result dict compatible with existing backends
    """
    router_result = None

    if prefer_v2:
        # Try v2 first
        router_result = _route_with_v2(
            category=category,
            prompt=prompt,
            messages=messages,
            estimated_cost_usd=estimated_cost_usd,
        )

    # Fall back to legacy if v2 failed or v2 is disabled
    if router_result is None:
        router_result = _route_with_legacy(
            category=category,
            prompt=prompt,
            estimated_cost_usd=estimated_cost_usd,
        )
        # Mark as legacy for debugging
        router_result["_router_version"] = "legacy"
    else:
        router_result["_router_version"] = "v2"

    return router_result


def main() -> None:
    # Status mode: inspect budgets/usage instead of routing a task
    if any(arg in ("--status", "-s") for arg in sys.argv[1:]):
        view_mode = _get_view_mode(sys.argv[1:])
        summary = router_core.status_summary(view_mode=view_mode)
        json.dump(summary, sys.stdout)
        sys.stdout.write("\n")
        return

    # Quota status mode: check subscription-based quotas (Ollama Cloud, etc.)
    if any(arg in ("--quota", "-q") for arg in sys.argv[1:]):
        ollama_status = get_ollama_status()
        json.dump({"status": "ok", "quotas": {"ollama": ollama_status}}, sys.stdout)
        sys.stdout.write("\n")
        return

    # Adjust quota limits manually (e.g., from dashboard observation)
    if any(arg in ("--adjust-quota",) for arg in sys.argv[1:]):
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        cfg = {}
        if raw.strip():
            try:
                cfg = json.loads(raw)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON input: {e}") from e
        provider = cfg.get("provider", "ollama")
        session_tokens = cfg.get("session_tokens")
        weekly_tokens = cfg.get("weekly_tokens")
        tracker = QuotaTracker()
        tracker.adjust_limits(provider, session_tokens=session_tokens, weekly_tokens=weekly_tokens)
        json.dump({"status": "ok", "provider": provider, "adjusted": True}, sys.stdout)
        sys.stdout.write("\n")
        return

    # Import OpenClaw transcripts into the unified ledger (no LLM calls).
    if any(arg in ("--import-openclaw-usage",) for arg in sys.argv[1:]):
        # Read optional JSON config from stdin.
        # If stdin is a TTY, do not block waiting for EOF.
        raw = ""
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
        cfg = {}
        if raw.strip():
            try:
                cfg = json.loads(raw)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON input: {e}") from e
        category = (cfg.get("category") or "Brain")
        mode = (cfg.get("mode") or router_core.OPENCLAW_MODE)
        max_files = cfg.get("max_files")
        result = import_openclaw_usage(
            category=str(category),
            mode=str(mode),
            max_files=int(max_files) if max_files is not None else None,
        )
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return

    # Unified usage ingestion (for direct/no-router calls or external bookkeeping)
    if any(arg in ("--log-usage", "--ingest-usage") for arg in sys.argv[1:]):
        payload = read_request()
        mode = str(payload.get("mode") or router_core.OPENCLAW_MODE)
        category = str(payload.get("category") or "").strip() or "Uncategorized"
        cost_usd = float(payload.get("cost_usd") or 0.0)
        provider = payload.get("provider")
        model = payload.get("model")
        tokens_in = payload.get("tokens_in")
        tokens_out = payload.get("tokens_out")
        is_estimate = bool(payload.get("is_estimate") or False)
        source = payload.get("source")

        units_other = payload.get("units_other")
        ev = router_core.log_usage_event(
            mode=mode,
            category=category,
            cost_usd=cost_usd,
            provider=provider,
            model=model,
            tokens_in=int(tokens_in) if tokens_in is not None else None,
            tokens_out=int(tokens_out) if tokens_out is not None else None,
            is_estimate=is_estimate,
            source=source,
            units_other=units_other,
        )

        json.dump({"status": "ok", "event": ev.to_dict()}, sys.stdout)
        sys.stdout.write("\n")
        return

    # V2 router test mode (for debugging)
    if any(arg in ("--test-v2",) for arg in sys.argv[1:]):
        # Read prompt from args or stdin
        if len(sys.argv) > 2 and not sys.argv[2].startswith("-"):
            test_prompt = sys.argv[2]
            test_category = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else "Brain"
        else:
            print("Usage: --test-v2 'prompt here' [category]", file=sys.stderr)
            sys.exit(1)

        router_result = _route_with_v2(
            category=test_category,
            prompt=test_prompt,
            messages=[{"role": "user", "content": test_prompt}],
        )

        if router_result:
            json.dump({"status": "ok", "router_result": router_result}, sys.stdout, indent=2)
        else:
            json.dump({"status": "error", "error": "V2 routing failed"}, sys.stdout)
        sys.stdout.write("\n")
        return

    # Normal request handling
    request = read_request()

    category = request.get("category") or os.getenv("LLM_ROUTER_DEFAULT_CATEGORY")
    if not category:
        raise SystemExit("Missing 'category' and LLM_ROUTER_DEFAULT_CATEGORY is not set.")

    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SystemExit("'messages' must be a non-empty list of chat messages.")

    options = request.get("options") or {}

    # Optional cost hint for the router's budget tracking.
    est_cost = options.get("estimated_cost_usd")
    try:
        est_cost_f = float(est_cost) if est_cost is not None else None
    except (TypeError, ValueError):
        est_cost_f = None

    # Build prompt from messages for routing
    prompt = _build_prompt_from_messages(messages)

    # Check for explicit Model= hint in prompt (backwards compat)
    explicit_model = router_core.parse_explicit_model(prompt)

    # Check if V2 is disabled via env
    prefer_v2 = os.getenv("LLM_ROUTER_DISABLE_V2") != "1"

    if explicit_model:
        # Explicit model hint overrides routing - use legacy for this case
        router_result = _route_with_legacy(
            category=category,
            prompt=prompt,
            estimated_cost_usd=est_cost_f,
        )
        # Override with explicit model
        router_result.update(explicit_model)
    else:
        # Use v2 with fallback to legacy
        router_result = route_with_fallback(
            category=category,
            prompt=prompt,
            messages=messages,
            estimated_cost_usd=est_cost_f,
            prefer_v2=prefer_v2,
        )

    backend = get_backend_for_router_result(router_result)

    response = backend.run(
        router_result=router_result,
        messages=messages,
        options=options,
    )

    # Log actual usage from backend response (if available)
    actual_cost = response.get("cost_usd")
    if actual_cost is not None:
        try:
            router_core.log_usage_event(
                mode=router_core.OPENCLAW_MODE,
                category=category,
                cost_usd=float(actual_cost),
                provider=router_result.get("provider"),
                model=router_result.get("model_name"),
                tokens_in=response.get("tokens_in"),
                tokens_out=response.get("tokens_out"),
                is_estimate=response.get("is_estimate", False),
                source="openclaw-skill-execution",
                units_other=response.get("units_other"),
            )
        except Exception:
            # Don't fail the request if logging errors out
            pass

    # Ensure we emit a normalized payload.
    normalized = types.normalize_response(
        router_result=router_result,
        backend_response=response,
        category=category,
    )

    json.dump(normalized, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except SystemExit as e:
        # Preserve exit code but print message to stderr if present.
        if str(e):
            print(str(e), file=sys.stderr)
        raise
