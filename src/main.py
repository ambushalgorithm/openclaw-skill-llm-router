#!/usr/bin/env python3
"""Entry point for the openclaw-skill-llm-router.

This is a thin CLI wrapper that:
- Reads a single JSON request from stdin (or an optional file/arg in future).
- Calls the llm-router via a configured command.
- Dispatches to the appropriate LLM backend adapter.
- Writes a normalized JSON response to stdout.

The detailed behavior and configuration are documented in SKILL.md.

This file is a skeleton; backend- and router-specific logic still needs to be
implemented.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict

from . import router_client
from . import types
from . import router_core
from .backends import get_backend_for_router_result


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


def main() -> None:
    # Status mode: inspect budgets/usage instead of routing a task
    if any(arg in ("--status", "-s") for arg in sys.argv[1:]):
        summary = router_core.status_summary()
        json.dump(summary, sys.stdout)
        sys.stdout.write("\n")
        return

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

    # Call router to pick provider/model/backend.
    router_result = router_client.route(category=category, estimated_cost_usd=est_cost_f)

    backend = get_backend_for_router_result(router_result)

    response = backend.run(
        router_result=router_result,
        messages=messages,
        options=options,
    )

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
