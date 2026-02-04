"""Thin client for calling the external llm-router CLI.

This module is intentionally minimal and path-agnostic. The concrete
router command is provided via the `LLM_ROUTER_COMMAND` environment
variable, which should include any fixed flags (e.g. `--mode openclaw`).

The router is expected to:
- Accept a `--category <name>` argument.
- Emit a single JSON object to stdout describing the chosen backend.

Example router JSON (illustrative only):

    {
        "provider": "anthropic",
        "model": "claude-3.7-sonnet",
        "backend": "claude-cli",
        "extra": {"max_context": 200000}
    }

The exact schema is up to your router; this module just passes it
through as a dict.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import Any, Dict


class RouterError(RuntimeError):
    pass


def get_router_command() -> str:
    cmd = os.getenv("LLM_ROUTER_COMMAND")
    if not cmd:
        raise RouterError("LLM_ROUTER_COMMAND is not set; cannot call llm-router.")
    return cmd


def route(category: str) -> Dict[str, Any]:
    cmd = get_router_command()
    # Allow the env var to contain spaces/flags; shlex.split keeps it flexible.
    parts = shlex.split(cmd) + ["--category", category]

    try:
        proc = subprocess.run(
            parts,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as e:
        raise RouterError(f"Failed to execute llm-router command: {e}") from e

    if proc.returncode != 0:
        raise RouterError(
            f"llm-router exited with {proc.returncode}: {proc.stderr.strip()}"
        )

    stdout = proc.stdout.strip()
    if not stdout:
        raise RouterError("llm-router produced no output on stdout.")

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RouterError(f"llm-router output is not valid JSON: {e}\nOutput: {stdout}") from e

    if not isinstance(data, dict):
        raise RouterError(f"llm-router JSON output must be an object, got: {type(data)!r}")

    return data
