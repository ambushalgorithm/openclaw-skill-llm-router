"""Anthropic Claude CLI backend adapter.

This adapter shells out to the `claude` CLI to run a single-turn
completion based on the incoming chat-style messages.

Configuration is controlled via environment variables:

- `LLM_BACKEND_ANTHROPIC_COMMAND` (default: `claude`)

The router is expected to provide at least `provider` and `model_id`.
`model_id` will be passed directly as the `--model` argument. For this to
work correctly, the `model_id` value for anthropic rows in
`model_routing.csv` should match a valid Claude CLI model name or alias.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any, Dict, Mapping

from . import Backend
from .. import types as _types


class ClaudeCliBackend(Backend):
    def _command(self) -> str:
        return os.getenv("LLM_BACKEND_ANTHROPIC_COMMAND", "claude")

    def _build_prompt(self, messages: list[Dict[str, Any]]) -> str:
        """Flatten chat messages into a single text prompt.

        This is intentionally simple and stateless. System messages are
        concatenated at the top; user/assistant turns are prefixed with
        their roles to preserve some structure.
        """

        system_parts: list[str] = []
        convo_parts: list[str] = []

        for msg in messages:
            role = str(msg.get("role", "")).strip() or "user"
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                convo_parts.append(f"Assistant: {content}")
            else:  # user / other
                convo_parts.append(f"User: {content}")

        prompt_parts: list[str] = []
        if system_parts:
            prompt_parts.append("\n\n".join(system_parts))
        if convo_parts:
            if prompt_parts:
                prompt_parts.append("")  # blank line between system and convo
            prompt_parts.append("\n".join(convo_parts))

        return "\n".join(prompt_parts) if prompt_parts else ""

    def run(
        self,
        *,
        router_result: Mapping[str, Any],
        messages: list[Dict[str, Any]],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        rr = _types.coerce_router_result(router_result)
        cmd_str = self._command()
        if not cmd_str:
            raise RuntimeError("LLM_BACKEND_ANTHROPIC_COMMAND is not set and 'claude' is not available.")

        prompt = self._build_prompt(messages)

        args = shlex.split(cmd_str)
        args += [
            "--model",
            rr.model,
            "--output-format",
            "json",
            "--print",
            prompt,
        ]

        try:
            proc = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as e:
            raise RuntimeError(f"Failed to execute Claude CLI: {e}") from e

        if proc.returncode != 0:
            raise RuntimeError(
                f"Claude CLI exited with {proc.returncode}: {proc.stderr.strip()}"
            )

        raw = proc.stdout

        # The Claude CLI with --output-format json returns a JSON object.
        # We don't know its exact schema ahead of time, so we return the
        # raw JSON string under `raw_json` and set `content` to the
        # entire text output for now. Callers can refine this once the
        # concrete schema is known.
        return {
            "content": raw.strip(),
            "role": "assistant",
            "raw_json": raw,
        }
