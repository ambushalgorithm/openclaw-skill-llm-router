"""Ollama backend adapter (local and cloud).

This adapter shells out to the `ollama` CLI to run completions.
Configuration is controlled via environment variables:
- `LLM_BACKEND_OLLAMA_COMMAND` (default: `ollama`)
- `LLM_BACKEND_OLLAMA_HOST` (optional, for remote Ollama instances)

For Ollama Cloud (subscription-based), we estimate costs since the
API doesn't return USD values. We track:
- Estimated cost_usd (calculated from token estimates)
- tokens_in / tokens_out (estimated if not provided by API)
- units_other for Ollama-specific metrics (like "session_percent")
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import Any, Dict, Mapping, Optional, List

from .. import types as _types
from ..quota_tracker import check_ollama_quota


class OllamaCliBackend:
    """Ollama CLI backend with hybrid usage tracking."""

    # Rough cost estimation per 1K tokens for Ollama Cloud Pro
    # This is a placeholder - adjust based on actual subscription limits
    ESTIMATED_COST_PER_1K_TOKENS = 0.0001  # $0.0001 per 1K tokens

    def _command(self) -> str:
        return os.getenv("LLM_BACKEND_OLLAMA_COMMAND", "ollama")

    def _host(self) -> Optional[str]:
        return os.getenv("LLM_BACKEND_OLLAMA_HOST")

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 chars per token for English text."""
        if not text:
            return 0
        return len(text) // 4

    def _build_messages_payload(self, messages: list[Dict[str, Any]]) -> list[Dict[str, str]]:
        """Convert messages to Ollama chat format."""
        ollama_msgs = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = str(msg.get("content", ""))
            if content:
                # Map system/user/assistant roles
                ollama_role = role if role in ("system", "user", "assistant") else "user"
                ollama_msgs.append({"role": ollama_role, "content": content})
        return ollama_msgs

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
            raise RuntimeError("LLM_BACKEND_OLLAMA_COMMAND is not set and 'ollama' is not available.")

        # Get model from router result
        raw_router = rr.raw or {}
        model = rr.model or raw_router.get("model_name", "llama3.2")
        # Strip provider prefix if present (e.g., "ollama/kimi-k2.5" -> "kimi-k2.5")
        if model and "/" in model:
            model = model.split("/", 1)[1]

        # Build messages
        ollama_messages = self._build_messages_payload(messages)

        # Build the request payload
        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
        }
        if options.get("max_tokens"):
            payload["options"] = {"num_predict": options["max_tokens"]}

        # Create curl command for Ollama API
        host = self._host() or "http://127.0.0.1:11434"
        url = f"{host}/api/chat"

        args = shlex.split(cmd_str)
        # Use ollama as a curl-like tool or direct CLI
        # Try standard Ollama CLI chat first
        try:
            # Try using ollama CLI chat mode
            prompt_parts = []
            for msg in messages:
                role = str(msg.get("role", "user"))
                content = str(msg.get("content", ""))
                if content:
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                    else:
                        prompt_parts.append(f"User: {content}")

            full_prompt = "\n\n".join(prompt_parts)

            args += ["run", model, full_prompt, "--format", "json"]

            env = os.environ.copy()
            if self._host():
                env["OLLAMA_HOST"] = self._host()

            proc = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=30,  # Add timeout to prevent hanging on CLI
            )

            # If CLI fails, try API mode
            if proc.returncode != 0:
                # Fall back to direct HTTP via curl or python
                return self._run_via_api(host, model, ollama_messages, options)

            raw = proc.stdout
            text_content = raw.strip()

        except Exception as e:
            # Try API mode
            return self._run_via_api(host, model, ollama_messages, options)

        # Estimate tokens
        prompt_text = " ".join(m.get("content", "") for m in messages)
        tokens_in = self._estimate_tokens(prompt_text)
        tokens_out = self._estimate_tokens(text_content)

        # Estimate cost
        total_tokens = tokens_in + tokens_out
        estimated_cost = (total_tokens / 1000) * self.ESTIMATED_COST_PER_1K_TOKENS

        # Check quota and get warnings
        quota_warnings = check_ollama_quota(tokens_in, tokens_out)

        result_payload = {
            "content": text_content,
            "role": "assistant",
            "raw_json": raw if 'raw' in dir() else None,
            "cost_usd": estimated_cost,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "is_estimate": True,
            "units_other": {
                "type": "ollama_usage",
                "total_tokens": total_tokens,
                "description": "Ollama Cloud subscription usage - see dashboard for limits",
            },
        }

        if quota_warnings:
            result_payload["quota_warnings"] = quota_warnings

        return result_payload

    def _run_via_api(
        self,
        host: str,
        model: str,
        messages: list[Dict[str, str]],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run via HTTP API using curl or python requests."""
        import urllib.request
        import urllib.error

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if options.get("temperature"):
            payload["options"] = {"temperature": options["temperature"]}
        if options.get("max_tokens"):
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = options["max_tokens"]

        data = json.dumps(payload).encode("utf-8")
        url = f"{host}/api/chat"

        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))

            message = result.get("message", {})
            text_content = message.get("content", "")

            # Get actual tokens if available
            tokens_in = result.get("prompt_eval_count")
            tokens_out = result.get("eval_count")

            # Fallback estimation
            if tokens_in is None:
                prompt_text = " ".join(m.get("content", "") for m in messages)
                tokens_in = self._estimate_tokens(prompt_text)
            if tokens_out is None:
                tokens_out = self._estimate_tokens(text_content)

            total_tokens = tokens_in + tokens_out
            estimated_cost = (total_tokens / 1000) * self.ESTIMATED_COST_PER_1K_TOKENS

            # Check quota and get warnings
            quota_warnings = check_ollama_quota(tokens_in, tokens_out)

            result_payload = {
                "content": text_content,
                "role": "assistant",
                "raw_json": json.dumps(result),
                "cost_usd": estimated_cost,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "is_estimate": True,
                "units_other": {
                    "type": "ollama_usage",
                    "total_tokens": total_tokens,
                    "description": "Ollama Cloud subscription usage - see dashboard for limits",
                },
            }

            if quota_warnings:
                result_payload["quota_warnings"] = quota_warnings

            return result_payload

        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Ollama API at {host}: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")
