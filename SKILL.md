---
name: llm-router
description: Universal brain skill for OpenClaw that routes LLM calls via the external llm-router, then dispatches to the appropriate LLM CLI backend (Claude, OpenAI, or others) based on router output. Use when an agent needs powerful model selection and backend-agnostic LLM execution.
---

# llm-router Skill

This skill exposes your external **llm-router** as a single, backend-agnostic "brain" for OpenClaw agents (main and sub-agents).

## What this skill does

- Accepts a **category** (e.g. `Coding`, `Brain`, `WebSearch`) and chat-style **messages`**.
- Calls your configured `llm-router` CLI to select:
  - Provider (e.g. `anthropic`, `openai`, `custom`)
  - Model (e.g. `claude-3.7-sonnet`)
  - Backend identifier (e.g. `claude-cli`, `openai-cli`, `http`)
- Dispatches the request to the correct **LLM backend CLI** based on router output.
- Returns a **normalized chat-style response** plus optional debug metadata.

Agents never talk to Claude CLI or other CLIs directly—they just call this skill.

## Inputs

The skill expects a JSON payload with:

```jsonc
{
  "category": "Coding",              // required; router category
  "messages": [                       // required; chat-style messages
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." }
  ],
  "options": {                        // optional; backend hints
    "max_tokens": 2000,
    "temperature": 0.3
  }
}
```

### Required fields

- `category` — must map to a category understood by `llm-router`.
- `messages` — array of chat messages with `role` ∈ {`system`, `user`, `assistant`} and `content` as a string.

### Optional fields

- `options.max_tokens` — soft cap for tokens (honored where the backend supports it).
- `options.temperature` — sampling temperature.

Backends that do not support some options may ignore them.

## Outputs

The skill returns a JSON object:

```jsonc
{
  "provider": "anthropic",
  "model": "claude-3.7-sonnet",
  "category": "Coding",
  "response": {
    "role": "assistant",
    "content": "..."          // primary assistant text
  },
  "raw": {                      // optional debugging info
    "router": { /* raw router response */ },
    "backend": { /* backend-specific metadata */ }
  }
}
```

Agents should generally use `response.content` as the assistant text and treat `raw` as debugging/telemetry.

## Configuration

This skill is **path-agnostic**. All integration details are provided via environment variables or a config file when wiring it into an OpenClaw instance.

## CLI helpers (recommended)

This repo includes a small status helper script you can run from anywhere.

### Aliases

Add these to your shell config (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# Raw usage counters (USD by mode/category)
alias llm-router-usage='cat ~/.llm-router-usage.json | jq'

# Pretty table: Category, $ Used, $ Limit, $ Remaining, %
alias llm-router-status='$HOME/Projects/openclaw-skill-llm-router/llm-router-status.sh'

# Raw status JSON (limits/used/remaining by category) with jq formatting
alias llm-router-status-raw='(cd ~/Projects/openclaw-skill-llm-router && python3 -m src.main --status | jq)'
```

### Direct usage (no aliases)

```bash
cd ~/Projects/openclaw-skill-llm-router
./llm-router-status.sh        # table
./llm-router-status.sh --raw  # raw JSON
```

Typical environment variables:

- `LLM_ROUTER_COMMAND` — full CLI used to call the router, e.g.:
  - `python3 /home/USER/Projects/llm-router/router.py --mode openclaw`
- Backend dispatch configuration (examples):
  - `LLM_BACKEND_ANTHROPIC_COMMAND="claude"`
  - `LLM_BACKEND_OPENAI_COMMAND="openai-chat"`
  - `LLM_BACKEND_HTTP_URL="https://llm-backend.example.com/query"`
- Defaults and limits:
  - `LLM_ROUTER_DEFAULT_CATEGORY="Brain"`
  - `LLM_SKILL_MAX_TOKENS=4000`

The concrete environment variable names and supported backends are defined in this repo's `src/` implementation.

## How agents should use this skill

### Main agent

- For non-trivial work that requires model selection and serious reasoning, call this skill instead of directly choosing a model.
- Pick a `category` that matches the task type (e.g. `Coding`, `WebSearch`, `Brain`, `Summarization`).

### Sub-agents

- Each sub-agent should be associated with a **default category**. Examples:
  - Coding sub-agent → `category = "Coding"`
  - Research sub-agent → `category = "WebSearch"`
  - General brain sub-agent → `category = "Brain"`
- When a sub-agent needs to "think hard", it should:
  1. Construct its current `messages` (system + conversation context + latest user request).
  2. Call this skill with its category and messages.
  3. Use the returned `response.content` as its assistant output.

## Implementation notes

- The implementation lives under `src/` and is responsible for:
  - Calling `LLM_ROUTER_COMMAND` with the given `category` and parsing JSON output.
  - Selecting the appropriate backend adapter from `src/backends/` based on router fields (`provider`, `backend`, etc.).
  - Converting `messages` into the backend-specific request format.
  - Parsing backend responses and normalizing them to the output schema above.
- Backends should be small, focused adapters (e.g. `claude_cli.py`, `openai_cli.py`, `http_generic.py`).

When extending to new backends, prefer adding a backend adapter rather than changing the skill interface.
