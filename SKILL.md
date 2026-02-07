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

This repo includes a unified CLI entry point (`llm-router`) with subcommands for all operations.

### Quick setup

```bash
# Add to ~/.bashrc or ~/.zshrc
alias llm-router='$HOME/Projects/openclaw-skill-llm-router/llm-router'
```

### Usage

```bash
llm-router <command> [args]
```

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| `dashboard` | Full overview: sync + ledger + log + status | `llm-router dashboard` |
| `status` | Pretty budget table (combined view — default) | `llm-router status` |
| `status --real` | Real costs only (Ollama shows \$0) | `llm-router status --real` |
| `status --normalized` | Normalized rates for cost comparison | `llm-router status --normalized` |
| `status-raw` | Raw JSON budget data | `llm-router status-raw \| jq` |
| `log [N]` | Status snapshot history (default: 10) | `llm-router log 5` |
| `ledger [N]` | Individual usage entries (default: 20) | `llm-router ledger 10` |
| `sync` | Import OpenClaw transcripts | `llm-router sync \| jq` |
| `cron` | Sequential import → snapshot (for cron) | `llm-router cron` |
| `help` | Show help message | `llm-router help` |

### View modes (cost tracking)

The `status` command supports three view modes for cost tracking:

**Combined view (default):**
- Shows real cost (API charges), normalized cost (estimated), tokens, and budget %
- Best for comprehensive overview
- Uses normalized costs for budget percentage

**Real mode (`--real`):**
- Shows actual API costs only
- Ollama shows $0 (subscription-based)
- Use case: "What did I actually spend?" (true P&L)

**Normalized mode (`--normalized`):**
- Applies provider rates to all models uniformly for cost comparison
- Use case: "Should I use Ollama or Codex for this workload?"

**Rate sources:**
- **OpenAI/Codex:** Direct from provider (gpt-5.2: $0.005/$0.015 per 1K)
- **Anthropic:** Direct from provider (claude-sonnet-4-5: $0.003/$0.015 per 1K)
- **Ollama Cloud:** Sourced from provider APIs — Together AI, Fireworks.ai, DeepSeek direct
  - Kimi K2.5: $0.00050/$0.00280 per 1K (Together AI)
  - DeepSeek V3.2: $0.00028/$0.00042 per 1K (DeepSeek direct)
  - Qwen3: $0.00050/$0.00120 per 1K (Together AI)
  - Note: Ollama Cloud is subscription-based; these rates enable cost benchmarking

**Rates file:** `config/rates.json` (in repo, version controlled)
- Override via `LLM_ROUTER_RATES_PATH` environment variable
- Update command: `python3 -m src.pricing` (regenerates from hardcoded values)

### Convenience aliases

```bash
# Full dashboard (recommended for daily checks)
alias llm-router-dashboard='llm-router dashboard'

# Raw usage JSON (from router's usage file)
alias llm-router-usage='cat ~/.llm-router-usage.json | jq'
```

### Log usage events (unified tracking)

You can append a usage event into the router's canonical ledger (useful for
"direct/no-router" calls).

```bash
echo '{
  "mode": "OpenClaw",
  "category": "Brain",
  "provider": "openai",
  "model": "gpt-5.2",
  "tokens_in": 123,
  "tokens_out": 45,
  "cost_usd": 0.001,
  "is_estimate": true,
  "source": "direct-no-router"
}' | python3 -m src.main --log-usage | jq
```

### Automatic category assignment (OpenClaw transcripts)

When importing OpenClaw session transcripts, the system uses a **3-layer categorization** with strict precedence:

**Priority order (highest wins):**
1. **Explicit header in message text** — assistant messages starting with `Router: Category=X` or `Direct (no router): Category=Y`
2. **Agent default category** — from `~/.openclaw/agents/<agentId>/config.json`
3. **Heuristic pattern matching** — based on `agent_id` naming conventions
4. **Fallback** — `"Brain"` (if none of the above match)

#### Agent config files

Create `~/.openclaw/agents/<agentId>/config.json` to set an agent's default category:

```json
{
  "default_category": "Primary LLM",
  "description": "Main assistant for direct chat"
}
```

The `main` agent should typically use `"Primary LLM"` to distinguish direct LLM calls from routed calls.

#### Heuristic patterns

If no header and no agent config, the system matches `agent_id` against these patterns:

| Pattern | Category |
|---------|----------|
| `coding\*`, `codex\*`, `cline\*`, `roo\*`, `claude?code\*` | Coding |
| `heartbeat\*`, `cron\*`, `scheduler\*` | Heartbeat |
| `image\*`, `vision\*`, `look\*`, `see\*` | Image Understanding |
| `voice\*`, `speak\*`, `tts\*`, `audio\*` | Voice |
| `web\*`, `search\*`, `browse\*`, `fetch\*` | Web Search |
| `write\*`, `content\*`, `blog\*`, `draft\*` | Writing Content |
| `main\*`, `primary\*`, `default\*`, `core\*` | Primary LLM |

#### Recommended header format

To guarantee correct categorization, prepend your assistant replies with:

- **For routed calls:** `Router: Category=Coding`
- **For direct LLM calls:** `Direct (no router): Category=Primary LLM`

This header-based approach overrides all other mechanisms and ensures predictable tracking.

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
- Unified usage tracking:
  - `LLM_ROUTER_LEDGER_PATH` — JSONL event ledger path (default: `~/.llm-router-ledger.jsonl`)
  - `LLM_ROUTER_RATES_PATH` — Provider rates JSON path (default: `config/rates.json` in repo)
  - `LLM_ROUTER_TZ` — timezone used for "today" totals in `--status` (default: `America/Bogota`)

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
