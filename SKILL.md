---
name: llm-router
description: Universal brain skill for OpenClaw that routes LLM calls via capability-aware model selection (v2), then dispatches to the appropriate LLM CLI backend (Claude, OpenAI, Ollama, or others). Features 41 models, tier-based classification, cost-optimized routing, and unified usage tracking.
---

# llm-router Skill

This skill provides **capability-aware model routing** for OpenClaw agents. It classifies prompts by complexity (tier), selects the best model from a 41-model catalog, and dispatches to the appropriate backend CLI.

## What this skill does

- **Classifies prompts** by complexity using a 14-dimension classifier (SIMPLE/MEDIUM/COMPLEX/REASONING)
- **Auto-detects categories** from prompt content (Coding, Brain, Web Search, etc.)
- Selects from **41 models** across 3 providers with policy-driven constraints
- Supports **cost-optimized mode** to prefer cheaper models for lower-complexity tasks
- Dispatches to the correct **LLM backend CLI** (Claude, OpenAI, Ollama Cloud)
- **Tracks all routing decisions** for analytics and debugging
- Returns a **normalized chat-style response** plus routing metadata

Agents never talk to LLM CLIs directly—they just call this skill with their task.

## Router v2 Architecture

The skill uses a two-layer routing system:

### Layer 1: Classification & Selection (router_v2)
- **Prompt Classifier**: 14-dimension analysis (length, code patterns, reasoning markers, etc.)
- **Tier Assignment**: SIMPLE → MEDIUM → COMPLEX → REASONING
- **Model Catalog**: 41 models with capability tags, cost profiles, context windows
- **Policy Engine**: Category-specific rules (max cost, required capabilities, preferred models)
- **Cost Optimization**: Optional mode to prefer cheaper models for lower tiers

### Layer 2: Backend Dispatch
- Maps selected model to appropriate CLI backend
- Handles provider-specific authentication and formatting
- Returns normalized response structure

## Model Catalog (41 Models)

| Provider | Count | Key Models | Cost Range |
|----------|-------|------------|------------|
| **Ollama Cloud** | 22 | kimi-k2.5 (agentic/vision), deepseek-v3.2 (cheap), qwen3-coder, devstral-2, gemini-3-pro, etc. | $0.0002 - $0.0031/1K |
| **Anthropic** | 7 | claude-sonnet-4-5 (workhorse), claude-opus-4/4-5 (powerful), haiku-4-5 (cheap) | $0.001 - $0.075/1K |
| **OpenAI** | 12 | gpt-5.2, o3-mini, o4-mini, etc. | $0.00005 - $0.168/1K |

**Model capabilities tracked:** chat, code, reasoning, vision, long_context, function_calling, json_mode, agentic, thinking

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

## Routing Features

### Category Auto-Detection

If no `category` is specified, the router auto-detects from prompt content:

| Detected Pattern | Category | Models Considered |
|------------------|----------|-------------------|
| Code/build/implement/debug/error | Coding | qwen3-coder, devstral-2, kimi-k2.5 |
| Write/draft/email/content | Writing_Content | kimi, deepseek-v3.2 |
| Search/latest/news/lookup | Web_Search | Filters out expensive models |
| Explain/analyze/compare/why | Brain | deepseek-v3.1, kimi-k2.5 |
| Short/simple greetings/math | Simple | ministral-3 (cheapest) |

### Cost-Optimized Mode

Enable to prefer cheaper models for SIMPLE/MEDIUM tiers while keeping premium models for COMPLEX/REASONING:

```bash
python3 -m src.router_v2 "Your prompt" --cost-optimized
```

**Effect:**
- SIMPLE/MEDIUM tasks → MiniMax, DeepSeek (50-60% cheaper)
- COMPLEX/REASONING tasks → Kimi, Claude (best capabilities)

### Explicit Model Override

For backwards compatibility, explicit model hints in prompt text:
```
Model=anthropic/claude-sonnet-4-5 Explain quantum mechanics
```

This bypasses classification and forces the specified model.

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
| `status` | Pretty budget table (includes routing stats) | `llm-router status` |
| `status --real` | Real costs only | `llm-router status --real` |
| `status --normalized` | Normalized rates for comparison | `llm-router status --normalized` |
| `why` | Explain last routing decision | `python3 -m src.main --why` |
| `routing-stats` | 24h analytics (v2 vs legacy, tier breakdown) | `python3 -m src.main --routing-stats` |
| `test-v2` | Test routing without executing (see tier/model) | `python3 -m src.main --test-v2 "Write code"` |
| `status-raw` | Raw JSON budget + routing stats | `llm-router status-raw \| jq .routing_stats` |
| `log [N]` | Status snapshot history | `llm-router log 5` |
| `ledger [N]` | Individual usage entries | `llm-router ledger 10` |
| `sync` | Import OpenClaw transcripts | `llm-router sync \| jq` |
| `cron` | Sequential import → snapshot | `llm-router cron` |
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

## Monitoring & Analytics

### Routing Decision Tracking

Every routing decision is recorded with full context:

```bash
# Show last routing decision with classification signals
python3 -m src.main --why
```

Output includes:
- Router version (v2 vs legacy)
- Tier assignment (SIMPLE/MEDIUM/COMPLEX/REASONING)
- Selected model and provider
- Classification signals (why it chose that tier)
- Capabilities matched
- Selection reason (cost, preferred rank, reliability)

### Routing Statistics

24-hour analytics on routing behavior:

```bash
# Default: last 24 hours
python3 -m src.main --routing-stats

# Last 6 hours
python3 -m src.main --routing-stats --hours 6
```

Output includes:
- Total requests
- v2 vs legacy percentage
- Tier distribution (by count)
- Model usage distribution
- Provider breakdown
- Average cost per 1K tokens

**Included in `--status`:**
```bash
llm-router status | jq '.routing_stats'
```

## Policy Configuration

Edit `config/router_policy.yaml` to customize routing behavior:

```yaml
# Globally disable specific models or providers
disabled_models: []
disabled_providers: []

# Per-category policies
category_policies:
  Coding:
    allowed_tiers: [MEDIUM, COMPLEX, REASONING]
    max_cost_per_1k: 0.020
    required_capabilities: [code]
    preferred_models:
      - ollama/kimi-k2.5
      - ollama/qwen3-coder-next
    exclude_quota_limited: true

  Brain:
    allowed_tiers: [MEDIUM, COMPLEX]
    max_cost_per_1k: 0.010
    required_capabilities: [chat, reasoning]
    cost_optimized: false  # Prefer quality over cost

# Global defaults
global_limits:
  max_cost_per_1k: 0.050
  exclude_quota_limited: true
  cost_optimized: false
```

**Key options:**
- `allowed_tiers` — Which complexity tiers can be used
- `max_cost_per_1k` — Maximum cost ceiling
- `required_capabilities` — Must-have model capabilities
- `preferred_models` — Ordered list of preferred models
- `exclude_models` — Per-category exclusions
- `cost_optimized` — Prefer cheaper models for this category

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
