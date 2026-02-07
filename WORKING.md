# WORKING.md - llm-router v2 Migration

Active work tracker for the capability-aware router migration.

## Current State

✅ **Completed:**
- Ported ClawRouter v2.0 14-dimension classifier to Python (`src/prompt_classifier/`)
- Created `router_v2.py` with capability-aware model selection
- **41 models** in catalog: 22 Ollama (primary), 7 Anthropic (secondary), 12 OpenAI (quota-limited)
- **Policy system** for controlling model availability per task type
- **Router v2 wired into main skill** with legacy fallback

## Model Catalog (41 Models)

| Provider | Count | Key Models |
|----------|-------|------------|
| **Ollama** | 22 | kimi-k2.5 (agentic/vision), deepseek-v3.2 (cheap), gemini-3-pro (1M context), qwen3-coder, devstral-2, etc. |
| **Anthropic** | 7 | claude-sonnet-4-5 (workhorse), claude-opus-4/4-5 (powerful), haiku-4-5 (cheap), 3.x legacy |
| **OpenAI** | 12 | gpt-5.2, o3-mini, etc. (all marked `quota_limited`) |

## Policy System (config/router_policy.json)

**Global Controls:**
- `disabled_models`: Ban specific models entirely
- `disabled_providers`: Ban entire provider pools
- `global_limits.max_cost_per_1k`: Absolute ceiling ($0.05/1K)

**Per-Category Policies:**
| Category | Restrictions | Preferred Models |
|----------|--------------|------------------|
| **Heartbeat** | SIMPLE only, max $0.001/1K | ministral-3, gemini-flash, haiku |
| **Coding** | Requires `code`, max $0.02/1K | kimi-k2.5, qwen3-coder-next, devstral-2 |
| **Image_Understanding** | Requires `vision`, max $0.01/1K | kimi-k2.5, qwen3-vl, gemini-pro, claude-sonnet |
| **Brain** | Requires `reasoning`, max $0.01/1K | kimi-k2.5, deepseek-v3.1 (has reasoning), claude-sonnet |
| **Web_Search** | Exclude expensive (opus-4, gpt-5.2-pro, o1) | None specified |
| **Writing_Content** | Max $0.005/1K | kimi-k2.5, deepseek-v3.2 |
| **Primary_LLM** | SIMPLE/MEDIUM only | None specified |

## Step 2: Wire Router v2 into Main Skill ✅ COMPLETE

**Changes:**
- `main.py` now uses `route_with_fallback()` which tries v2 first
- If v2 fails, automatically falls back to legacy `router_client.route()`
- Explicit `Model=...` hints still use legacy (backwards compatibility)
- `LLM_ROUTER_DISABLE_V1=1` forces legacy mode if needed
- Added `--test-v2` flag for debugging

**Test it:**
```bash
cd ~/Projects/openclaw-skill-llm-router

# Test v2 routing
python3 -m src.main --test-v2 "Build a Python function" Coding
# → Router: Category=Coding
# → Tier=COMPLEX, model=ollama/kimi-k2.5

# Full request (requires Ollama running)
echo '{"category": "Coding", "messages": [{"role": "user", "content": "Hello"}]}' | python3 -m src.main
```

## Next Steps

### Step 4: Update Status/Monitoring ✅ COMPLETED 2026-02-07

- [x] **Update `router_core.status_summary()`** — Added `routing_stats` field to status output
- [x] **Add `--why` flag** — Shows classification signals, tier, model, confidence, reason
- [x] **Track % of requests using v2 vs legacy** — `routing_stats.v2_percentage` and `legacy_percentage`
- [x] **Track model selection by tier over time** — `routing_stats.by_tier`, `by_model`, `by_provider`

**New files:**
- `src/routing_tracker.py` — Decision recording and analytics

**Usage:**
```bash
# Show routing analytics in status
python3 -m src.main --status  | jq '.routing_stats'

# Explain last routing decision
python3 -m src.main --why

# Detailed routing stats for last N hours
python3 -m src.main --routing-stats --hours 6
```

### Step 3: Tune Classification Aggressiveness ⏳ Pending (moved to last)

Current classifier leans toward cheap models.

**Test Results (Current):**
```bash
$ python3 router_v2.py "Implement a backtesting engine"
→ COMPLEX → ollama/devstral-2 (has reasoning+code, $0.0016/1K) ✓

$ python3 router_v2.py "Prove the Riemann hypothesis"
→ REASONING → ollama/gemini-3-pro-preview (has reasoning, $0.0031/1K) ✓

$ python3 router_v2.py "What is 2+2?"
→ SIMPLE → ollama/ministral-3 ($0.0002/1K) ✓

$ python3 router_v2.py "Describe this image" --category Image_Understanding
→ COMPLEX → ollama/kimi-k2.5 (has vision, policy rank #1) ✓
```

**Observation:** Classification working well. Policy system correctly enforcing constraints.

- [ ] Run extended test suite with real prompts from your use cases
- [ ] Adjust tier boundaries if too aggressive toward cheap models
- [ ] Add custom keywords for quant/finance terms if needed

### Step 5: Documentation & Cleanup ✅ COMPLETED 2026-02-07

- [x] Update `SKILL.md` with router v2 architecture, 41 models, monitoring features ✅
- [x] Remove `model_routing.csv` (archived to `.legacy/`) ✅
- [x] Clean up `rates.json` vs `config/models.yaml` — **Decision: KEEP models.yaml** ✅
- [x] Update shell aliases (already current) ✅
- [x] Deprecate `router_core.py` — Added deprecation notice ✅

**Cleanup Summary:**
| File | Action |
|------|--------|
| `config/model_routing.csv` | Moved to `.legacy/model_routing.csv` with README |
| `config/models.yaml` | **RETAINED** as human-readable docs |
| `src/router_core.py` | Added deprecation notice, kept as fallback |
| `src/router_v2.py` | Primary router (active) |

**Deprecation Status:**
- router_core.py functional but marked deprecated
- Legacy CSV archived with migration guide
- v2 has 100% usage since deployment (21 requests, 0 fallbacks)

## Testing Commands

```bash
cd ~/Projects/openclaw-skill-llm-router

# Test v2 routing only
python3 -m src.main --test-v2 "Your prompt" Coding

# List all available models
python3 src/router_v2.py --list-models

# Test with category
python3 src/router_v2.py "Your prompt here" --category Coding

# Test policy constraints
python3 src/router_v2.py "Analyze this" --category Image_Understanding

# Full end-to-end test (requires backend running)
echo '{"category": "Heartbeat", "messages": [{"role": "user", "content": "Check status"}]}' | python3 -m src.main

# Interactive test
python3 -c "
from src.router_v2 import Router, RoutingRequest
r = Router()
for prompt, cat in [
    ('What is 2+2?', 'Heartbeat'),
    ('Build a React component', 'Coding'),
    ('Prove this theorem', 'Brain'),
    ('Describe the image', 'Image_Understanding'),
]:
    result = r.route(RoutingRequest(prompt=prompt, category_hint=cat))
    print(f'{cat:20} → {result.tier:10} → {result.model_id}')
"
```

## Quick Reference: Policy Configuration

Edit `config/router_policy.json` to customize:

```json
{
  "disabled_models": [
    "ollama/glm-4.6",
    "anthropic/claude-opus-4"
  ],
  "category_policies": {
    "Coding": {
      "max_cost_per_1k": 0.05,
      "exclude_models": ["ollama/ministral-3"]
    }
  }
}
```

Then restart your OpenClaw gateway to reload.

## Debug & Environment Variables

| Variable | Purpose |
|----------|---------|
| `LLM_ROUTER_DEBUG=1` | Print debug info (v2 routing failures, etc.) |
| `LLM_ROUTER_DISABLE_V2=1` | Force legacy router only |
| `LLM_ROUTER_DEFAULT_CATEGORY` | Default category if not specified in request |

## Files Status

| File | Purpose | Status |
|------|---------|--------|
| `src/router_v2.py` | Main router, 41 models, policy-aware | ✅ Complete |
| `config/router_policy.yaml` | Category-specific restrictions | ✅ Complete |
| `src/prompt_classifier/` | 14-dimension classifier | ✅ Complete |
| `src/main.py` | Skill entry point - v2 wired with fallback | ✅ Complete |
| `src/routing_tracker.py` | Decision recording & analytics | ✅ Complete |
| `config/models.yaml` | Human-readable model catalog | ✅ Kept as docs |
| `src/router_core.py` | Legacy router (fallback only) | ⚠️ Deprecated |
| `.legacy/` | Archived model_routing.csv + README | ✅ Archived |
| `SKILL.md` | Documentation | ✅ Updated |
| `WORKING.md` | This tracker | ✅ Updated |
