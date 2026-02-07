# WORKING.md - llm-router v2 Migration

Active work tracker for the capability-aware router migration.

## Current State

✅ **Completed:**
- Ported ClawRouter v2.0 14-dimension classifier to Python (`src/prompt_classifier/`)
- Created `router_v2.py` with capability-aware model selection
- **41 models** in catalog: 22 Ollama (primary), 7 Anthropic (secondary), 12 OpenAI (quota-limited)
- **Policy system** for controlling model availability per task type

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

## Next Steps

### Step 2: Wire Router v2 into Main Skill 🔄 IN PROGRESS

Replace `router_core.py` calls with `router_v2.py`:

- [ ] Update `src/main.py` to use `Router` instead of `router_client.route()`
- [ ] Add fallback to legacy router if v2 fails
- [ ] Update backend adapters to accept `RoutingResult` format
- [ ] Test end-to-end flow: classify → select → dispatch → response

### Step 3: Tune Classification Aggressiveness ⏳ Pending

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

### Step 4: Update Status/Monitoring ⏳ Pending

- [ ] Update `router_core.status_summary()` to show capability-based selection stats
- [ ] Add `--why` flag to show classification signals for last request
- [ ] Track model selection by tier over time

### Step 5: Documentation & Cleanup ⏳ Pending

- [ ] Update `SKILL.md` with new router v2 architecture
- [ ] Remove `model_routing.csv` (archive to `.legacy/`)
- [ ] Clean up `rates.json` vs `config/models.yaml` (decide which to keep)
- [ ] Update CLI aliases in shell configs

## Testing Commands

```bash
# List all available models
cd ~/Projects/openclaw-skill-llm-router/src && python3 router_v2.py --list-models

# Test single prompt routing
cd ~/Projects/openclaw-skill-llm-router/src && python3 router_v2.py "Your prompt here"

# Test with category override
cd ~/Projects/openclaw-skill-llm-router/src && python3 router_v2.py "Your prompt" --category Coding

# Test policy constraints
cd ~/Projects/openclaw-skill-llm-router/src && python3 router_v2.py "Analyze this" --category Image_Understanding

# Interactive test
cd ~/Projects/openclaw-skill-llm-router/src && python3 -c "
from router_v2 import Router, RoutingRequest
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

## Files Status

| File | Purpose | Status |
|------|---------|--------|
| `src/router_v2.py` | Main router, 41 models, policy-aware | ✅ Complete |
| `config/router_policy.json` | Category-specific restrictions | ✅ Complete |
| `src/prompt_classifier/` | 14-dimension classifier | ✅ Complete |
| `src/main.py` | Skill entry point - needs v2 wiring | 🔄 In Progress |
| `config/models.yaml` | YAML catalog (redundant) | ⏳ Decide: keep or delete |
| `src/router_core.py` | Legacy router | ⏳ Deprecate after v2 wired |
| `SKILL.md` | Documentation | ⏳ Update |
