# WORKING.md - llm-router v2 Migration

Active work tracker for the capability-aware router migration.

## Current State

✅ **Completed:**
- Ported ClawRouter v2.0 14-dimension classifier to Python (`src/prompt_classifier/`)
- Created `router_v2.py` with capability-aware model selection
- **Expanded model catalog to 41 models:**
  - Ollama Cloud: 22 models (primary)
  - Anthropic: 7 models (secondary fallback)
  - OpenAI: 12 models (quota-limited, last resort)
- Tested: classifier working, routing logic working

## Next Steps

### Step 1: Expand Model Catalog ✅ COMPLETE

**Status:** ✅ Done - 41 models populated in `router_v2.py`

**Ollama Cloud (22 models):**
- ✅ kimi-k2.5, kimi-k2, kimi-k2-thinking
- ✅ deepseek-v3.2, deepseek-v3.1, deepseek-r1
- ✅ qwen3-coder, qwen3-coder-next, qwen3-vl, qwen3-next
- ✅ gemini-3-pro-preview, gemini-3-flash-preview
- ✅ glm-4.7, glm-4.6
- ✅ ministral-3
- ✅ minimax-m2, minimax-m2.1
- ✅ devstral-small-2, devstral-2
- ✅ cogito-2.1, rnj-1, nemotron-3-nano

**Anthropic (7 models):**
- ✅ claude-sonnet-4-5, claude-opus-4-5, claude-opus-4, claude-haiku-4-5
- ✅ claude-3-7-sonnet, claude-3-5-sonnet, claude-3-5-haiku

**OpenAI (12 models - all quota_limited):**
- ✅ gpt-5.2, gpt-5.2-pro
- ✅ gpt-5.1, gpt-5-mini, gpt-5-nano
- ✅ gpt-4o, gpt-4o-mini
- ✅ o1, o1-mini, o3, o3-mini, o4-mini

### Step 2: Wire Router v2 into Main Skill 🔄 IN PROGRESS

Replace `router_core.py` calls with `router_v2.py`:

- [ ] Update `src/main.py` to use `Router` instead of `router_client.route()`
- [ ] Add fallback to legacy router if v2 fails
- [ ] Update backend adapters to accept `RoutingResult` format
- [ ] Test end-to-end flow: classify → select → dispatch → response

### Step 3: Tune Classification Aggressiveness ⏳ Pending

Current classifier leans toward cheap models. Tune to hit expected tiers:

- [ ] Review test prompts that should be COMPLEX/REASONING but landed in SIMPLE/MEDIUM
- [ ] Adjust tier boundaries or keyword weights if needed
- [ ] Add category hints to test file for verification

**Test Results (Current):**
```
"Implement a backtesting engine" → SIMPLE → ollama/ministral-3
"Prove the Riemann hypothesis..." → REASONING → ollama/gemini-3-pro-preview ✓
```

The "Implement" prompt should probably be COMPLEX (code + reasoning). 
May need to tune tier boundaries or weight "implement" more heavily.

### Step 4: Update Status/Monitoring ⏳ Pending

- [ ] Update `router_core.status_summary()` to show capability-based selection stats
- [ ] Add `--why` flag to show classification signals for last request
- [ ] Track model selection by tier over time

### Step 5: Documentation & Cleanup ⏳ Pending

- [ ] Update `SKILL.md` with new router v2 architecture
- [ ] Remove `model_routing.csv` (archive to `.legacy/`)
- [ ] Clean up `rates.json` (decide if still needed alongside models.yaml)
- [ ] Update CLI aliases in shell configs
- [ ] Sync `config/models.yaml` with inline catalog (or remove if redundant)

## Quick Stats

```bash
cd ~/Projects/openclaw-skill-llm-router/src && python3 router_v2.py --list-models
{"total": 41, "by_provider": {"ollama": 22, "anthropic": 7, "openai": 12}}
```

## Testing Commands

```bash
# List all models by provider
cd ~/Projects/openclaw-skill-llm-router/src && python3 router_v2.py --list-models

# Test single prompt routing
cd ~/Projects/openclaw-skill-llm-router/src && python3 router_v2.py "Your prompt here"

# Test with category override
cd ~/Projects/openclaw-skill-llm-router/src && python3 router_v2.py "Your prompt" --category Coding

# Interactive test
cd ~/Projects/openclaw-skill-llm-router/src && python3 -c "
from router_v2 import Router, RoutingRequest
r = Router()
for prompt in [
    'What is 2+2?',
    'Build a React component',
    'Prove this theorem',
]:
    result = r.route(RoutingRequest(prompt=prompt))
    print(f'{prompt[:30]:30} → {result.tier:10} → {result.model_id}')
"
```

## Files Status

| File | Purpose | Status |
|------|---------|--------|
| `src/router_v2.py` | Main router, 41-model inline catalog | ✅ Complete |
| `src/prompt_classifier/` | 14-dimension classifier | ✅ Complete |
| `config/models.yaml` | YAML catalog (redundant?) | 🔄 May remove |
| `src/main.py` | Skill entry point - needs v2 wiring | 🔄 In Progress |
| `src/router_core.py` | Legacy router - deprecate | ⏳ Pending |
| `SKILL.md` | Documentation - update needed | ⏳ Pending |
