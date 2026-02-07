# WORKING.md - llm-router v2 Migration

Active work tracker for the capability-aware router migration.

## Current State

✅ **Completed:**
- Ported ClawRouter v2.0 14-dimension classifier to Python (`src/prompt_classifier/`)
- Created `router_v2.py` with capability-aware model selection
- Inline model catalog with 3 providers (Ollama primary, Anthropic secondary, OpenAI quota-limited)
- Tested: classifier working, routing logic working

## Next Steps

### Step 1: Expand Model Catalog
**Status:** 🔄 In Progress

Add all supported models to the inline catalog in `router_v2.py` and `config/models.yaml`.

#### 1.1 Ollama Cloud Models
Source: https://ollama.com/search?c=cloud

**High Priority (Already Working):**
- [x] kimi-k2.5
- [x] deepseek-v3.2
- [x] gemini-3-flash-preview

**Add These:**
- [ ] kimi-k2.5:cloud (if different from kimi-k2.5)
- [ ] kimi-k2
- [ ] kimi-k2-thinking
- [ ] deepseek-v3.1
- [ ] deepseek-r1
- [ ] qwen3-coder
- [ ] qwen3-coder-next
- [ ] qwen3-235b
- [ ] qwen3-next
- [ ] qwen3-vl
- [ ] glm-4.7
- [ ] glm-4.6
- [ ] ministral-3
- [ ] minimax-m2
- [ ] minimax-m2.1
- [ ] cogito-2.1
- [ ] gemini-3-pro-preview
- [ ] gemini-2.5-pro
- [ ] gemini-2.5-flash
- [ ] devstral-small-2
- [ ] devstral-2
- [ ] nemotron-3-nano
- [ ] rnj-1
- [ ] llama3.3
- [ ] qwen2.5

#### 1.2 Anthropic Models (via Claude CLI)
Source: `claude` CLI availability

**Already Included:**
- [x] claude-sonnet-4-5
- [x] claude-opus-4
- [x] claude-opus-4.5
- [x] claude-haiku-3-5

**Verify/Add if missing:**
- [ ] claude-3-7-sonnet
- [ ] claude-3-5-sonnet
- [ ] claude-3-5-haiku

#### 1.3 OpenAI Models (Quota Limited)
Source: ChatGPT / API (currently hitting subscription limits)

**Already Included:**
- [x] gpt-5.2
- [x] gpt-5.2-pro
- [x] o3-mini
- [x] gpt-4o-mini

**Add All Available:**
- [ ] gpt-5.2-xhigh (confirm if different from gpt-5.2)
- [ ] gpt-5.1
- [ ] gpt-5-mini
- [ ] gpt-5-nano
- [ ] gpt-4o
- [ ] o1
- [ ] o1-mini
- [ ] o3
- [ ] o4-mini

### Step 2: Wire Router v2 into Main Skill
**Status:** ⏳ Pending

Replace `router_core.py` calls with `router_v2.py`:

- [ ] Update `src/main.py` to use `Router` instead of `router_client.route()`
- [ ] Add fallback to legacy router if v2 fails
- [ ] Update backend adapters to accept `RoutingResult` format
- [ ] Test end-to-end flow: classify → select → dispatch → response

### Step 3: Tune Classification Aggressiveness
**Status:** ⏳ Pending

Current classifier leans toward cheap models. Tune to hit expected tiers:

- [ ] Review test prompts that should be COMPLEX/REASONING but landed in SIMPLE/MEDIUM
- [ ] Adjust tier boundaries or keyword weights if needed
- [ ] Add category hints to test file for verification

### Step 4: Update Status/Monitoring
**Status:** ⏳ Pending

- [ ] Update `router_core.status_summary()` to show capability-based selection stats
- [ ] Add `--why` flag to show classification signals for last request
- [ ] Track model selection by tier over time

### Step 5: Documentation & Cleanup
**Status:** ⏳ Pending

- [ ] Update `SKILL.md` with new router v2 architecture
- [ ] Remove `model_routing.csv` (archive to `.legacy/`)
- [ ] Clean up `rates.json` (decide if still needed alongside models.yaml)
- [ ] Update CLI aliases in shell configs

## Files to Touch

| File | Purpose | Status |
|------|---------|--------|
| `src/router_v2.py` | Main router, inline catalog | 🔄 Update models |
| `config/models.yaml` | Full catalog (YAML) | 🔄 Add all models |
| `src/prompt_classifier/config.py` | Classifier config | ⏳ May tune weights |
| `src/main.py` | Skill entry point | ⏳ Wire v2 router |
| `src/router_core.py` | Legacy router | ⏳ Deprecate |
| `SKILL.md` | Documentation | ⏳ Update |

## Quick Reference: Adding a Model

To add a model to `router_v2.py` inline catalog:

```python
"model-id": {
    "name": "Display Name",
    "costs": {"input_per_1k": X, "output_per_1k": Y},
    "capabilities": {
        "chat": True/False,
        "code": True/False,
        "reasoning": True/False,
        "vision": True/False,
        "long_context": True/False,
        "function_calling": True/False,
        "json_mode": True/False,
    },
    "context_window": N,
    "reliability": 0.0-1.0,
    "tags": ["tag1", "tag2"],
    # "quota_limited": True,  # Only for OpenAI
}
```

## Testing Command

```bash
cd ~/Projects/openclaw-skill-llm-router/src

python3 -c "
from router_v2 import Router, RoutingRequest
r = Router()

# Test classification
for prompt in [
    'What is 2+2?',
    'Build a React component',
    'Prove this theorem',
]:
    result = r.route(RoutingRequest(prompt=prompt))
    print(f'{prompt[:30]:30} → {result.tier:10} → {result.model_id}')
"
```
