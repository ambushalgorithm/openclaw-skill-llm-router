# Legacy Files Archive

This directory contains deprecated files from the llm-router skill.

## Contents

| File | Original Location | Deprecated Date | Notes |
|------|-------------------|-----------------|-------|
| `model_routing.csv` | `config/model_routing.csv` | 2026-02-07 | Legacy CSV-based routing. Replaced by capability-aware v2 router with `config/router_policy.yaml` and inline model catalog in `src/router_v2.py`. |

## Migration Notes

**Old system (CSV-based):**
- Categories defined by CSV rows
- Primary/Secondary/Balanced model columns
- Budget tracking per category
- Simple variant-based selection

**New system (v2 capability-aware):**
- 14-dimension prompt classifier
- 41 models with capability tags
- Policy-driven constraints
- Tier-based selection (SIMPLE/MEDIUM/COMPLEX/REASONING)
- Auto-category detection
- Cost-optimization mode

**If you need to reference old routing config:**
```bash
# Old CSV-based routing logic lives in src/router_core.py
# New routing logic in src/router_v2.py
```

These files are kept for historical reference only. Do not use for new development.
