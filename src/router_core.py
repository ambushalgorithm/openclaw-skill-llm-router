"""[DEPRECATED] Legacy CSV-based router - kept as fallback.

⚠️ DEPRECATION NOTICE ⚠️
This module implements the legacy CSV-based routing system from the original
llm-router skill. It is kept for backward compatibility as a fallback when
router v2 fails.

**Current routing behavior (main.py):**
1. Try router v2 (capability-aware, 41 models)
2. If v2 fails, fall back to this legacy router
3. If explicit Model= hint, use legacy (backward compat)

**New routing system (router_v2.py):**
- 14-dimension prompt classifier
- 41 models with capability tags
- Tier-based selection (SIMPLE/MEDIUM/COMPLEX/REASONING)
- Policy-driven constraints
- Auto-category detection
- Cost-optimization mode

**Migration:**
- Old CSV: `.legacy/model_routing.csv` (archived 2026-02-07)
- New policy: `config/router_policy.yaml`
- New catalog: Inline in `src/router_v2.py`

This file will be removed once router v2 proves stable (target: after 30 days
of 100% v2 usage with no fallbacks to legacy).

Internal router logic for openclaw-skill-llm-router.

Responsibilities:
- Load model routing config from a CSV file.
- Track per-category usage and budgets.
- Implement "OpenClaw" plan mode routing.
- Provide a status summary of usage vs budgets.

CSV and usage locations are configurable via environment variables:
- LLM_ROUTER_CSV_PATH: overrides the default model_routing.csv path.
- LLM_ROUTER_USAGE_PATH: overrides the default usage JSON path.
"""

from __future__ import annotations

# ... rest of file unchanged ...
