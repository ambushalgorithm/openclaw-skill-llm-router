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

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Constants
OPENCLAW_MODE = "OpenClaw"
DEFAULT_LEDGER_PATH = Path.home() / ".llm-router-ledger.jsonl"


def _get_ledger_path() -> Path:
    """Get ledger path from env or default."""
    env_path = os.getenv("LLM_ROUTER_LEDGER_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_LEDGER_PATH


def parse_explicit_model(prompt: str):
    """Stub - kept for API compatibility."""
    return None


@dataclass
class UsageEvent:
    """A usage event for the ledger."""
    ts_ms: int
    mode: str
    category: str
    cost_usd: float
    provider: Optional[str] = None
    model: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    is_estimate: bool = False
    source: Optional[str] = None
    units_other: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "ts_ms": self.ts_ms,
            "mode": self.mode,
            "category": self.category,
            "cost_usd": self.cost_usd,
            "is_estimate": self.is_estimate,
        }
        if self.provider is not None:
            d["provider"] = self.provider
        if self.model is not None:
            d["model"] = self.model
        if self.tokens_in is not None:
            d["tokens_in"] = self.tokens_in
        if self.tokens_out is not None:
            d["tokens_out"] = self.tokens_out
        if self.source is not None:
            d["source"] = self.source
        if self.units_other is not None:
            d["units_other"] = self.units_other
        return d


def log_usage_event(
    mode: str,
    category: str,
    cost_usd: float,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    is_estimate: bool = False,
    source: Optional[str] = None,
    ts_ms: Optional[int] = None,
    units_other: Optional[Dict[str, Any]] = None,
) -> UsageEvent:
    """Log a usage event to the ledger (JSONL).

    Appends a single line to ~/.llm-router-ledger.jsonl (or $LLM_ROUTER_LEDGER_PATH).
    Thread-safe for line-at-a-time appends.
    """
    import time

    if ts_ms is None:
        ts_ms = int(time.time() * 1000)

    ev = UsageEvent(
        ts_ms=ts_ms,
        mode=mode,
        category=category,
        cost_usd=float(cost_usd),
        provider=provider,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        is_estimate=bool(is_estimate),
        source=source,
        units_other=units_other,
    )

    ledger_path = _get_ledger_path()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(ev.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(line)

    return ev


def _load_events(ledger_path: Path) -> List[UsageEvent]:
    """Load all events from ledger (internal helper)."""
    events: List[UsageEvent] = []
    if not ledger_path.exists():
        return events

    with ledger_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ev = UsageEvent(
                    ts_ms=int(d.get("ts_ms", 0)),
                    mode=d.get("mode", "unknown"),
                    category=d.get("category", "Uncategorized"),
                    cost_usd=float(d.get("cost_usd", 0.0)),
                    provider=d.get("provider"),
                    model=d.get("model"),
                    tokens_in=d.get("tokens_in"),
                    tokens_out=d.get("tokens_out"),
                    is_estimate=bool(d.get("is_estimate", False)),
                    source=d.get("source"),
                    units_other=d.get("units_other"),
                )
                events.append(ev)
            except Exception:
                continue
    return events


def status_summary(view_mode: str = "combined") -> Dict[str, Any]:
    """Compute budget status from ledger (used by CLI dashboard).

    Returns a structure for display by the llm-router CLI.
    """
    # Default category budgets (can be customized via env in future)
    default_limits = {
        "Brain": 30.0,
        "Coding": 30.0,
        "Heartbeat": 10.0,
        "Image Understanding": 30.0,
        "Primary LLM": 30.0,
        "Voice": 30.0,
        "Web Search": 30.0,
        "Writing Content": 30.0,
    }

    ledger_path = _get_ledger_path()
    events = _load_events(ledger_path)

    # Aggregate by category
    cat_totals: Dict[str, Dict[str, float]] = {}
    for cat in default_limits:
        cat_totals[cat] = {"real_usd": 0.0, "normalized_usd": 0.0, "tokens_total": 0}

    for ev in events:
        cat = ev.category
        if cat not in cat_totals:
            # Unknown category - add dynamically
            cat_totals[cat] = {"real_usd": 0.0, "normalized_usd": 0.0, "tokens_total": 0, "limit_usd": 30.0}

        cat_totals[cat]["real_usd"] += ev.cost_usd
        # For now, normalized = real (normalized rates would come from a rates table)
        cat_totals[cat]["normalized_usd"] += ev.cost_usd
        tokens = (ev.tokens_in or 0) + (ev.tokens_out or 0)
        cat_totals[cat]["tokens_total"] += tokens

    # Build output structure matching expected CLI format
    mode_data: Dict[str, Any] = {}
    for cat, totals in cat_totals.items():
        limit = default_limits.get(cat, 30.0)
        mode_data[cat] = {
            "real_usd": float(totals["real_usd"]),
            "normalized_usd": float(totals["normalized_usd"]),
            "tokens_total": int(totals["tokens_total"]),
            "limit_usd": float(limit),
            "remaining_usd": float(limit - totals["normalized_usd"]),
            "percent_used": float(100.0 * totals["normalized_usd"] / limit) if limit > 0 else 0.0,
        }

    return {"modes": {OPENCLAW_MODE: mode_data}}
