"""Track routing decisions for debugging and monitoring.

Stores the last N routing decisions with classification signals,
enabling --why flag and usage analytics.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class RoutingDecision:
    """Record of a single routing decision."""
    ts_ms: int
    router_version: str  # "v2" or "legacy"
    category: str
    prompt_preview: str  # First 100 chars
    tier: str
    model_id: str
    provider: str
    cost_per_1k: float
    confidence: float
    signals: List[str]
    capabilities: List[str]
    candidates_considered: int
    reason: str
    policy_applied: Optional[str] = None
    quota_limited: bool = False


DEFAULT_TRACKER_PATH = Path.home() / ".llm-router-decisions.jsonl"
MAX_DECISIONS = 100  # Keep last 100 for memory efficiency


def _get_tracker_path() -> Path:
    """Get the path to the decisions log file."""
    override = os.getenv("LLM_ROUTER_DECISIONS_PATH")
    if override:
        return Path(override).expanduser()
    return DEFAULT_TRACKER_PATH


def record_decision(decision: RoutingDecision) -> None:
    """Record a routing decision to the log."""
    path = _get_tracker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(decision), ensure_ascii=False))
        f.write("\n")


def get_last_decision() -> Optional[RoutingDecision]:
    """Get the most recent routing decision."""
    path = _get_tracker_path()
    if not path.exists():
        return None
    
    last_line = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    
    if not last_line:
        return None
    
    try:
        data = json.loads(last_line)
        return RoutingDecision(**data)
    except Exception:
        return None


def get_recent_decisions(n: int = 10) -> List[RoutingDecision]:
    """Get the N most recent routing decisions (most recent first)."""
    path = _get_tracker_path()
    if not path.exists():
        return []
    
    lines = []
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Take last N and reverse (most recent first)
    recent_lines = lines[-n:][::-1]
    
    decisions = []
    for line in recent_lines:
        try:
            data = json.loads(line)
            decisions.append(RoutingDecision(**data))
        except Exception:
            continue
    
    return decisions


def get_routing_stats(hours: int = 24) -> Dict[str, Any]:
    """Get routing statistics for the last N hours.
    
    Returns:
        Dict with:
        - total_requests: int
        - v2_percentage: float
        - legacy_percentage: float
        - by_tier: Dict[str, int]
        - by_model: Dict[str, int]
        - by_provider: Dict[str, int]
        - avg_cost_per_1k: float
    """
    cutoff_ms = int((datetime.now(timezone.utc).timestamp() - hours * 3600) * 1000)
    
    decisions = get_recent_decisions(n=MAX_DECISIONS)
    recent = [d for d in decisions if d.ts_ms >= cutoff_ms]
    
    if not recent:
        return {
            "period_hours": hours,
            "total_requests": 0,
            "v2_percentage": 0.0,
            "legacy_percentage": 0.0,
            "by_tier": {},
            "by_model": {},
            "by_provider": {},
            "avg_cost_per_1k": 0.0,
        }
    
    total = len(recent)
    v2_count = sum(1 for d in recent if d.router_version == "v2")
    
    by_tier: Dict[str, int] = {}
    by_model: Dict[str, int] = {}
    by_provider: Dict[str, int] = {}
    total_cost = 0.0
    
    for d in recent:
        by_tier[d.tier] = by_tier.get(d.tier, 0) + 1
        by_model[d.model_id] = by_model.get(d.model_id, 0) + 1
        by_provider[d.provider] = by_provider.get(d.provider, 0) + 1
        total_cost += d.cost_per_1k
    
    return {
        "period_hours": hours,
        "total_requests": total,
        "v2_percentage": round(v2_count / total * 100, 1),
        "legacy_percentage": round((total - v2_count) / total * 100, 1),
        "by_tier": by_tier,
        "by_model": by_model,
        "by_provider": by_provider,
        "avg_cost_per_1k": round(total_cost / total, 4),
    }


def format_why_output(decision: Optional[RoutingDecision] = None) -> str:
    """Format a routing decision for --why output."""
    if decision is None:
        decision = get_last_decision()
    
    if decision is None:
        return "No routing decisions recorded yet."
    
    ts = datetime.fromtimestamp(decision.ts_ms / 1000, timezone.utc)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    lines = [
        f"Last Routing Decision ({ts_str})",
        "=" * 50,
        f"Router Version: {decision.router_version}",
        f"Category:       {decision.category}",
        f"Tier:           {decision.tier}",
        f"Model:          {decision.model_id}",
        f"Provider:       {decision.provider}",
        f"Cost:           ${decision.cost_per_1k:.4f}/1K",
        f"Confidence:     {decision.confidence:.0%}",
        f"Candidates:     {decision.candidates_considered}",
        "",
        "Classification Signals:",
    ]
    
    for signal in decision.signals:
        lines.append(f"  → {signal}")
    
    lines.extend([
        "",
        "Capabilities Required:",
        f"  → {', '.join(decision.capabilities)}",
        "",
        "Selection Reason:",
        f"  → {decision.reason}",
    ])
    
    if decision.policy_applied:
        lines.append(f"\nPolicy Applied: {decision.policy_applied}")
    
    if decision.quota_limited:
        lines.append("\n⚠️  Model is quota-limited (fallback)")
    
    return "\n".join(lines)
