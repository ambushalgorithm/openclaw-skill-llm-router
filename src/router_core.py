"""Internal router logic for openclaw-skill-llm-router.

This module embeds the core functionality previously provided by the
separate `llm-router` repo, adapted to be self-contained inside this
skill.

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

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = REPO_ROOT / "config" / "model_routing.csv"
DEFAULT_USAGE_PATH = Path.home() / ".llm-router-usage.json"

OPENCLAW_MODE = "OpenClaw"
EXTERNAL_CALL_MODE = "External"  # Reserved; not implemented.


def _csv_path() -> Path:
    override = os.getenv("LLM_ROUTER_CSV_PATH")
    return Path(override).expanduser() if override else DEFAULT_CSV_PATH


def _usage_path() -> Path:
    override = os.getenv("LLM_ROUTER_USAGE_PATH")
    return Path(override).expanduser() if override else DEFAULT_USAGE_PATH


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CategoryConfig:
    category: str

    primary_model: str
    primary_provider: Optional[str]
    primary_model_id: Optional[str]
    primary_profile: Optional[str]

    secondary_model: str
    secondary_provider: Optional[str]
    secondary_model_id: Optional[str]
    secondary_profile: Optional[str]

    balanced_model: str
    balanced_provider: Optional[str]
    balanced_model_id: Optional[str]
    balanced_profile: Optional[str]

    default_variant: str  # "Primary" | "Secondary" | "Balanced"
    budget_openclaw_usd: float
    budget_external_usd: float


@dataclass
class Usage:
    # Usage tracked as: usage[mode][category] = used_usd
    data: Dict[str, Dict[str, float]]

    @classmethod
    def load(cls, path: Path) -> "Usage":
        if not path.exists():
            return cls(data={})
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return cls(data={})
            return cls(data=raw)
        except Exception:
            # On any error, fail safe with empty usage
            return cls(data={})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f)

    def get_used(self, mode: str, category: str) -> float:
        return float(self.data.get(mode, {}).get(category, 0.0))

    def add(self, mode: str, category: str, cost_usd: float) -> None:
        mode_map = self.data.setdefault(mode, {})
        mode_map[category] = float(mode_map.get(category, 0.0) + float(cost_usd))


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _get(row: Dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def load_category_configs(csv_path: Path | None = None) -> Dict[str, CategoryConfig]:
    configs: Dict[str, CategoryConfig] = {}
    path = csv_path or _csv_path()
    if not path.exists():
        raise FileNotFoundError(f"model routing csv not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = _get(row, "Category")
            if not category:
                continue

            primary_model = _get(row, "Primary Model")
            primary_provider = _get(row, "Primary Provider") or None
            primary_model_id = _get(row, "Primary Model ID") or None
            primary_profile = _get(row, "Primary Profile") or None

            secondary_model = _get(row, "Secondary Model")
            secondary_provider = _get(row, "Secondary Provider") or None
            secondary_model_id = _get(row, "Secondary Model ID") or None
            secondary_profile = _get(row, "Secondary Profile") or None

            balanced_model = _get(row, "Balanced Model")
            balanced_provider = _get(row, "Balanced Provider") or None
            balanced_model_id = _get(row, "Balanced Model ID") or None
            balanced_profile = _get(row, "Balanced Profile") or None

            default_variant = _get(row, "Default Variant") or "Balanced"

            budget_openclaw = float(row.get("OpenClaw Mode Monthly Budget USD", 0) or 0)
            budget_external = float(row.get("External Call Mode Monthly Budget USD", 0) or 0)

            configs[category] = CategoryConfig(
                category=category,
                primary_model=primary_model,
                primary_provider=primary_provider,
                primary_model_id=primary_model_id,
                primary_profile=primary_profile,
                secondary_model=secondary_model,
                secondary_provider=secondary_provider,
                secondary_model_id=secondary_model_id,
                secondary_profile=secondary_profile,
                balanced_model=balanced_model,
                balanced_provider=balanced_provider,
                balanced_model_id=balanced_model_id,
                balanced_profile=balanced_profile,
                default_variant=default_variant,
                budget_openclaw_usd=budget_openclaw,
                budget_external_usd=budget_external,
            )

    return configs


# ---------------------------------------------------------------------------
# Core routing logic (OpenClaw Mode / plan mode)
# ---------------------------------------------------------------------------

def normalize_variant(raw_variant: Optional[str], cfg: CategoryConfig) -> str:
    if not raw_variant or raw_variant == "Default":
        return cfg.default_variant
    raw_variant = raw_variant.capitalize()
    if raw_variant not in ("Primary", "Secondary", "Balanced"):
        return cfg.default_variant
    return raw_variant


def pick_model(cfg: CategoryConfig, variant: str) -> Dict[str, Optional[str]]:
    if variant == "Primary":
        return {
            "model_name": cfg.primary_model,
            "provider": cfg.primary_provider,
            "model_id": cfg.primary_model_id,
            "profile": cfg.primary_profile,
        }
    if variant == "Secondary":
        return {
            "model_name": cfg.secondary_model,
            "provider": cfg.secondary_provider,
            "model_id": cfg.secondary_model_id,
            "profile": cfg.secondary_profile,
        }
    # Balanced
    return {
        "model_name": cfg.balanced_model,
        "provider": cfg.balanced_provider,
        "model_id": cfg.balanced_model_id,
        "profile": cfg.balanced_profile,
    }


def route_openclaw(task: Dict[str, Any], *, csv_path: Path | None = None, usage_path: Path | None = None) -> Dict[str, Any]:
    """Route a task in OpenClaw mode.

    `task` is expected to match the JSON contract described in the
    original llm-router README (mode/category/variant/prompt/meta).
    """

    configs = load_category_configs(csv_path)
    usage = Usage.load(usage_path or _usage_path())

    category = str(task.get("category", "")).strip()
    if not category:
        return {
            "mode": OPENCLAW_MODE,
            "status": "error",
            "error": "missing_category",
        }

    cfg = configs.get(category)
    if not cfg:
        return {
            "mode": OPENCLAW_MODE,
            "status": "error",
            "error": "unknown_category",
            "category": category,
        }

    raw_variant = task.get("variant")
    requested_variant = normalize_variant(raw_variant, cfg)

    # Enforce Secondary-only policy for now, regardless of requested/default variant.
    variant = "Secondary"

    meta = task.get("meta") or {}
    estimated_cost = float(meta.get("estimated_cost_usd") or 0.0)

    # Budget lookup for OpenClaw Mode
    budget_limit = cfg.budget_openclaw_usd
    used_before = usage.get_used(OPENCLAW_MODE, category)
    remaining = budget_limit - used_before

    model_info = pick_model(cfg, variant)

    # If cost estimate would exceed budget, refuse
    if budget_limit > 0 and estimated_cost > 0 and (used_before + estimated_cost) > budget_limit:
        return {
            "mode": OPENCLAW_MODE,
            "status": "error",
            "error": "budget_exceeded",
            "category": category,
            "variant": variant,
            "model_name": model_info["model_name"],
            "budget": {
                "limit_usd": budget_limit,
                "used_usd": used_before,
                "remaining_usd": max(0.0, remaining),
                "attempted_cost_usd": estimated_cost,
            },
        }

    alerts = {
        "crossed_whole_usd": [],
        "crossed_fraction": [],  # entries like {"fraction": 0.25, "threshold_usd": ...}
    }

    # Compute thresholds crossed (whole-dollar + fraction-of-budget)
    if budget_limit > 0 and estimated_cost > 0:
        used_after = used_before + estimated_cost

        # Whole-dollar thresholds: 1,2,... up to floor(limit)
        start_dollar = int(used_before) + 1
        end_dollar = int(used_after)
        if end_dollar >= start_dollar:
            alerts["crossed_whole_usd"] = list(range(start_dollar, end_dollar + 1))

        # Fractional thresholds: 25%, 50%, 75%, 90% of budget
        for frac in (0.25, 0.50, 0.75, 0.90):
            threshold = budget_limit * frac
            if used_before < threshold <= used_after:
                alerts["crossed_fraction"].append({
                    "fraction": frac,
                    "threshold_usd": threshold,
                })

    # Update usage if we got an estimate
    if estimated_cost > 0:
        usage.add(OPENCLAW_MODE, category, estimated_cost)
        usage.save(usage_path or _usage_path())

    used_now = usage.get_used(OPENCLAW_MODE, category)

    result: Dict[str, Any] = {
        "mode": OPENCLAW_MODE,
        "status": "ok",
        "category": category,
        "variant": variant,
        "model_name": model_info["model_name"],
        "provider": model_info["provider"],
        "model_id": model_info["model_id"],
        "profile": model_info["profile"],
        "budget": {
            "limit_usd": budget_limit,
            "used_usd": used_now,
            "remaining_usd": max(0.0, budget_limit - used_now),
        },
    }

    # Only include alerts key when something actually crossed
    if alerts["crossed_whole_usd"] or alerts["crossed_fraction"]:
        result["alerts"] = alerts

    # Also include which variant was originally requested.
    result["requested_variant"] = requested_variant

    return result


# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------

def status_summary(*, csv_path: Path | None = None, usage_path: Path | None = None) -> Dict[str, Any]:
    """Return a JSON-serializable summary of usage vs budgets for each category/mode."""

    try:
        configs = load_category_configs(csv_path)
    except Exception as e:  # pragma: no cover - defensive
        return {"status": "error", "error": f"config_load_failed: {e}"}

    usage = Usage.load(usage_path or _usage_path())

    summary: Dict[str, Any] = {"status": "ok", "modes": {}}

    for mode in (OPENCLAW_MODE, EXTERNAL_CALL_MODE):
        mode_usage = usage.data.get(mode, {})
        mode_summary: Dict[str, Any] = {}

        for category, cfg in configs.items():
            used = float(mode_usage.get(category, 0.0))
            limit = cfg.budget_openclaw_usd if mode == OPENCLAW_MODE else cfg.budget_external_usd
            mode_summary[category] = {
                "used_usd": used,
                "limit_usd": limit,
                "remaining_usd": max(0.0, limit - used) if limit > 0 else None,
            }

        summary["modes"][mode] = mode_summary

    return summary
