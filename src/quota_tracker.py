"""Quota tracking for subscription-based providers (Ollama Cloud, etc.).

Implements token-based limits with warning thresholds.
Based on Ollama Cloud Pro limits observed in practice:
- Session: ~X tokens per session (resets every few hours)
- Weekly: ~Y tokens per week (resets every 7 days)

Since exact limits aren't exposed via API, we estimate from user data
and track cumulative usage with configurable thresholds.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List


DEFAULT_QUOTA_PATH = Path.home() / ".llm-router-quota.json"

# Estimated Ollama Cloud Pro limits based on typical usage patterns
# These are estimates - adjust based on actual observed limits
DEFAULT_OLLAMA_SESSION_TOKENS = 500_000  # Per session (~2-4 hour window)
DEFAULT_OLLAMA_WEEKLY_TOKENS = 5_000_000  # Per week

WARNING_THRESHOLDS = [0.50, 0.75, 0.90, 0.95]  # 50%, 75%, 90%, 95%


@dataclass
class QuotaWindow:
    """A single quota window (session, daily, weekly, etc.)."""
    name: str  # e.g., "session", "weekly"
    limit_tokens: int
    used_tokens: int = 0
    reset_at: Optional[str] = None  # ISO timestamp
    last_warning_fraction: float = 0.0  # Last threshold we warned at

    def remaining(self) -> int:
        return max(0, self.limit_tokens - self.used_tokens)

    def fraction_used(self) -> float:
        if self.limit_tokens <= 0:
            return 0.0
        return self.used_tokens / self.limit_tokens

    def check_warnings(self) -> List[str]:
        """Return warning messages for any newly crossed thresholds."""
        warnings = []
        current_fraction = self.fraction_used()

        for threshold in WARNING_THRESHOLDS:
            if self.last_warning_fraction < threshold <= current_fraction:
                pct = int(threshold * 100)
                remaining_pct = int((1 - threshold) * 100)
                warnings.append(
                    f"⚠️ Ollama Cloud {self.name} quota: {pct}% used ({remaining_pct}% remaining). "
                    f"Resets in {self._time_until_reset()}."
                )
                self.last_warning_fraction = threshold

        return warnings

    def _time_until_reset(self) -> str:
        if not self.reset_at:
            return "unknown time"
        try:
            reset_dt = datetime.fromisoformat(self.reset_at)
            now = datetime.now(timezone.utc)
            if reset_dt > now:
                delta = reset_dt - now
                hours = int(delta.total_seconds() / 3600)
                if hours < 1:
                    minutes = int(delta.total_seconds() / 60)
                    return f"{minutes} minutes"
                return f"{hours} hours"
        except Exception:
            pass
        return "unknown time"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "limit_tokens": self.limit_tokens,
            "used_tokens": self.used_tokens,
            "reset_at": self.reset_at,
            "last_warning_fraction": self.last_warning_fraction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuotaWindow":
        return cls(
            name=data.get("name", "unknown"),
            limit_tokens=data.get("limit_tokens", 0),
            used_tokens=data.get("used_tokens", 0),
            reset_at=data.get("reset_at"),
            last_warning_fraction=data.get("last_warning_fraction", 0.0),
        )


@dataclass
class ProviderQuota:
    """Quota state for a single provider."""
    provider: str
    windows: Dict[str, QuotaWindow]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "windows": {k: v.to_dict() for k, v in self.windows.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProviderQuota":
        return cls(
            provider=data.get("provider", "unknown"),
            windows={
                k: QuotaWindow.from_dict(v)
                for k, v in data.get("windows", {}).items()
            },
        )


class QuotaTracker:
    """Tracks subscription-based quotas separately from dollar-based budgets."""

    def __init__(self, quota_path: Optional[Path] = None):
        self.quota_path = quota_path or DEFAULT_QUOTA_PATH
        self._data: Dict[str, ProviderQuota] = {}
        self._load()

    def _load(self) -> None:
        if self.quota_path.exists():
            try:
                with open(self.quota_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._data = {
                    k: ProviderQuota.from_dict(v)
                    for k, v in raw.get("providers", {}).items()
                }
            except Exception:
                self._data = {}

    def _save(self) -> None:
        self.quota_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.quota_path, "w", encoding="utf-8") as f:
            json.dump(
                {"providers": {k: v.to_dict() for k, v in self._data.items()}},
                f,
                indent=2,
            )

    def _get_or_create_ollama(self) -> ProviderQuota:
        if "ollama" not in self._data:
            now = datetime.now(timezone.utc)
            session_reset = (now + timedelta(hours=3)).isoformat()
            weekly_reset = (now + timedelta(days=7)).isoformat()

            self._data["ollama"] = ProviderQuota(
                provider="ollama",
                windows={
                    "session": QuotaWindow(
                        name="session",
                        limit_tokens=DEFAULT_OLLAMA_SESSION_TOKENS,
                        used_tokens=0,
                        reset_at=session_reset,
                    ),
                    "weekly": QuotaWindow(
                        name="weekly",
                        limit_tokens=DEFAULT_OLLAMA_WEEKLY_TOKENS,
                        used_tokens=0,
                        reset_at=weekly_reset,
                    ),
                },
            )
        return self._data["ollama"]

    def record_usage(
        self,
        provider: str,
        tokens_in: int,
        tokens_out: int,
        category: Optional[str] = None,
    ) -> List[str]:
        """Record token usage and return any warning messages."""
        warnings = []
        total_tokens = tokens_in + tokens_out

        if provider.lower() == "ollama":
            quota = self._get_or_create_ollama()
            now = datetime.now(timezone.utc)

            # Check if windows need resetting
            for window_name, window in quota.windows.items():
                if window.reset_at:
                    try:
                        reset_dt = datetime.fromisoformat(window.reset_at)
                        if now >= reset_dt:
                            # Reset the window
                            window.used_tokens = 0
                            window.last_warning_fraction = 0.0
                            if window_name == "session":
                                window.reset_at = (now + timedelta(hours=3)).isoformat()
                            else:
                                window.reset_at = (now + timedelta(days=7)).isoformat()
                    except Exception:
                        pass

                # Add usage
                window.used_tokens += total_tokens

                # Check for warnings
                new_warnings = window.check_warnings()
                warnings.extend(new_warnings)

            self._save()

        return warnings

    def get_status(self, provider: str) -> Dict[str, Any]:
        """Get current quota status for a provider."""
        if provider.lower() == "ollama":
            quota = self._get_or_create_ollama()
            return {
                "provider": provider,
                "windows": {
                    name: {
                        "limit_tokens": w.limit_tokens,
                        "used_tokens": w.used_tokens,
                        "remaining_tokens": w.remaining(),
                        "fraction_used": round(w.fraction_used(), 3),
                        "reset_at": w.reset_at,
                    }
                    for name, w in quota.windows.items()
                },
            }
        return {"provider": provider, "windows": {}}

    def adjust_limits(
        self,
        provider: str,
        session_tokens: Optional[int] = None,
        weekly_tokens: Optional[int] = None,
    ) -> None:
        """Manually adjust limits based on observed actuals (from dashboard)."""
        if provider.lower() == "ollama":
            quota = self._get_or_create_ollama()
            if session_tokens is not None:
                quota.windows["session"].limit_tokens = session_tokens
            if weekly_tokens is not None:
                quota.windows["weekly"].limit_tokens = weekly_tokens
            self._save()


def check_ollama_quota(
    tokens_in: int,
    tokens_out: int,
    quota_path: Optional[Path] = None,
) -> List[str]:
    """Convenience function to check Ollama quota and get warnings."""
    tracker = QuotaTracker(quota_path)
    return tracker.record_usage("ollama", tokens_in, tokens_out)


def get_ollama_status(quota_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get current Ollama quota status."""
    tracker = QuotaTracker(quota_path)
    return tracker.get_status("ollama")
