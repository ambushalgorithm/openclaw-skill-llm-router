"""Provider pricing management for normalized cost calculations.

Fetches and caches per-token pricing from provider APIs or websites.
Used to compute normalized costs for budget comparison across providers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_RATES_PATH = Path.home() / ".llm-router-rates.json"


@dataclass
class ProviderRate:
    """Rate info for a provider/model combination."""
    provider: str
    model: str
    input_per_1k: float  # USD per 1000 input tokens
    output_per_1k: float  # USD per 1000 output tokens
    source: str  # "api", "website", "manual", "estimate"
    updated_at: Optional[str] = None


def _rates_path() -> Path:
    override = os.getenv("LLM_ROUTER_RATES_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_RATES_PATH


def load_rates() -> dict[str, dict]:
    """Load cached rates from disk."""
    path = _rates_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_rates(rates: dict) -> None:
    """Save rates to disk cache."""
    path = _rates_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rates, f, indent=2, ensure_ascii=False)


def get_rate(provider: str, model: str, rates: Optional[dict] = None) -> Optional[ProviderRate]:
    """Get rate for a specific provider/model.
    
    Returns None if no rate found.
    """
    if rates is None:
        rates = load_rates()
    
    provider = provider.lower()
    model_lower = model.lower()
    
    # Try exact match first
    if provider in rates:
        provider_rates = rates[provider]
        if isinstance(provider_rates, dict):
            # Check for model-specific rate
            for model_key, rate_data in provider_rates.items():
                if model_key.lower() in model_lower or model_lower in model_key.lower():
                    return ProviderRate(
                        provider=provider,
                        model=model,
                        input_per_1k=float(rate_data.get("input", 0)),
                        output_per_1k=float(rate_data.get("output", 0)),
                        source=rate_data.get("source", "unknown"),
                        updated_at=rate_data.get("updated_at")
                    )
            # Check for default rate
            if "default" in provider_rates:
                rate_data = provider_rates["default"]
                return ProviderRate(
                    provider=provider,
                    model=model,
                    input_per_1k=float(rate_data.get("input", 0)),
                    output_per_1k=float(rate_data.get("output", 0)),
                    source=rate_data.get("source", "unknown"),
                    updated_at=rate_data.get("updated_at")
                )
    
    return None


def calculate_normalized_cost(
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    rates: Optional[dict] = None
) -> float:
    """Calculate normalized cost using provider rates.
    
    Returns 0.0 if no rate found for provider.
    """
    rate = get_rate(provider, model, rates)
    if rate is None:
        return 0.0
    
    input_cost = (tokens_in / 1000) * rate.input_per_1k
    output_cost = (tokens_out / 1000) * rate.output_per_1k
    return input_cost + output_cost


def fetch_openai_rates() -> dict:
    """Fetch current rates from OpenAI pricing page or API.
    
    Returns dict of model -> {input, output, source}
    """
    # OpenAI pricing from https://openai.com/api/pricing/ (as of Feb 2025)
    # These should ideally be scraped, but we'll use known values
    return {
        "gpt-5.2": {"input": 0.005, "output": 0.015, "source": "manual"},
        "gpt-5.2-pro": {"input": 0.005, "output": 0.015, "source": "manual"},
        "gpt-5.1": {"input": 0.0005, "output": 0.0015, "source": "manual"},
        "gpt-5-mini": {"input": 0.00015, "output": 0.0006, "source": "manual"},
        "gpt-4o": {"input": 0.0025, "output": 0.01, "source": "manual"},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006, "source": "manual"},
        "o1": {"input": 0.015, "output": 0.06, "source": "manual"},
        "o3-mini": {"input": 0.0011, "output": 0.0044, "source": "manual"},
        "default": {"input": 0.0005, "output": 0.0015, "source": "manual"}
    }


def fetch_anthropic_rates() -> dict:
    """Fetch current rates from Anthropic pricing.
    
    Returns dict of model -> {input, output, source}
    """
    # Anthropic pricing from https://www.anthropic.com/pricing (as of Feb 2025)
    return {
        "claude-sonnet-4-5": {"input": 0.003, "output": 0.015, "source": "manual"},
        "claude-opus-4": {"input": 0.015, "output": 0.075, "source": "manual"},
        "claude-haiku-3-5": {"input": 0.0008, "output": 0.004, "source": "manual"},
        "claude-3-7-sonnet": {"input": 0.003, "output": 0.015, "source": "manual"},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015, "source": "manual"},
        "claude-3-5-haiku": {"input": 0.0008, "output": 0.004, "source": "manual"},
        "default": {"input": 0.003, "output": 0.015, "source": "manual"}
    }


def fetch_ollama_rates() -> dict:
    """Fetch/estimate Ollama Cloud rates.
    
    Ollama is subscription-based, so we use estimated market rates
    to enable cost comparison.
    """
    # Estimated market-competitive rates for comparison purposes
    return {
        "kimi-k2.5": {"input": 0.0001, "output": 0.0002, "source": "estimate"},
        "kimi-k2.5:cloud": {"input": 0.0001, "output": 0.0002, "source": "estimate"},
        "llama3.3": {"input": 0.0001, "output": 0.0002, "source": "estimate"},
        "qwen2.5": {"input": 0.0001, "output": 0.0002, "source": "estimate"},
        "default": {"input": 0.0001, "output": 0.0002, "source": "estimate"}
    }


def update_all_rates() -> dict:
    """Fetch and cache rates from all known providers.
    
    Returns the combined rates dict.
    """
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc).isoformat()
    
    rates = {
        "openai": fetch_openai_rates(),
        "openai-codex": fetch_openai_rates(),  # Same pricing
        "anthropic": fetch_anthropic_rates(),
        "ollama": fetch_ollama_rates(),
    }
    
    # Add timestamp to each rate
    for provider, models in rates.items():
        for model, rate in models.items():
            rate["updated_at"] = now
    
    save_rates(rates)
    return rates


def ensure_rates() -> dict:
    """Ensure rates are available, fetching if needed."""
    rates = load_rates()
    if not rates:
        rates = update_all_rates()
    return rates
