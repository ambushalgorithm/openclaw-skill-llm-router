"""Router v2 - Capability-aware routing with prompt classification.

Replaces router_core.py with a cleaner architecture:
1. Classify prompt using ported ClawRouter logic
2. Map tier to capability requirements
3. Select best available model from catalog

Usage:
    from router_v2 import Router, RoutingRequest
    
    router = Router()
    result = router.route(RoutingRequest(
        prompt="Build a Python function to...",
        category_hint="Coding",  # optional override
        estimated_tokens=500,
    ))
    # result.model_id = "ollama/deepseek-v3.2"
    # result.tier = "COMPLEX"
    # result.confidence = 0.87
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from prompt_classifier import classify_prompt, Tier, ClassificationResult
from prompt_classifier.types import Tier as TierType


# Tier to capability requirements mapping (from models.yaml tier_profiles)
TIER_PROFILES = {
    "SIMPLE": {
        "required_capabilities": {"chat"},
        "cost_ceiling_per_1k": 0.001,
        "preferred_tags": {"cheap", "fast"},
    },
    "MEDIUM": {
        "required_capabilities": {"chat", "code"},
        "cost_ceiling_per_1k": 0.005,
        "preferred_tags": {"balanced"},
    },
    "COMPLEX": {
        "required_capabilities": {"reasoning", "code"},
        "cost_ceiling_per_1k": 0.020,
        "preferred_tags": {"powerful", "reliable"},
    },
    "REASONING": {
        "required_capabilities": {"reasoning"},
        "cost_ceiling_per_1k": 0.025,
        "preferred_tags": {"reasoning", "powerful"},
    },
}


@dataclass
class RoutingRequest:
    """Input to the router."""
    prompt: str
    system_prompt: Optional[str] = None
    category_hint: Optional[str] = None  # e.g., "Coding", "Brain" - overrides classification
    estimated_tokens: Optional[int] = None
    required_capabilities: Optional[Set[str]] = None  # manual override
    max_cost_per_1k: Optional[float] = None  # manual budget override
    exclude_quota_limited: bool = True  # skip OpenAI if True


@dataclass
class ModelMatch:
    """A model that matched the routing criteria."""
    provider: str
    model_id: str
    full_id: str  # "provider/model_id"
    name: str
    cost_per_1k: float  # average of input/output
    input_cost: float
    output_cost: float
    capabilities: Set[str]
    context_window: int
    reliability: float
    tags: Set[str]
    quota_limited: bool
    
    # Scoring metadata
    capability_score: int  # how many required caps matched
    tag_score: int  # how many preferred tags matched


@dataclass
class RoutingResult:
    """Output from the router."""
    model_id: str  # e.g., "ollama/deepseek-v3.2"
    provider: str
    tier: TierType
    confidence: float
    classification_signals: List[str]
    
    # Model info
    model_name: str
    cost_per_1k: float
    capabilities: Set[str]
    context_window: int
    
    # Decision metadata
    candidates_considered: int
    quota_limited: bool
    reason: str  # human-readable why this was selected


# Inline model catalog (converted from models.yaml)
DEFAULT_MODEL_CATALOG = {
    "providers": {
        # OLLAMA CLOUD - Primary (subscription-based)
        "ollama": {
            "_meta": {"availability": "primary"},
            "models": {
                "kimi-k2.5": {
                    "name": "Kimi K2.5",
                    "costs": {"input_per_1k": 0.0005, "output_per_1k": 0.0028},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": True, "long_context": True, "function_calling": True, "json_mode": True, "agentic": True},
                    "context_window": 262144, "reliability": 0.90,
                    "tags": ["default", "agentic", "vision", "reasoning"],
                },
                "deepseek-v3.2": {
                    "name": "DeepSeek V3.2",
                    "costs": {"input_per_1k": 0.00028, "output_per_1k": 0.00042},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000, "reliability": 0.85,
                    "tags": ["cheap", "fast", "code"],
                },
                "gemini-3-flash-preview": {
                    "name": "Gemini 3 Flash Preview",
                    "costs": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": True, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 1000000, "reliability": 0.85,
                    "tags": ["cheap", "fast", "long_context", "vision"],
                },
            },
        },
        # ANTHROPIC - Secondary (API key required)
        "anthropic": {
            "_meta": {"availability": "secondary"},
            "models": {
                "claude-sonnet-4-5": {
                    "name": "Claude Sonnet 4.5",
                    "costs": {"input_per_1k": 0.003, "output_per_1k": 0.015},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": True, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 200000, "reliability": 0.95,
                    "tags": ["default", "reliable", "balanced"],
                },
                "claude-opus-4": {
                    "name": "Claude Opus 4",
                    "costs": {"input_per_1k": 0.015, "output_per_1k": 0.075},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": True, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 200000, "reliability": 0.95,
                    "tags": ["powerful", "expensive", "reliable"],
                },
                "claude-haiku-3-5": {
                    "name": "Claude Haiku 3.5",
                    "costs": {"input_per_1k": 0.0008, "output_per_1k": 0.004},
                    "capabilities": {"chat": True, "code": False, "reasoning": False, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 200000, "reliability": 0.90,
                    "tags": ["cheap", "fast", "simple"],
                },
            },
        },
        # OPENAI - Quota limited
        "openai": {
            "_meta": {"availability": "quota_limited"},
            "models": {
                "gpt-5.2": {
                    "name": "GPT-5.2",
                    "costs": {"input_per_1k": 0.005, "output_per_1k": 0.015},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": True, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 400000, "reliability": 0.95,
                    "tags": ["default", "powerful"],
                    "quota_limited": True,
                },
                "o3-mini": {
                    "name": "o3-mini",
                    "costs": {"input_per_1k": 0.0011, "output_per_1k": 0.0044},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": False, "json_mode": True},
                    "context_window": 128000, "reliability": 0.92,
                    "tags": ["reasoning", "cheap"],
                    "quota_limited": True,
                },
                "gpt-4o-mini": {
                    "name": "GPT-4o Mini",
                    "costs": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": True, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000, "reliability": 0.88,
                    "tags": ["cheap", "fast", "vision"],
                    "quota_limited": True,
                },
            },
        },
    },
}


class ModelCatalog:
    """Loads and queries the model catalog."""
    
    def __init__(self, catalog_data: Optional[Dict] = None):
        self._data = catalog_data or DEFAULT_MODEL_CATALOG
        self._models: Dict[str, ModelMatch] = {}
        self._load()
    
    def _load(self) -> None:
        """Build flat model lookup from catalog data."""
        for provider, pdata in self._data.get("providers", {}).items():
            if provider.startswith("_"):
                continue
            
            provider_meta = pdata.get("_meta", {})
            is_quota_limited = provider_meta.get("availability") == "quota_limited"
            
            for model_id, mdata in pdata.get("models", {}).items():
                full_id = f"{provider}/{model_id}"
                costs = mdata.get("costs", {})
                input_cost = costs.get("input_per_1k", 0.0)
                output_cost = costs.get("output_per_1k", 0.0)
                avg_cost = (input_cost + output_cost) / 2
                
                caps = mdata.get("capabilities", {})
                caps_set = {k for k, v in caps.items() if v}
                
                self._models[full_id] = ModelMatch(
                    provider=provider,
                    model_id=model_id,
                    full_id=full_id,
                    name=mdata.get("name", model_id),
                    cost_per_1k=avg_cost,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    capabilities=caps_set,
                    context_window=mdata.get("context_window", 128000),
                    reliability=mdata.get("reliability", 0.8),
                    tags=set(mdata.get("tags", [])),
                    quota_limited=mdata.get("quota_limited", False) or is_quota_limited,
                    capability_score=0,
                    tag_score=0,
                )
    
    def find_matches(
        self,
        required_caps: Set[str],
        cost_ceiling: float,
        preferred_tags: Set[str],
        min_context: int = 0,
        exclude_quota_limited: bool = True,
    ) -> List[ModelMatch]:
        """Find all models matching the criteria, scored and sorted."""
        matches = []
        
        for model in self._models.values():
            # Skip quota-limited if requested
            if exclude_quota_limited and model.quota_limited:
                continue
            
            # Check capability requirements
            if not required_caps.issubset(model.capabilities):
                continue
            
            # Check cost ceiling
            if model.cost_per_1k > cost_ceiling:
                continue
            
            # Check context window
            if min_context > 0 and model.context_window < min_context:
                continue
            
            # Score the match
            match = ModelMatch(
                provider=model.provider,
                model_id=model.model_id,
                full_id=model.full_id,
                name=model.name,
                cost_per_1k=model.cost_per_1k,
                input_cost=model.input_cost,
                output_cost=model.output_cost,
                capabilities=model.capabilities,
                context_window=model.context_window,
                reliability=model.reliability,
                tags=model.tags,
                quota_limited=model.quota_limited,
                capability_score=len(required_caps),  # all required matched
                tag_score=len(preferred_tags & model.tags),
            )
            matches.append(match)
        
        # Sort: prefer matches with more preferred tags, then lower cost, then higher reliability
        matches.sort(key=lambda m: (-m.tag_score, m.cost_per_1k, -m.reliability))
        return matches
    
    def get_model(self, full_id: str) -> Optional[ModelMatch]:
        """Get a specific model by ID."""
        return self._models.get(full_id)
    
    def all_models(self) -> List[ModelMatch]:
        """List all loaded models."""
        return list(self._models.values())


class Router:
    """Main router interface using capability-aware selection."""
    
    def __init__(self, catalog_path: Optional[Path] = None):
        self.catalog = ModelCatalog(catalog_path)
    
    def route(self, request: RoutingRequest) -> RoutingResult:
        """Route a request to the best available model."""
        
        # Step 1: Classify prompt (unless category hint provided)
        if request.category_hint:
            # Map category hints to tiers
            tier = self._category_to_tier(request.category_hint)
            confidence = 1.0
            signals = [f"category_hint={request.category_hint}"]
        else:
            classification = classify_prompt(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                estimated_tokens=request.estimated_tokens,
            )
            tier = classification.tier or "MEDIUM"  # default on ambiguous
            confidence = classification.confidence
            signals = classification.signals
        
        # Step 2: Determine capability requirements
        if request.required_capabilities:
            required_caps = request.required_capabilities
            cost_ceiling = request.max_cost_per_1k or 0.50  # generous default
        else:
            profile = TIER_PROFILES[tier]
            required_caps = profile["required_capabilities"]
            cost_ceiling = request.max_cost_per_1k or profile["cost_ceiling_per_1k"]
        
        preferred_tags = TIER_PROFILES[tier]["preferred_tags"]
        
        # Step 3: Find matching models
        min_context = request.estimated_tokens * 2 if request.estimated_tokens else 0
        matches = self.catalog.find_matches(
            required_caps=required_caps,
            cost_ceiling=cost_ceiling,
            preferred_tags=preferred_tags,
            min_context=min_context,
            exclude_quota_limited=request.exclude_quota_limited,
        )
        
        # Step 4: Handle no matches
        if not matches:
            # Try without excluding quota-limited as last resort
            if request.exclude_quota_limited:
                matches = self.catalog.find_matches(
                    required_caps=required_caps,
                    cost_ceiling=cost_ceiling * 2,  # relax cost
                    preferred_tags=preferred_tags,
                    min_context=min_context,
                    exclude_quota_limited=False,
                )
            
            if not matches:
                # Ultimate fallback: cheapest model with any capability
                all_models = self.catalog.all_models()
                if all_models:
                    fallback = min(all_models, key=lambda m: m.cost_per_1k)
                    return self._build_result(
                        fallback, tier, confidence, signals, 0,
                        "No matches found - fell back to cheapest available"
                    )
                raise RuntimeError("No models available in catalog")
        
        # Step 5: Select best match
        selected = matches[0]
        reason = self._build_reason(selected, len(matches), tier)
        
        return self._build_result(selected, tier, confidence, signals, len(matches), reason)
    
    def _category_to_tier(self, category: str) -> TierType:
        """Map legacy category names to tiers."""
        mapping = {
            "heartbeat": "SIMPLE",
            "primary llm": "SIMPLE",
            "brain": "MEDIUM",
            "writing content": "MEDIUM",
            "web search": "COMPLEX",
            "coding": "COMPLEX",
            "image understanding": "COMPLEX",
            "voice": "MEDIUM",
        }
        return mapping.get(category.lower(), "MEDIUM")
    
    def _build_reason(self, model: ModelMatch, num_candidates: int, tier: str) -> str:
        """Build human-readable selection reason."""
        parts = [f"Tier={tier}"]
        parts.append(f"Capabilities matched: {model.capability_score}")
        if model.tag_score > 0:
            parts.append(f"Preferred tags matched: {model.tag_score}")
        parts.append(f"Cost: ${model.cost_per_1k:.4f}/1K tokens")
        parts.append(f"Reliability: {model.reliability:.0%}")
        parts.append(f"Candidates: {num_candidates}")
        if model.quota_limited:
            parts.append("(quota-limited - used as fallback)")
        return " | ".join(parts)
    
    def _build_result(
        self,
        model: ModelMatch,
        tier: TierType,
        confidence: float,
        signals: List[str],
        candidates: int,
        reason: str,
    ) -> RoutingResult:
        """Construct final routing result."""
        return RoutingResult(
            model_id=model.full_id,
            provider=model.provider,
            tier=tier,
            confidence=confidence,
            classification_signals=signals,
            model_name=model.name,
            cost_per_1k=model.cost_per_1k,
            capabilities=model.capabilities,
            context_window=model.context_window,
            candidates_considered=candidates,
            quota_limited=model.quota_limited,
            reason=reason,
        )


# CLI helpers for testing

def main():
    """CLI test interface."""
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python router_v2.py 'Your prompt here' [--category Coding]")
        sys.exit(1)
    
    prompt = sys.argv[1]
    category = None
    if "--category" in sys.argv:
        idx = sys.argv.index("--category")
        category = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    
    router = Router()
    request = RoutingRequest(
        prompt=prompt,
        category_hint=category,
    )
    
    result = router.route(request)
    
    print(json.dumps({
        "model_id": result.model_id,
        "model_name": result.model_name,
        "provider": result.provider,
        "tier": result.tier,
        "confidence": round(result.confidence, 2),
        "cost_per_1k": round(result.cost_per_1k, 4),
        "capabilities": sorted(result.capabilities),
        "signals": result.classification_signals,
        "reason": result.reason,
        "quota_limited": result.quota_limited,
    }, indent=2))


if __name__ == "__main__":
    main()
