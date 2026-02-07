"""Router v2 - Capability-aware routing with prompt classification.

Replaces router_core.py with a cleaner architecture:
1. Classify prompt using ported ClawRouter logic
2. Map tier to capability requirements
3. Apply policy constraints (disabled models, category rules)
4. Select best available model from catalog

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

from .prompt_classifier import classify_prompt, Tier, ClassificationResult
from .prompt_classifier.types import Tier as TierType


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
    exclude_models: Optional[Set[str]] = None  # per-request exclusions


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
    preferred_rank: int = 999  # rank in preferred list (lower = better)


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
    policy_applied: Optional[str] = None  # which category policy was used


# Inline model catalog - COMPLETE SET OF ALL SUPPORTED MODELS
# Generated: 2026-02-07
# Sources:
#   - Ollama Cloud: https://ollama.com/search?c=cloud
#   - Anthropic: Claude CLI (claude models)
#   - OpenAI: ChatGPT/API (currently quota limited)

DEFAULT_MODEL_CATALOG = {
    "providers": {
        # ==========================================================================================
        # OLLAMA CLOUD - Primary (subscription-based, generous limits)
        # ==========================================================================================
        "ollama": {
            "_meta": {
                "availability": "primary",
                "note": "Ollama Cloud Pro subscription - session/weekly limits tracked separately",
            },
            "models": {
                # Kimi Family (Moonshot AI)
                "kimi-k2.5": {
                    "name": "Kimi K2.5",
                    "costs": {"input_per_1k": 0.0005, "output_per_1k": 0.0028},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": True, "long_context": True, "function_calling": True, "json_mode": True, "agentic": True},
                    "context_window": 262144,
                    "max_output": 8192,
                    "reliability": 0.90,
                    "tags": ["default", "agentic", "vision", "reasoning", "popular"],
                },
                "kimi-k2": {
                    "name": "Kimi K2",
                    "costs": {"input_per_1k": 0.001, "output_per_1k": 0.003},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.88,
                    "tags": ["balanced", "reasoning", "moe"],
                },
                "kimi-k2-thinking": {
                    "name": "Kimi K2 Thinking",
                    "costs": {"input_per_1k": 0.0012, "output_per_1k": 0.004},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": False, "json_mode": True, "thinking": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.87,
                    "tags": ["reasoning", "thinking", "moe"],
                },
                
                # DeepSeek Family
                "deepseek-v3.2": {
                    "name": "DeepSeek V3.2",
                    "costs": {"input_per_1k": 0.00028, "output_per_1k": 0.00042},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 8192,
                    "reliability": 0.85,
                    "tags": ["cheap", "fast", "code", "efficient"],
                },
                "deepseek-v3.1": {
                    "name": "DeepSeek V3.1 Terminus",
                    "costs": {"input_per_1k": 0.0006, "output_per_1k": 0.0017},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": True, "json_mode": True, "thinking": True},
                    "context_window": 128000,
                    "max_output": 8192,
                    "reliability": 0.85,
                    "tags": ["cheap", "code", "thinking", "hybrid"],
                },
                "deepseek-r1": {
                    "name": "DeepSeek R1",
                    "costs": {"input_per_1k": 0.003, "output_per_1k": 0.007},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": False, "json_mode": True, "thinking": True},
                    "context_window": 128000,
                    "max_output": 8192,
                    "reliability": 0.86,
                    "tags": ["reasoning", "thinking", "expensive"],
                },
                
                # Qwen Family (Alibaba)
                "qwen3-coder": {
                    "name": "Qwen3 Coder",
                    "costs": {"input_per_1k": 0.0005, "output_per_1k": 0.0012},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.87,
                    "tags": ["code", "cheap"],
                },
                "qwen3-coder-next": {
                    "name": "Qwen3 Coder Next",
                    "costs": {"input_per_1k": 0.0005, "output_per_1k": 0.0012},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": False, "long_context": False, "function_calling": True, "json_mode": True, "agentic": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.87,
                    "tags": ["code", "agentic", "cheap"],
                },
                "qwen3-vl": {
                    "name": "Qwen3 VL",
                    "costs": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": True, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.86,
                    "tags": ["vision", "code", "multimodal"],
                },
                "qwen3-next": {
                    "name": "Qwen3 Next",
                    "costs": {"input_per_1k": 0.00015, "output_per_1k": 0.0015},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.88,
                    "tags": ["balanced", "efficient"],
                },
                
                # Google Gemini Family
                "gemini-3-pro-preview": {
                    "name": "Gemini 3 Pro Preview",
                    "costs": {"input_per_1k": 0.00125, "output_per_1k": 0.005},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": True, "long_context": True, "function_calling": True, "json_mode": True, "agentic": True},
                    "context_window": 1050000,
                    "max_output": 65536,
                    "reliability": 0.90,
                    "tags": ["long_context", "vision", "reasoning", "powerful"],
                },
                "gemini-3-flash-preview": {
                    "name": "Gemini 3 Flash Preview",
                    "costs": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": True, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 1000000,
                    "max_output": 65536,
                    "reliability": 0.85,
                    "tags": ["cheap", "fast", "long_context", "vision"],
                },
                
                # GLM Family (Zhipu AI)
                "glm-4.7": {
                    "name": "GLM 4.7",
                    "costs": {"input_per_1k": 0.00045, "output_per_1k": 0.002},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.86,
                    "tags": ["code", "balanced"],
                },
                "glm-4.6": {
                    "name": "GLM 4.6",
                    "costs": {"input_per_1k": 0.0002, "output_per_1k": 0.0011},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": True, "json_mode": True, "agentic": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.85,
                    "tags": ["cheap", "agentic"],
                },
                
                # Mistral / Ministral Family
                "ministral-3": {
                    "name": "Ministral 3",
                    "costs": {"input_per_1k": 0.0001, "output_per_1k": 0.0003},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": True, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.85,
                    "tags": ["cheap", "fast", "vision", "edge"],
                },
                
                # MiniMax Family
                "minimax-m2": {
                    "name": "MiniMax M2",
                    "costs": {"input_per_1k": 0.0003, "output_per_1k": 0.0012},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.85,
                    "tags": ["code", "efficient"],
                },
                "minimax-m2.1": {
                    "name": "MiniMax M2.1",
                    "costs": {"input_per_1k": 0.0003, "output_per_1k": 0.0012},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 256000,
                    "max_output": 4096,
                    "reliability": 0.86,
                    "tags": ["code", "multilingual"],
                },
                
                # Devstral Family (Mistral for coding)
                "devstral-small-2": {
                    "name": "Devstral Small 2",
                    "costs": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": True, "long_context": False, "function_calling": True, "json_mode": True, "agentic": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.87,
                    "tags": ["code", "agentic", "vision"],
                },
                "devstral-2": {
                    "name": "Devstral 2",
                    "costs": {"input_per_1k": 0.0008, "output_per_1k": 0.0025},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": True, "json_mode": True, "agentic": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.88,
                    "tags": ["code", "agentic", "powerful"],
                },
                
                # Specialized Models
                "cogito-2.1": {
                    "name": "Cogito 2.1",
                    "costs": {"input_per_1k": 0.00018, "output_per_1k": 0.00059},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.85,
                    "tags": ["cheap", "balanced", "mit_license"],
                },
                "rnj-1": {
                    "name": "Rnj-1",
                    "costs": {"input_per_1k": 0.0004, "output_per_1k": 0.0012},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.86,
                    "tags": ["code", "stem", "efficient"],
                },
                "nemotron-3-nano": {
                    "name": "Nemotron 3 Nano",
                    "costs": {"input_per_1k": 0.0003, "output_per_1k": 0.0012},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": True, "json_mode": True, "agentic": True},
                    "context_window": 128000,
                    "max_output": 4096,
                    "reliability": 0.87,
                    "tags": ["agentic", "efficient"],
                },
            },
        },
        
        # ==========================================================================================
        # ANTHROPIC - Secondary (API key required)
        # ==========================================================================================
        "anthropic": {
            "_meta": {
                "availability": "secondary",
                "api_type": "anthropic-messages",
                "note": "Fallback when Ollama unavailable or for specific capabilities",
            },
            "models": {
                # Claude 4.x Series (Current Generation)
                "claude-sonnet-4-5": {
                    "name": "Claude Sonnet 4.5",
                    "costs": {"input_per_1k": 0.003, "output_per_1k": 0.015},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": True, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 200000,
                    "max_output": 64000,
                    "reliability": 0.95,
                    "tags": ["default", "reliable", "balanced", "workhorse"],
                },
                "claude-opus-4-5": {
                    "name": "Claude Opus 4.5",
                    "costs": {"input_per_1k": 0.005, "output_per_1k": 0.025},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": True, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 200000,
                    "max_output": 32000,
                    "reliability": 0.94,
                    "tags": ["powerful", "reasoning", "expensive"],
                },
                "claude-opus-4": {
                    "name": "Claude Opus 4",
                    "costs": {"input_per_1k": 0.015, "output_per_1k": 0.075},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": True, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 200000,
                    "max_output": 32000,
                    "reliability": 0.95,
                    "tags": ["powerhouse", "expensive", "reliable", "max_quality"],
                },
                "claude-haiku-4-5": {
                    "name": "Claude Haiku 4.5",
                    "costs": {"input_per_1k": 0.001, "output_per_1k": 0.005},
                    "capabilities": {"chat": True, "code": False, "reasoning": False, "vision": False, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 200000,
                    "max_output": 8192,
                    "reliability": 0.90,
                    "tags": ["cheap", "fast", "simple"],
                },
                
                # Claude 3.x Series (Previous Generation - Still Available)
                "claude-3-7-sonnet": {
                    "name": "Claude 3.7 Sonnet",
                    "costs": {"input_per_1k": 0.003, "output_per_1k": 0.015},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": True, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 200000,
                    "max_output": 8192,
                    "reliability": 0.94,
                    "tags": ["balanced", "legacy"],
                },
                "claude-3-5-sonnet": {
                    "name": "Claude 3.5 Sonnet",
                    "costs": {"input_per_1k": 0.003, "output_per_1k": 0.015},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": True, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 200000,
                    "max_output": 8192,
                    "reliability": 0.93,
                    "tags": ["balanced", "legacy"],
                },
                "claude-3-5-haiku": {
                    "name": "Claude 3.5 Haiku",
                    "costs": {"input_per_1k": 0.0008, "output_per_1k": 0.004},
                    "capabilities": {"chat": True, "code": False, "reasoning": False, "vision": False, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 200000,
                    "max_output": 4096,
                    "reliability": 0.89,
                    "tags": ["cheap", "fast", "legacy"],
                },
            },
        },
        
        # ==========================================================================================
        # OPENAI - Quota Limited (ChatGPT subscription limits exceeded)
        # ==========================================================================================
        "openai": {
            "_meta": {
                "availability": "quota_limited",
                "api_type": "openai-chat",
                "note": "Currently hitting subscription limits - used as last resort",
            },
            "models": {
                # GPT-5 Family
                "gpt-5.2": {
                    "name": "GPT-5.2",
                    "costs": {"input_per_1k": 0.005, "output_per_1k": 0.015},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": True, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 400000,
                    "max_output": 128000,
                    "reliability": 0.95,
                    "tags": ["default", "powerful"],
                    "quota_limited": True,
                },
                "gpt-5.2-pro": {
                    "name": "GPT-5.2 Pro",
                    "costs": {"input_per_1k": 0.021, "output_per_1k": 0.168},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": True, "function_calling": True, "json_mode": True},
                    "context_window": 400000,
                    "max_output": 128000,
                    "reliability": 0.94,
                    "tags": ["powerful", "expensive"],
                    "quota_limited": True,
                },
                "gpt-5.1": {
                    "name": "GPT-5.1",
                    "costs": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 200000,
                    "max_output": 65536,
                    "reliability": 0.92,
                    "tags": ["balanced"],
                    "quota_limited": True,
                },
                "gpt-5-mini": {
                    "name": "GPT-5 Mini",
                    "costs": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": False, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 200000,
                    "max_output": 65536,
                    "reliability": 0.90,
                    "tags": ["cheap", "fast"],
                    "quota_limited": True,
                },
                "gpt-5-nano": {
                    "name": "GPT-5 Nano",
                    "costs": {"input_per_1k": 0.00005, "output_per_1k": 0.0004},
                    "capabilities": {"chat": True, "code": False, "reasoning": False, "vision": False, "long_context": False, "function_calling": False, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 32768,
                    "reliability": 0.87,
                    "tags": ["cheap", "simple"],
                    "quota_limited": True,
                },
                
                # GPT-4 Family
                "gpt-4o": {
                    "name": "GPT-4o",
                    "costs": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": True, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 16384,
                    "reliability": 0.93,
                    "tags": ["vision", "balanced"],
                    "quota_limited": True,
                },
                "gpt-4o-mini": {
                    "name": "GPT-4o Mini",
                    "costs": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
                    "capabilities": {"chat": True, "code": True, "reasoning": False, "vision": True, "long_context": False, "function_calling": True, "json_mode": True},
                    "context_window": 128000,
                    "max_output": 16384,
                    "reliability": 0.88,
                    "tags": ["cheap", "fast", "vision"],
                    "quota_limited": True,
                },
                
                # O-series (Reasoning Models)
                "o1": {
                    "name": "o1",
                    "costs": {"input_per_1k": 0.015, "output_per_1k": 0.06},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": True, "long_context": False, "function_calling": False, "json_mode": True, "thinking": True},
                    "context_window": 200000,
                    "max_output": 100000,
                    "reliability": 0.94,
                    "tags": ["reasoning", "thinking", "expensive"],
                    "quota_limited": True,
                },
                "o1-mini": {
                    "name": "o1-mini",
                    "costs": {"input_per_1k": 0.0011, "output_per_1k": 0.0044},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": False, "json_mode": True, "thinking": True},
                    "context_window": 128000,
                    "max_output": 65536,
                    "reliability": 0.91,
                    "tags": ["reasoning", "thinking", "cheap"],
                    "quota_limited": True,
                },
                "o3": {
                    "name": "o3",
                    "costs": {"input_per_1k": 0.002, "output_per_1k": 0.008},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": False, "json_mode": True, "thinking": True},
                    "context_window": 200000,
                    "max_output": 100000,
                    "reliability": 0.94,
                    "tags": ["reasoning", "thinking"],
                    "quota_limited": True,
                },
                "o3-mini": {
                    "name": "o3-mini",
                    "costs": {"input_per_1k": 0.0011, "output_per_1k": 0.0044},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": False, "json_mode": True, "thinking": True},
                    "context_window": 128000,
                    "max_output": 65536,
                    "reliability": 0.92,
                    "tags": ["reasoning", "thinking", "cheap"],
                    "quota_limited": True,
                },
                "o4-mini": {
                    "name": "o4-mini",
                    "costs": {"input_per_1k": 0.0011, "output_per_1k": 0.0044},
                    "capabilities": {"chat": True, "code": True, "reasoning": True, "vision": False, "long_context": False, "function_calling": False, "json_mode": True, "thinking": True},
                    "context_window": 128000,
                    "max_output": 65536,
                    "reliability": 0.92,
                    "tags": ["reasoning", "thinking", "cheap"],
                    "quota_limited": True,
                },
            },
        },
    },
}


class RouterPolicy:
    """Loads and applies routing policy from config/router_policy.yaml"""
    
    def __init__(self, policy_path: Optional[Path] = None):
        self.path = policy_path or self._default_path()
        self._data: Dict = {}
        self._load()
    
    def _default_path(self) -> Path:
        """Find router_policy.yaml in config directory."""
        repo_root = Path(__file__).resolve().parent.parent
        return repo_root / "config" / "router_policy.yaml"
    
    def _load(self) -> None:
        """Load policy from JSON/YAML if it exists, otherwise use defaults."""
        # Try JSON first (no external dependencies)
        json_path = self.path.with_suffix(".json")
        if json_path.exists():
            try:
                import json
                with open(json_path, "r") as f:
                    self._data = json.load(f)
                return
            except Exception:
                pass  # Fall through to YAML or defaults
        
        # Try YAML if JSON not available
        if self.path.exists():
            try:
                import yaml
                with open(self.path, "r") as f:
                    self._data = yaml.safe_load(f) or {}
                return
            except ImportError:
                pass  # Fall through to defaults
        
        # Default policy: no restrictions
        self._data = {
            "disabled_models": [],
            "disabled_providers": [],
            "category_policies": {},
            "global_limits": {
                "max_cost_per_1k": 0.050,
                "exclude_quota_limited": True,
                "default_tier_on_ambiguous": "MEDIUM",
            },
        }
    
    @property
    def disabled_models(self) -> Set[str]:
        """Globally disabled model IDs."""
        return set(self._data.get("disabled_models", []))
    
    @property
    def disabled_providers(self) -> Set[str]:
        """Globally disabled providers."""
        return set(self._data.get("disabled_providers", []))
    
    def get_category_policy(self, category: str) -> Dict:
        """Get policy for a specific category (legacy name)."""
        # Normalize category name
        cat_key = category.replace(" ", "_").lower()
        policies = self._data.get("category_policies", {})
        
        # Try exact match first
        if cat_key in policies:
            return policies[cat_key]
        
        # Try common variations
        variations = {
            "heartbeat": "Heartbeat",
            "primary_llm": "Primary_LLM",
            "brain": "Brain",
            "writing_content": "Writing_Content",
            "web_search": "Web_Search",
            "coding": "Coding",
            "image_understanding": "Image_Understanding",
            "voice": "Voice",
        }
        if cat_key in variations:
            return policies.get(variations[cat_key], {})
        
        return {}
    
    @property
    def global_limits(self) -> Dict:
        """Global limit defaults."""
        return self._data.get("global_limits", {
            "max_cost_per_1k": 0.050,
            "exclude_quota_limited": True,
            "default_tier_on_ambiguous": "MEDIUM",
        })


class ModelCatalog:
    """Loads and queries the model catalog with policy filtering."""
    
    def __init__(self, catalog_data: Optional[Dict] = None, policy: Optional[RouterPolicy] = None):
        self._data = catalog_data or DEFAULT_MODEL_CATALOG
        self.policy = policy or RouterPolicy()
        self._models: Dict[str, ModelMatch] = {}
        self._load()
    
    def _load(self) -> None:
        """Build flat model lookup from catalog data, applying policy filters."""
        disabled_models = self.policy.disabled_models
        disabled_providers = self.policy.disabled_providers
        
        for provider, pdata in self._data.get("providers", {}).items():
            if provider.startswith("_"):
                continue
            
            # Skip disabled providers
            if provider in disabled_providers:
                continue
            
            provider_meta = pdata.get("_meta", {})
            is_quota_limited = provider_meta.get("availability") == "quota_limited"
            
            for model_id, mdata in pdata.get("models", {}).items():
                full_id = f"{provider}/{model_id}"
                
                # Skip globally disabled models
                if full_id in disabled_models:
                    continue
                
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
        per_request_exclusions: Optional[Set[str]] = None,
        preferred_models_order: Optional[List[str]] = None,
    ) -> List[ModelMatch]:
        """Find all models matching the criteria, scored and sorted."""
        matches = []
        exclusions = per_request_exclusions or set()
        preferred_order = preferred_models_order or []
        
        for model in self._models.values():
            # Skip quota-limited if requested
            if exclude_quota_limited and model.quota_limited:
                continue
            
            # Skip per-request exclusions
            if model.full_id in exclusions:
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
            
            # Calculate preferred rank (lower = better)
            try:
                preferred_rank = preferred_order.index(model.full_id)
            except ValueError:
                preferred_rank = 999
            
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
                capability_score=len(required_caps),
                tag_score=len(preferred_tags & model.tags),
                preferred_rank=preferred_rank,
            )
            matches.append(match)
        
        # Sort: preferred rank first, then tag score, then cost, then reliability
        matches.sort(key=lambda m: (m.preferred_rank, -m.tag_score, m.cost_per_1k, -m.reliability))
        return matches
    
    def get_model(self, full_id: str) -> Optional[ModelMatch]:
        """Get a specific model by ID."""
        return self._models.get(full_id)
    
    def all_models(self) -> List[ModelMatch]:
        """List all loaded models."""
        return list(self._models.values())
    
    def count_models(self) -> Dict[str, int]:
        """Count models by provider."""
        counts = {}
        for model in self._models.values():
            counts[model.provider] = counts.get(model.provider, 0) + 1
        return counts


class Router:
    """Main router interface using capability-aware selection with policy constraints."""
    
    def __init__(self, catalog_path: Optional[Path] = None, policy_path: Optional[Path] = None):
        self.policy = RouterPolicy(policy_path)
        self.catalog = ModelCatalog(catalog_path, self.policy)
    
    def route(self, request: RoutingRequest) -> RoutingResult:
        """Route a request to the best available model."""
        
        # Step 1: Determine category and tier
        category = None
        if request.category_hint:
            category = request.category_hint
            tier = self._category_to_tier(category)
            confidence = 1.0
            signals = [f"category_hint={category}"]
        else:
            classification = classify_prompt(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                estimated_tokens=request.estimated_tokens,
            )
            tier = classification.tier
            
            # If ambiguous, check global default or policy default
            if tier is None:
                tier = self._get_default_tier()
                signals = classification.signals + [f"ambiguous_default={tier}"]
            else:
                signals = classification.signals
            
            confidence = classification.confidence
        
        # Step 2: Get constraints from category policy or tier profile
        cat_policy = self.policy.get_category_policy(category) if category else {}
        
        # Determine required capabilities
        if request.required_capabilities:
            required_caps = request.required_capabilities
        elif "required_capabilities" in cat_policy:
            required_caps = set(cat_policy["required_capabilities"])
        else:
            # Check allowed_tiers from policy - if tier not allowed, bump up
            allowed_tiers = cat_policy.get("allowed_tiers")
            if allowed_tiers and tier not in allowed_tiers:
                # Bump to minimum allowed tier
                tier = allowed_tiers[0]
            required_caps = TIER_PROFILES[tier]["required_capabilities"]
        
        # Determine cost ceiling
        if request.max_cost_per_1k:
            cost_ceiling = request.max_cost_per_1k
        elif "max_cost_per_1k" in cat_policy:
            cost_ceiling = cat_policy["max_cost_per_1k"]
        else:
            cost_ceiling = TIER_PROFILES[tier]["cost_ceiling_per_1k"]
        
        # Determine preferred tags
        if "preferred_tags" in cat_policy:
            preferred_tags = set(cat_policy["preferred_tags"])
        else:
            preferred_tags = TIER_PROFILES[tier]["preferred_tags"]
        
        # Determine exclusions
        exclusions = set()
        if request.exclude_models:
            exclusions.update(request.exclude_models)
        if "exclude_models" in cat_policy:
            exclusions.update(cat_policy["exclude_models"])
        
        # Determine preferred models order
        preferred_order = cat_policy.get("preferred_models", [])
        
        # Determine quota-limited exclusion
        exclude_quota = request.exclude_quota_limited
        if "exclude_quota_limited" in cat_policy:
            exclude_quota = cat_policy["exclude_quota_limited"]
        
        # Step 3: Find matching models
        min_context = request.estimated_tokens * 2 if request.estimated_tokens else 0
        matches = self.catalog.find_matches(
            required_caps=required_caps,
            cost_ceiling=cost_ceiling,
            preferred_tags=preferred_tags,
            min_context=min_context,
            exclude_quota_limited=exclude_quota,
            per_request_exclusions=exclusions,
            preferred_models_order=preferred_order,
        )
        
        # Step 4: Handle no matches with fallbacks
        fallback_reason = None
        
        if not matches:
            fallback_reason = "No matches with constraints"
            
            # Try without excluding quota-limited as last resort
            if exclude_quota:
                fallback_reason += " - trying quota-limited"
                matches = self.catalog.find_matches(
                    required_caps=required_caps,
                    cost_ceiling=cost_ceiling * 2,
                    preferred_tags=preferred_tags,
                    min_context=min_context,
                    exclude_quota_limited=False,
                    per_request_exclusions=exclusions,
                )
            
            if not matches:
                # Ultimate fallback: cheapest model with any capability
                all_models = self.catalog.all_models()
                if all_models:
                    fallback_reason += " - fell back to cheapest available"
                    fallback = min(all_models, key=lambda m: m.cost_per_1k)
                    return self._build_result(
                        fallback, tier, confidence, signals, 0,
                        fallback_reason, category
                    )
                raise RuntimeError("No models available in catalog")
        
        # Step 5: Select best match
        selected = matches[0]
        reason = self._build_reason(selected, len(matches), tier, fallback_reason)
        
        return self._build_result(selected, tier, confidence, signals, len(matches), reason, category)
    
    def _get_default_tier(self) -> TierType:
        """Get default tier when classification is ambiguous."""
        global_limits = self.policy.global_limits
        return global_limits.get("default_tier_on_ambiguous", "MEDIUM")
    
    def _category_to_tier(self, category: str) -> TierType:
        """Map legacy category names to tiers."""
        mapping = {
            "heartbeat": "SIMPLE",
            "primary_llm": "SIMPLE",
            "primary llm": "SIMPLE",
            "brain": "MEDIUM",
            "writing_content": "MEDIUM",
            "writing content": "MEDIUM",
            "web_search": "COMPLEX",
            "web search": "COMPLEX",
            "coding": "COMPLEX",
            "image_understanding": "COMPLEX",
            "image understanding": "COMPLEX",
            "voice": "MEDIUM",
        }
        return mapping.get(category.lower(), "MEDIUM")
    
    def _build_reason(self, model: ModelMatch, num_candidates: int, tier: str, fallback: Optional[str] = None) -> str:
        """Build human-readable selection reason."""
        parts = []
        if fallback:
            parts.append(f"({fallback})")
        parts.append(f"Tier={tier}")
        if model.preferred_rank < 999:
            parts.append(f"Preferred rank: #{model.preferred_rank + 1}")
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
        category: Optional[str],
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
            policy_applied=category,
        )
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models grouped by provider, respecting policy."""
        by_provider: Dict[str, List[str]] = {}
        for model in self.catalog.all_models():
            if model.provider not in by_provider:
                by_provider[model.provider] = []
            by_provider[model.provider].append(model.full_id)
        return by_provider


# CLI helpers for testing

def main():
    """CLI test interface."""
    import sys
    import json
    
    # Special command to list all models
    if len(sys.argv) == 2 and sys.argv[1] == "--list-models":
        router = Router()
        counts = router.catalog.count_models()
        available = router.get_available_models()
        print(json.dumps({
            "total": sum(counts.values()),
            "by_provider": counts,
            "available": available,
            "disabled_by_policy": {
                "models": list(router.policy.disabled_models),
                "providers": list(router.policy.disabled_providers),
            },
        }, indent=2))
        return
    
    if len(sys.argv) < 2:
        print("Usage: python router_v2.py 'Your prompt here' [--category Coding]")
        print("       python router_v2.py --list-models")
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
        "policy_applied": result.policy_applied,
    }, indent=2))


if __name__ == "__main__":
    main()
