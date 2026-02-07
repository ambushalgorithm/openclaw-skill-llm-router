"""Rule-based classifier - 14 dimension weighted scoring.

Ported from ClawRouter v2.0:
https://github.com/BlockRunAI/ClawRouter/blob/main/src/router/rules.ts

Scores a request across 14 weighted dimensions and maps the aggregate
score to a tier using configurable boundaries. Confidence is calibrated
via sigmoid - low confidence triggers ambiguous classification.

Handles ~80% of requests in <1ms with zero cost.
"""

from __future__ import annotations
import math
import re
from typing import List, Optional, Tuple

from .types import Tier, ClassificationResult, DimensionScore
from .config import ScoringConfig, DEFAULT_CONFIG


def _score_token_count(estimated_tokens: int, thresholds) -> DimensionScore:
    """Score based on prompt length."""
    if estimated_tokens < thresholds.simple:
        return DimensionScore(
            name="tokenCount",
            score=-1.0,
            signal=f"short ({estimated_tokens} tokens)",
        )
    if estimated_tokens > thresholds.complex:
        return DimensionScore(
            name="tokenCount",
            score=1.0,
            signal=f"long ({estimated_tokens} tokens)",
        )
    return DimensionScore(name="tokenCount", score=0.0, signal=None)


def _score_keyword_match(
    text: str,
    keywords: List[str],
    name: str,
    signal_label: str,
    low_threshold: int,
    high_threshold: int,
    none_score: float,
    low_score: float,
    high_score: float,
) -> DimensionScore:
    """Generic keyword matcher with thresholds."""
    matches = [kw for kw in keywords if kw.lower() in text]
    
    if len(matches) >= high_threshold:
        return DimensionScore(
            name=name,
            score=high_score,
            signal=f"{signal_label} ({', '.join(matches[:3])})",
        )
    if len(matches) >= low_threshold:
        return DimensionScore(
            name=name,
            score=low_score,
            signal=f"{signal_label} ({', '.join(matches[:3])})",
        )
    return DimensionScore(name=name, score=none_score, signal=None)


def _score_multi_step(text: str) -> DimensionScore:
    """Detect multi-step patterns."""
    patterns = [
        r"first.*then",
        r"step \d",
        r"\d\.\s",
    ]
    hits = [p for p in patterns if re.search(p, text, re.IGNORECASE)]
    
    if hits:
        return DimensionScore(name="multiStepPatterns", score=0.5, signal="multi-step")
    return DimensionScore(name="multiStepPatterns", score=0.0, signal=None)


def _score_question_complexity(prompt: str) -> DimensionScore:
    """Score based on number of questions."""
    count = len(re.findall(r"\?", prompt))
    if count > 3:
        return DimensionScore(
            name="questionComplexity",
            score=0.5,
            signal=f"{count} questions",
        )
    return DimensionScore(name="questionComplexity", score=0.0, signal=None)


def _calibrate_confidence(distance: float, steepness: float) -> float:
    """Sigmoid confidence calibration. Maps distance from tier boundary to [0.5, 1.0]."""
    return 1.0 / (1.0 + math.exp(-steepness * distance))


def classify_prompt(
    prompt: str,
    system_prompt: Optional[str] = None,
    estimated_tokens: Optional[int] = None,
    config: Optional[ScoringConfig] = None,
) -> ClassificationResult:
    """Classify a prompt using 14-dimension weighted scoring.
    
    Args:
        prompt: The user prompt to classify
        system_prompt: Optional system prompt (included in analysis)
        estimated_tokens: Optional token count (estimated if not provided)
        config: Optional custom scoring config (uses DEFAULT_CONFIG if not provided)
    
    Returns:
        ClassificationResult with tier, confidence, and detected signals
    """
    cfg = config or DEFAULT_CONFIG
    
    # Combine text for analysis
    text = f"{system_prompt or ''} {prompt}".lower().strip()
    
    # Estimate tokens if not provided (~4 chars per token)
    if estimated_tokens is None:
        estimated_tokens = math.ceil(len(text) / 4)
    
    # Score all 14 dimensions
    dimensions: List[DimensionScore] = [
        # 1. Token count
        _score_token_count(estimated_tokens, cfg.token_count_thresholds),
        
        # 2. Code presence
        _score_keyword_match(
            text, cfg.code_keywords, "codePresence", "code",
            low_threshold=1, high_threshold=2,
            none_score=0.0, low_score=0.5, high_score=1.0,
        ),
        
        # 3. Reasoning markers (high weight)
        _score_keyword_match(
            text, cfg.reasoning_keywords, "reasoningMarkers", "reasoning",
            low_threshold=1, high_threshold=2,
            none_score=0.0, low_score=0.7, high_score=1.0,
        ),
        
        # 4. Technical terms
        _score_keyword_match(
            text, cfg.technical_keywords, "technicalTerms", "technical",
            low_threshold=2, high_threshold=4,
            none_score=0.0, low_score=0.5, high_score=1.0,
        ),
        
        # 5. Creative markers
        _score_keyword_match(
            text, cfg.creative_keywords, "creativeMarkers", "creative",
            low_threshold=1, high_threshold=2,
            none_score=0.0, low_score=0.5, high_score=0.7,
        ),
        
        # 6. Simple indicators (negative score)
        _score_keyword_match(
            text, cfg.simple_keywords, "simpleIndicators", "simple",
            low_threshold=1, high_threshold=2,
            none_score=0.0, low_score=-1.0, high_score=-1.0,
        ),
        
        # 7. Multi-step patterns
        _score_multi_step(text),
        
        # 8. Question complexity
        _score_question_complexity(prompt),
        
        # 9. Imperative verbs
        _score_keyword_match(
            text, cfg.imperative_verbs, "imperativeVerbs", "imperative",
            low_threshold=1, high_threshold=2,
            none_score=0.0, low_score=0.3, high_score=0.5,
        ),
        
        # 10. Constraint indicators
        _score_keyword_match(
            text, cfg.constraint_indicators, "constraintCount", "constraints",
            low_threshold=1, high_threshold=3,
            none_score=0.0, low_score=0.3, high_score=0.7,
        ),
        
        # 11. Output format
        _score_keyword_match(
            text, cfg.output_format_keywords, "outputFormat", "format",
            low_threshold=1, high_threshold=2,
            none_score=0.0, low_score=0.4, high_score=0.7,
        ),
        
        # 12. Reference complexity
        _score_keyword_match(
            text, cfg.reference_keywords, "referenceComplexity", "references",
            low_threshold=1, high_threshold=2,
            none_score=0.0, low_score=0.3, high_score=0.5,
        ),
        
        # 13. Negation complexity
        _score_keyword_match(
            text, cfg.negation_keywords, "negationComplexity", "negation",
            low_threshold=2, high_threshold=3,
            none_score=0.0, low_score=0.3, high_score=0.5,
        ),
        
        # 14. Domain specificity
        _score_keyword_match(
            text, cfg.domain_specific_keywords, "domainSpecificity", "domain-specific",
            low_threshold=1, high_threshold=2,
            none_score=0.0, low_score=0.5, high_score=0.8,
        ),
    ]
    
    # Collect signals for debugging
    signals = [d.signal for d in dimensions if d.signal is not None]
    
    # Compute weighted score
    weighted_score = 0.0
    for d in dimensions:
        weight = cfg.dimension_weights.get(d.name, 0.0)
        weighted_score += d.score * weight
    
    # Count reasoning matches for direct override
    reasoning_matches = sum(
        1 for kw in cfg.reasoning_keywords if kw.lower() in text
    )
    
    # Direct reasoning override: 2+ reasoning markers = REASONING at high confidence
    if reasoning_matches >= 2:
        confidence = _calibrate_confidence(
            max(weighted_score, 0.3),
            cfg.confidence_steepness,
        )
        return ClassificationResult(
            score=weighted_score,
            tier="REASONING",
            confidence=max(confidence, 0.85),
            signals=signals,
            reasoning_matches=reasoning_matches,
        )
    
    # Map weighted score to tier using boundaries
    boundaries = cfg.tier_boundaries
    
    if weighted_score < boundaries.simple_medium:
        tier: Optional[Tier] = "SIMPLE"
        distance_from_boundary = boundaries.simple_medium - weighted_score
    elif weighted_score < boundaries.medium_complex:
        tier = "MEDIUM"
        distance_from_boundary = min(
            weighted_score - boundaries.simple_medium,
            boundaries.medium_complex - weighted_score,
        )
    elif weighted_score < boundaries.complex_reasoning:
        tier = "COMPLEX"
        distance_from_boundary = min(
            weighted_score - boundaries.medium_complex,
            boundaries.complex_reasoning - weighted_score,
        )
    else:
        tier = "REASONING"
        distance_from_boundary = weighted_score - boundaries.complex_reasoning
    
    # Calibrate confidence via sigmoid
    confidence = _calibrate_confidence(distance_from_boundary, cfg.confidence_steepness)
    
    # If confidence below threshold -> ambiguous (None tier)
    if confidence < cfg.confidence_threshold:
        return ClassificationResult(
            score=weighted_score,
            tier=None,
            confidence=confidence,
            signals=signals,
            reasoning_matches=reasoning_matches,
        )
    
    return ClassificationResult(
        score=weighted_score,
        tier=tier,
        confidence=confidence,
        signals=signals,
        reasoning_matches=reasoning_matches,
    )


def classify_prompt_safe(
    prompt: str,
    system_prompt: Optional[str] = None,
    estimated_tokens: Optional[int] = None,
    default_tier: Tier = "MEDIUM",
) -> Tuple[Tier, float, List[str]]:
    """Simplified interface that always returns a tier (never None).
    
    Returns:
        (tier, confidence, signals) - tier defaults to default_tier if ambiguous
    """
    result = classify_prompt(prompt, system_prompt, estimated_tokens)
    
    tier = result.tier if result.tier is not None else default_tier
    return tier, result.confidence, result.signals
