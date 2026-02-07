"""Configuration for the prompt classifier.

Ported from ClawRouter v2.0:
https://github.com/BlockRunAI/ClawRouter/blob/main/src/router/config.ts
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TokenCountThresholds:
    simple: int
    complex: int


@dataclass
class TierBoundaries:
    simple_medium: float
    medium_complex: float
    complex_reasoning: float


@dataclass
class ScoringConfig:
    """Full configuration for 14-dimension weighted scoring."""
    
    # Token count thresholds
    token_count_thresholds: TokenCountThresholds
    
    # Keyword lists for each dimension
    code_keywords: List[str]
    reasoning_keywords: List[str]
    simple_keywords: List[str]
    technical_keywords: List[str]
    creative_keywords: List[str]
    
    # New dimension keyword lists (v2.0 additions)
    imperative_verbs: List[str]
    constraint_indicators: List[str]
    output_format_keywords: List[str]
    reference_keywords: List[str]
    negation_keywords: List[str]
    domain_specific_keywords: List[str]
    
    # Weights for each dimension (sum to 1.0)
    dimension_weights: Dict[str, float]
    
    # Tier boundaries on weighted score axis
    tier_boundaries: TierBoundaries
    
    # Confidence calibration
    confidence_steepness: float
    confidence_threshold: float


# Default configuration - exact port from ClawRouter v2.0
DEFAULT_CONFIG = ScoringConfig(
    token_count_thresholds=TokenCountThresholds(
        simple=50,
        complex=500,
    ),
    
    code_keywords=[
        "function", "class", "import", "def", "SELECT",
        "async", "await", "const", "let", "var", "return", "```",
    ],
    
    reasoning_keywords=[
        "prove", "theorem", "derive", "step by step",
        "chain of thought", "formally", "mathematical", "proof", "logically",
    ],
    
    simple_keywords=[
        "what is", "define", "translate", "hello",
        "yes or no", "capital of", "how old", "who is", "when was",
    ],
    
    technical_keywords=[
        "algorithm", "optimize", "architecture", "distributed",
        "kubernetes", "microservice", "database", "infrastructure",
        # Explanations and analysis
        "explain", "explanation", "detailed", "thorough", "in-depth",
        "compare", "contrast", "difference between", "how does",
        "why does", "what is the best way", "design pattern",
        # Quant/Trading/Finance domain
        "sharpe", "cointegration", "futures", "correlation", "skew", "kurtosis",
        "volatility", "backtest", "backtesting", "pine script", "tradingview",
        "position sizing", "risk management", "adaptive", "adf test", "augmented dickey-fuller",
        "kelly criterion", "kelly", "max drawdown", "calmar ratio", "sortino",
        "var", "cvar", "value at risk", "monte carlo", "bootstrap",
        "btc", "bitcoin", "gold", "es", "nq", "spy", "spx", "vix",
        "fomc", "fed", "interest rate", "inflation", "cpi", "macro",
    ],
    
    creative_keywords=[
        "story", "poem", "compose", "brainstorm", "creative", "imagine", "write a",
    ],
    
    # New dimension keywords (v2.0)
    imperative_verbs=[
        "build", "create", "implement", "design", "develop",
        "construct", "generate", "deploy", "configure", "set up",
    ],
    
    constraint_indicators=[
        "under", "at most", "at least", "within", "no more than",
        "o(", "maximum", "minimum", "limit", "budget",
    ],
    
    output_format_keywords=[
        "json", "yaml", "xml", "table", "csv", "markdown", "schema", "format as", "structured",
    ],
    
    reference_keywords=[
        "above", "below", "previous", "following", "the docs", "the api", "the code", "earlier", "attached",
    ],
    
    negation_keywords=[
        "don't", "do not", "avoid", "never", "without", "except", "exclude", "no longer",
    ],
    
    domain_specific_keywords=[
        "quantum", "fpga", "vlsi", "risc-v", "asic", "photonics",
        "genomics", "proteomics", "topological", "homomorphic",
        "zero-knowledge", "lattice-based",
    ],
    
    # Dimension weights (sum to 1.0) - exact from ClawRouter
    dimension_weights={
        "tokenCount": 0.08,
        "codePresence": 0.15,
        "reasoningMarkers": 0.18,
        "technicalTerms": 0.10,
        "creativeMarkers": 0.05,
        "simpleIndicators": 0.12,
        "multiStepPatterns": 0.12,
        "questionComplexity": 0.05,
        "imperativeVerbs": 0.03,
        "constraintCount": 0.04,
        "outputFormat": 0.03,
        "referenceComplexity": 0.02,
        "negationComplexity": 0.01,
        "domainSpecificity": 0.02,
    },
    
    tier_boundaries=TierBoundaries(
        simple_medium=-0.15,  # Even lower: fewer prompts hit SIMPLE
        medium_complex=0.05,   # Lowered: explanations now hit COMPLEX
        complex_reasoning=0.30,  # Raised slightly: REASONING is special
    ),
    
    confidence_steepness=12.0,
    confidence_threshold=0.70,
)
