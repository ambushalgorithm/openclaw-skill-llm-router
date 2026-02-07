"""Prompt classifier ported from ClawRouter v2.0.

Analyzes prompts across 14 weighted dimensions to classify complexity tier.
"""

from .rules import classify_prompt, Tier, ClassificationResult
from .config import DEFAULT_CONFIG, ScoringConfig

__all__ = ["classify_prompt", "Tier", "ClassificationResult", "DEFAULT_CONFIG", "ScoringConfig"]
