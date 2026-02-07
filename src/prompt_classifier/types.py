"""Type definitions for the prompt classifier."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, List

Tier = Literal["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]


@dataclass
class DimensionScore:
    """Score for a single dimension."""
    name: str
    score: float  # [-1, 1]
    signal: Optional[str] = None


@dataclass
class ClassificationResult:
    """Result of prompt classification."""
    score: float  # weighted float roughly [-0.3, 0.4]
    tier: Optional[Tier]  # None = ambiguous
    confidence: float  # [0, 1]
    signals: List[str]
    reasoning_matches: int  # count of reasoning keywords found
