from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    fingerprint: str
    dimension: int
    preprocessing: str

    def __post_init__(self):
        if not self.fingerprint or not self.preprocessing or self.dimension < 1:
            raise ValueError("Model fingerprint, dimension and preprocessing are required")


@dataclass(frozen=True)
class Identity:
    id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnrollmentSample:
    id: str
    identity_id: str
    embedding: np.ndarray
    model: ModelSpec
    quality: float = 1.0
    source_id: str = ""


@dataclass(frozen=True)
class Detection:
    """bbox is unmirrored image pixels (x, y, width, height)."""
    bbox: tuple[float, float, float, float]
    embedding: np.ndarray
    model: ModelSpec
    confidence: float
    quality: float
    aligned: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class Candidate:
    identity_id: str
    score: float


@dataclass(frozen=True)
class Match:
    identity_id: str | None
    score: float
    margin: float
    candidates: tuple[Candidate, ...] = ()
    reason: str = "unknown"


@dataclass(frozen=True)
class LiveResult:
    track_id: int
    detection: Detection
    match: Match
    stable: bool


@dataclass(frozen=True)
class MatchPolicy:
    threshold: float = 0.45
    margin: float = 0.05

    def __post_init__(self):
        if not -1 <= self.threshold <= 1 or not 0 <= self.margin <= 2:
            raise ValueError("Invalid cosine threshold or margin")


@dataclass(frozen=True)
class ValidationSample:
    id: str
    identity_id: str | None
    embedding: np.ndarray
    model: ModelSpec
    source_id: str = ""
