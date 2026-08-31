"""Verifier ABC and VerificationResult dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..schemas import InvoiceExtraction


@dataclass
class VerificationResult:
    """Result from a single verifier."""

    verifier_name: str
    score: float  # raw score
    max_score: float  # maximum possible score (for normalization)
    details: dict = field(default_factory=dict)
    #: False when the ground truth holds nothing this verifier can check (e.g. an annotation
    #: with no line items). The composite drops inapplicable components and renormalizes the
    #: remaining weights, so an unannotated field neither rewards nor punishes the model.
    applicable: bool = True

    @property
    def normalized_score(self) -> float:
        return self.score / self.max_score if self.max_score > 0 else 0.0


class Verifier(ABC):
    """Base verifier interface for Harbor RLVR."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def verify(
        self,
        prediction: InvoiceExtraction,
        ground_truth: InvoiceExtraction,
    ) -> VerificationResult: ...
