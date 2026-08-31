"""Synthetic credit application generator.

Produces applications with known ground truth — income, obligations, credit
history, requested amount — so episodes are reproducible from a seed and no real
borrower data is needed.

Right now that means loading the frozen ten in `scenarios/applicants.json`, which
is what a baseline has to be measured on. Randomized generation comes once the
baseline exists and there is something for a larger set to be compared against.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import Applicant

DEFAULT_SCENARIO_PATH = Path(__file__).resolve().parents[2] / "scenarios" / "applicants.json"


def load_applicants(path: Path | None = None) -> list[Applicant]:
    """Load and validate the frozen applicant set, in file order."""
    payload = json.loads((path or DEFAULT_SCENARIO_PATH).read_text())
    return [Applicant(**row) for row in payload["applicants"]]
