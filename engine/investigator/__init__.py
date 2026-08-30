"""AI investigation layer (Phase 6).

Builds deterministic, evidence-grounded investigation reports over RiskLattice
campaign assessments and simulated containment recommendations. The AI layer
explains and summarizes only — it never determines fraud, never invents
evidence, and never executes containment. Works fully offline with the
deterministic mock provider (no API key required).
"""

from __future__ import annotations

from engine.investigator.schemas import (
    InvestigationEvidence,
    InvestigationReport,
)

__all__ = ["InvestigationEvidence", "InvestigationReport"]