"""Investigator schemas (Phase 6).

Deterministic data structures that carry RiskLattice evidence into the AI
investigation layer and its hallucination validator. All IDs here are synthetic
only; no secrets, card numbers, or credentials are ever represented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class InvestigationValidationError(Exception):
    """Raised when a report references evidence that is not in the package."""


@dataclass
class EvidenceFinding:
    """A single grounding item (structured, data-derived)."""

    evidence_id: str
    type: str            # SHARED_DEVICE, TEMPORAL_BURST, ...
    description: str
    source: str          # graph_features | risk_engine | containment | ...
    entity_ids: list[str] = field(default_factory=list)
    transaction_ids: list[str] = field(default_factory=list)
    severity: str = "MEDIUM"   # LOW | MEDIUM | HIGH
    confidence: float = 1.0    # 0..1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskDimensions:
    transaction_risk: float = 0.0
    relationship_risk: float = 0.0
    temporal_risk: float = 0.0
    concentration_risk: float = 0.0
    behavioral_risk: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InvestigationEvidence:
    """Deterministic evidence package consumed by the investigator."""

    campaign_id: str
    campaign_score: float
    risk_level: str
    confidence: float
    transaction_ids: list[str] = field(default_factory=list)
    user_ids: list[str] = field(default_factory=list)
    device_ids: list[str] = field(default_factory=list)
    ip_ids: list[str] = field(default_factory=list)
    payment_instrument_ids: list[str] = field(default_factory=list)
    transaction_count: int = 0
    total_exposure: float = 0.0
    risk_dimensions: RiskDimensions = field(default_factory=RiskDimensions)
    findings: list[EvidenceFinding] = field(default_factory=list)
    containment_options: list[dict] = field(default_factory=list)  # describe()
    recommended_action: dict = field(default_factory=dict)
    collateral_metrics: dict = field(default_factory=dict)
    uncertainty_flags: list[str] = field(default_factory=list)
    # Ground-truth fields are deliberately absent.

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["risk_dimensions"] = self.risk_dimensions.to_dict()
        out["findings"] = [f.to_dict() for f in self.findings]
        return out


@dataclass
class Finding:
    """A claim in the report, typed FACT / INFERENCE / UNCERTAINTY."""

    type: str            # FACT | INFERENCE | UNCERTAINTY
    text: str
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InvestigationReport:
    """A validated investigation report."""

    campaign_id: str
    executive_summary: str
    why_flagged: list[Finding] = field(default_factory=list)
    risk_assessment: str = ""
    recommended_action: dict = field(default_factory=dict)
    alternative_actions: list[dict] = field(default_factory=list)
    collateral_warning: str = ""
    uncertainty: list[str] = field(default_factory=list)
    questions_for_reviewer: list[str] = field(default_factory=list)
    audit_trail: dict = field(default_factory=dict)
    provider: str = "mock"
    model: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["why_flagged"] = [f.to_dict() for f in self.why_flagged]
        return out