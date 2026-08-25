"""Structured result types for campaign risk assessment (Phase 4).

Plain dataclasses kept deliberately simple so results are JSON-serializable and
easy to reason about.

NOTE (documented structure deviation): this module is named ``result_types``
rather than ``types`` because a module named ``types.py`` shadows the Python
standard-library ``types`` package (imported internally by ``enum`` and
``dataclasses``), which breaks imports under the current repository layout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceItem:
    """A single piece of structured, data-derived evidence.

    ``entities`` and ``supporting_transactions`` must only contain real IDs
    observed in the campaign/graph — never invented.
    """

    type: str
    severity: str  # low | medium | high
    description: str
    entities: list[str]
    supporting_transactions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignAssessment:
    """Full, transparent assessment of one candidate campaign."""

    campaign_id: str
    risk_score: float  # 0..100
    risk_level: str  # LOW | MEDIUM | HIGH | CRITICAL
    confidence: float  # 0..1, separate from risk
    transaction_count: int
    user_count: int
    device_count: int
    ip_count: int
    payment_instrument_count: int
    start_time: str
    end_time: str
    duration_seconds: int
    estimated_exposure: float  # "estimated transaction exposure" (sum of amounts)
    evidence: list[EvidenceItem]
    transaction_risk: float
    relationship_risk: float
    temporal_risk: float
    concentration_risk: float
    behavioral_risk: float
    high_risk_transaction_count: int
    # Diagnostic identifiers (kept for evaluation; never used by scoring).
    transaction_ids: list[str] = field(default_factory=list)
    # Entity identifiers used by the containment optimizer for action
    # generation (Phase 5); also diagnostic only.
    user_ids: list[str] = field(default_factory=list)
    device_ids: list[str] = field(default_factory=list)
    ip_ids: list[str] = field(default_factory=list)
    payment_instrument_ids: list[str] = field(default_factory=list)
    # Diagnostics kept separately (never used by the scoring formula itself).
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["evidence"] = [e.to_dict() for e in self.evidence]
        return out