"""Configurable rules and thresholds for campaign risk (Phase 4).

Everything here is documented and configurable. These are *heuristic reference
points* for scale normalization (log-scaled where useful) and severity labels —
they are not hidden magic numbers and not claims of statistical optimality.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Risk-level bands. Exposed and documented; configurable.
#   [0, 30)     LOW
#   [30, 60)    MEDIUM
#   [60, 80)    HIGH
#   [80, 100]   CRITICAL
# ---------------------------------------------------------------------------
RISK_LEVEL_BANDS: list[tuple[float, str]] = [
    (0.0, "LOW"),
    (30.0, "MEDIUM"),
    (60.0, "HIGH"),
    (80.0, "CRITICAL"),
]

# High-risk transaction definition (matches Phase-2 threshold default).
HIGH_RISK_TX_THRESHOLD = 0.5


def risk_level_for_score(score: float) -> str:
    """Map a 0..100 campaign score to a risk level using the documented bands."""
    if score < 0:
        score = 0.0
    level = RISK_LEVEL_BANDS[-1][1]  # default CRITICAL
    for floor, band_level in RISK_LEVEL_BANDS:
        if score >= floor:
            level = band_level
    return level


# ---------------------------------------------------------------------------
# Dimension weights for the combined campaign score.
# Label: "transparent heuristic campaign score" (not statistically optimal).
# ---------------------------------------------------------------------------
DIMENSION_WEIGHTS: dict[str, float] = {
    "transaction_risk": 0.35,
    "relationship_risk": 0.25,
    "temporal_risk": 0.20,
    "concentration_risk": 0.10,
    "behavioral_risk": 0.10,
}


# ---------------------------------------------------------------------------
# Scoring reference scales (documented heuristics, log-scaled).
# ---------------------------------------------------------------------------
# Transactions per hour at which temporal_risk saturates (~2 tx/min sustained).
TEMPORAL_RATE_REF_PER_HOUR = 120.0

# Reference "strong concentration" of users sharing a single entity type
# (device / IP / payment instrument). Log-scaled so the signal is smooth.
SHARED_ENTITY_REF_USERS = 24.0
# Reference transaction fan-out per entity.
ENTITY_TX_REF = 60.0

# balances for component sub-signals (sum to 1 within each component).
TRANSACTION_RISK_BALANCE = {"mean": 0.45, "high_ratio": 0.35, "p90": 0.20}
RELATIONSHIP_RISK_BALANCE = {"density": 0.35, "fanout": 0.35, "multi_signal": 0.30}
BEHAVIORAL_RISK_BALANCE = {"refund": 0.40, "failed": 0.30, "amount_top_share": 0.30}


# ---------------------------------------------------------------------------
# Confidence building blocks (confidence is separate from risk).
# ---------------------------------------------------------------------------
CONFIDENCE_W = {
    "signal_fraction": 0.30,      # how many dimensions are clearly elevated
    "completeness": 0.25,         # does the campaign expose all entity types?
    "risk_consistency": 0.20,     # low dispersion of per-tx risk score
    "graph_evidence": 0.15,       # structural strength of shared-entity counts
    "temporal_richness": 0.10,    # nontrivial duration/transaction spread
}

# A dimension is "elevated" (independent signal present) above this.
SIGNAL_TRIGGER = 0.50


# ---------------------------------------------------------------------------
# Evidence types supported by the evidence extractor.
# ---------------------------------------------------------------------------
EVIDENCE_TYPES: list[str] = [
    "shared_device",
    "shared_ip",
    "shared_payment_instrument",
    "temporal_burst",
    "high_transaction_risk",
    "high_velocity",
    "refund_pattern",
    "failed_payment_pattern",
    "entity_concentration",
]

# Severity thresholds for the "shared_*" evidence kinds (users sharing entity).
SHARED_SEVERITY: dict[str, tuple[float, float]] = {
    "low": 2.0,    # >= 2 users sharing => low
    "medium": 4.0,  # >= 4 users => medium
    "high": 8.0,    # >= 8 users => high
}


def severity_for_count(count: int | float, low: float, medium: float, high: float) -> str:
    """Map a count to low/medium/high using documented reference cut points."""
    c = float(count)
    if c >= high:
        return "high"
    if c >= medium:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Deduplication rule for overlapping candidate campaigns.
# ---------------------------------------------------------------------------
JACCARD_MERGE_THRESHOLD = 0.7


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)