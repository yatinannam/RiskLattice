"""Deterministic, explainable campaign risk scoring (Phase 4).

Five normalized dimensions (each 0..1) feed a documented weighted blend to a
0..100 campaign score. No LLM, no opaque model, no ground-truth labels.

Formulas (documented in docs/architecture/architecture.md):

  transaction_risk     = 0.45*mean_risk + 0.35*high_ratio + 0.20*p90_risk
  relationship_risk    = 0.35*density  + 0.35*fanout       + 0.30*multi_signal
  temporal_risk        = clip(tx_per_hour / 120, 0, 1)
  concentration_risk   = log-scaled mean over entity types of (users, tx)
  behavioral_risk      = 0.40*refund_ratio + 0.30*failed_ratio
                         + 0.30*amount_top_share
  campaign_score       = sum(weight * dimension) * 100   (weights in rules.py)

Confidence is separate from risk.
"""

from __future__ import annotations

import math

import numpy as np

from engine.risk.rules import (
    CONFIDENCE_W,
    DIMENSION_WEIGHTS,
    ENTITY_TX_REF,
    HIGH_RISK_TX_THRESHOLD,
    RELATIONSHIP_RISK_BALANCE,
    SHARED_ENTITY_REF_USERS,
    SIGNAL_TRIGGER,
    TEMPORAL_RATE_REF_PER_HOUR,
    TRANSACTION_RISK_BALANCE,
)


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def _log1p(value: float, ref: float) -> float:
    """Log-scaled normalization: clip(log1p(value)/log1p(ref), 0, 1)."""
    if value <= 0:
        return 0.0
    return _clip(math.log1p(value) / math.log1p(ref))


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------

def transaction_risk_score(risks: list[float]) -> float:
    """0..1 from Phase-2 probabilities (mean, high ratio, 90th percentile)."""
    if not risks:
        return 0.0
    arr = np.asarray(risks, dtype=float)
    mean_r = float(arr.mean())
    high_ratio = float((arr >= HIGH_RISK_TX_THRESHOLD).mean())
    p90 = float(np.percentile(arr, 90))
    return _clip(
        TRANSACTION_RISK_BALANCE["mean"] * mean_r
        + TRANSACTION_RISK_BALANCE["high_ratio"] * high_ratio
        + TRANSACTION_RISK_BALANCE["p90"] * p90
    )


def relationship_risk_score(
    relationship_density: float,
    max_shared_users_per_device: int,
    max_shared_users_per_ip: int,
    max_shared_users_per_pi: int,
) -> float:
    """0..1 weighing density, entity fanout, and number of independent signals.

    Shared infrastructure alone does NOT saturate this: fanout is log-scaled,
    and part of the score rewards *multiple distinct* signals rather than any
    single one.
    """
    fanout = _log1p(
        max(max_shared_users_per_device, max_shared_users_per_ip, max_shared_users_per_pi),
        SHARED_ENTITY_REF_USERS,
    )
    signals = 0
    for count in (max_shared_users_per_device, max_shared_users_per_ip, max_shared_users_per_pi):
        if count >= 4:  # documented reference for "meaningful sharing"
            signals += 1
    multi = signals / 3.0

    return _clip(
        RELATIONSHIP_RISK_BALANCE["density"] * _clip(relationship_density)
        + RELATIONSHIP_RISK_BALANCE["fanout"] * fanout
        + RELATIONSHIP_RISK_BALANCE["multi_signal"] * multi
    )


def temporal_risk_score(transaction_count: int, duration_seconds: int) -> float:
    """0..1 from sustained transactions-per-hour vs. documented reference.

    20 tx in 4 minutes => ~300/hr (saturates); 20 tx over 30 days => ~0.03/hr.
    """
    if duration_seconds <= 0 or transaction_count <= 0:
        return 0.0
    hours = max(duration_seconds / 3600.0, 1.0 / 3600.0)
    per_hour = transaction_count / hours
    return _clip(per_hour / TEMPORAL_RATE_REF_PER_HOUR)


def concentration_risk_score(
    tx_per_device: list[int],
    tx_per_ip: list[int],
    tx_per_pi: list[int],
    users_per_device: list[int],
    users_per_ip: list[int],
    users_per_pi: list[int],
) -> float:
    """Log-scaled concentration of activity around campaign entities."""
    factors: list[float] = []
    for tx_list, u_list in ((tx_per_device, users_per_device),
                            (tx_per_ip, users_per_ip),
                            (tx_per_pi, users_per_pi)):
        if tx_list:
            factors.append(_log1p(max(tx_list), ENTITY_TX_REF))
        if u_list:
            factors.append(_log1p(max(u_list), SHARED_ENTITY_REF_USERS))
    return _clip(sum(factors) / len(factors)) if factors else 0.0


def behavioral_risk_score(statuses: list[str], amounts: list[float]) -> float:
    """0..1 from refund ratio, failed ratio, and amount concentration."""
    if not statuses or not amounts:
        return 0.0
    n = len(statuses)
    refund_r = sum(1 for s in statuses if s == "refunded") / n
    failed_r = sum(1 for s in statuses if s == "failed") / n
    top_share = max(amounts) / sum(amounts) if sum(amounts) > 0 else 0.0
    return _clip(0.40 * refund_r + 0.30 * failed_r + 0.30 * top_share)


# ---------------------------------------------------------------------------
# Combined score & confidence
# ---------------------------------------------------------------------------

def combined_campaign_score(dimensions: dict[str, float]) -> float:
    """Weighted blend of the five dimensions -> 0..100 (documented heuristic)."""
    total = sum(
        weight * _clip(dimensions.get(name, 0.0))
        for name, weight in DIMENSION_WEIGHTS.items()
    )
    return round(total * 100.0, 2)


def campaign_confidence(
    dimensions: dict[str, float],
    n_transactions: int,
    completeness: float,
    risk_dispersion: float,
    graph_evidence_strength: float,
) -> float:
    """Confidence (0..1) in the assessment — separate from risk on purpose.

    High when many independent dimensions are elevated, entity coverage is
    complete, per-transaction risk agrees, and graph evidence is substantive.
    """
    strong = sum(1 for value in dimensions.values() if value >= SIGNAL_TRIGGER)
    signal_fraction = strong / max(len(DIMENSION_WEIGHTS), 1)

    temporal = dimensions.get("temporal_risk", 0.0)
    temporal_richness = 1.0 if (n_transactions >= 3 and 0.0 < temporal < 1.0) else 0.0

    confidence = (
        CONFIDENCE_W["signal_fraction"] * signal_fraction
        + CONFIDENCE_W["completeness"] * _clip(completeness)
        + CONFIDENCE_W["risk_consistency"] * _clip(1.0 - risk_dispersion)
        + CONFIDENCE_W["graph_evidence"] * _clip(graph_evidence_strength)
        + CONFIDENCE_W["temporal_richness"] * temporal_richness
    )
    return round(_clip(confidence), 4)