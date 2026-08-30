"""Deterministic evidence-package builder (Phase 6).

Turns a CampaignAssessment + a validated containment recommendation into a
grounded InvestigationEvidence package. Every finding here is derived from the
structured output of the risk/graph/containment engines — nothing is inferred
from raw data, and no ground-truth field is ever included.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine.investigator.schemas import (
    EvidenceFinding,
    InvestigationEvidence,
    RiskDimensions,
)


def evidence_hash(package: InvestigationEvidence) -> str:
    """Stable hash of the package's canonical JSON (for audit trail)."""
    payload = json.dumps(package.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_risk_dimensions(assessment) -> RiskDimensions:
    return RiskDimensions(
        transaction_risk=float(assessment.transaction_risk),
        relationship_risk=float(assessment.relationship_risk),
        temporal_risk=float(assessment.temporal_risk),
        concentration_risk=float(assessment.concentration_risk),
        behavioral_risk=float(assessment.behavioral_risk),
    )


def _uncertainty_flags(assessment) -> list[str]:
    """Deterministic uncertainty notes derived from evidence."""
    flags = []
    if assessment.confidence < 0.6:
        flags.append("Low model confidence in the campaign assessment.")
    if assessment.relationship_risk > 0.5 or assessment.temporal_risk > 0.5:
        flags.append(
            "Shared infrastructure / temporal concentration is correlational, "
            "not proof of fraud."
        )
    # If any similar entity count exists it may contain legitimate users.
    if assessment.device_count > 0 or assessment.ip_count > 0:
        flags.append(
            "The campaign entities may include legitimate users; collateral "
            "is described by containment simulation."
        )
    return flags


def build_evidence(assessment, containment: dict, campaign=None) -> InvestigationEvidence:
    """Build a grounded evidence package from risk + containment output."""
    findings: list[EvidenceFinding] = []

    # 1. Transaction-level risk evidence (source risk_engine).
    if getattr(assessment, "high_risk_transaction_count", 0) > 0 and assessment.transaction_count > 0:
        findings.append(EvidenceFinding(
            evidence_id="EVID_TX_RISK",
            type="HIGH_TRANSACTION_RISK",
            description=(
                f"{assessment.high_risk_transaction_count}/{assessment.transaction_count} "
                "transactions have elevated transaction-level risk."
            ),
            source="risk_engine",
            transaction_ids=list(assessment.transaction_ids),
            severity="HIGH" if assessment.high_risk_transaction_count >= 3 else "MEDIUM",
        ))

    # 2. Relationship evidence from graph evidence items.
    for i, item in enumerate(assessment.evidence, start=1):
        etype = item.type.upper()
        findings.append(EvidenceFinding(
            evidence_id=f"EVID_{i:03d}",
            type=etype,
            description=str(item.description),
            source="graph_features",
            entity_ids=list(item.entities),
            transaction_ids=list(item.supporting_transactions),
            severity=str(item.severity).upper(),
        ))

    # Deduplicate by type (e.g. extract_evidence may already emit
    # HIGH_TRANSACTION_RISK alongside our explicit EVID_TX_RISK), keeping the
    # first occurrence so the report never repeats the same signal.
    seen: set[str] = set()
    unique: list[EvidenceFinding] = []
    for finding in findings:
        if finding.type in seen:
            continue
        seen.add(finding.type)
        unique.append(finding)
    findings = unique

    # 3. Containment options & recommended action (source containment).
    containment_options = [
        s for s in containment.get("alternative_strategies", [])
    ]
    recommended = containment.get("recommended_strategy", {})

    collateral_metrics = {
        "recommended_fraud_containment_rate": containment.get(
            "expected_fraud_containment", 0.0),
        "recommended_fraud_exposure_contained": containment.get(
            "expected_fraud_exposure_contained", 0.0),
        "recommended_legitimate_users_affected": containment.get(
            "expected_legitimate_users_affected", 0),
        "recommended_collateral_level": containment.get("collateral_level", "LOW"),
    }

    return InvestigationEvidence(
        campaign_id=str(assessment.campaign_id),
        campaign_score=float(assessment.risk_score),
        risk_level=str(assessment.risk_level),
        confidence=float(assessment.confidence),
        transaction_ids=list(assessment.transaction_ids),
        user_ids=list(getattr(assessment, "user_ids", [])),
        device_ids=list(getattr(assessment, "device_ids", [])),
        ip_ids=list(getattr(assessment, "ip_ids", [])),
        payment_instrument_ids=list(getattr(assessment, "payment_instrument_ids", [])),
        transaction_count=int(assessment.transaction_count),
        total_exposure=float(assessment.estimated_exposure),
        risk_dimensions=build_risk_dimensions(assessment),
        findings=findings,
        containment_options=containment_options,
        recommended_action=recommended,
        collateral_metrics=collateral_metrics,
        uncertainty_flags=_uncertainty_flags(assessment),
    )