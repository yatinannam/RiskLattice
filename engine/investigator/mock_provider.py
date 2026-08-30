"""Deterministic MockInvestigatorProvider (Phase 6).

Generates a grounded, reproducible InvestigationReport purely from the evidence
package. No LLM, no API key, no random output. Every material claim cites an
evidence_id; FACT / INFERENCE / UNCERTAINTY are distinguished; collateral and
NO_SAFE_ACTION are first-class outcomes.
"""

from __future__ import annotations

from engine.investigator.investigator import InvestigatorProvider
from engine.investigator.prompts import PROMPT_VERSION
from engine.investigator.schemas import (
    Finding,
    InvestigationEvidence,
    InvestigationReport,
)

UNCERTAINTY_TEMPLATES = [
    "Shared infrastructure is not itself proof of fraud; the entities above "
    "are associated with the suspicious campaign, not declared fraudulent.",
    "Observed relationships are correlational and may include legitimate "
    "activity on the same entities.",
    "The transaction-level model provides weak-to-moderate individual "
    "evidence; the campaign assessment relies on the combination of signals.",
]


class MockInvestigatorProvider(InvestigatorProvider):
    """Grounds every claim in the supplied evidence only (no invention)."""

    name = "mock"

    def generate_report(self, evidence: InvestigationEvidence) -> InvestigationReport:
        by_type = _priority_findings(evidence.findings)

        return InvestigationReport(
            campaign_id=evidence.campaign_id,
            executive_summary=_build_executive_summary(evidence),
            why_flagged=_build_findings(evidence, by_type),
            risk_assessment=_build_risk_assessment(evidence),
            recommended_action=(evidence.recommended_action or {}),
            alternative_actions=(evidence.containment_options or []),
            collateral_warning=_build_collateral_warning(evidence),
            uncertainty=list(evidence.uncertainty_flags) + list(UNCERTAINTY_TEMPLATES),
            questions_for_reviewer=_build_questions(evidence),
            audit_trail={
                "provider": self.name,
                "model": "deterministic-mock",
                "prompt_version": PROMPT_VERSION,
                "evidence_hash": _evidence_hash(evidence),
                "validation_status": "PENDING",
            },
            provider=self.name,
            model="deterministic-mock",
        )


def _priority_findings(findings):
    """Return findings ordered by declared priority (risk → graph → time)."""
    order = {
        "HIGH_TRANSACTION_RISK": 0,
        "SHARED_DEVICE": 1,
        "SHARED_IP": 2,
        "SHARED_PAYMENT_INSTRUMENT": 3,
        "TEMPORAL_BURST": 4,
        "HIGH_VELOCITY": 5,
        "ENTITY_CONCENTRATION": 6,
        "REFUND_PATTERN": 7,
        "FAILED_PAYMENT_PATTERN": 8,
    }
    return sorted(findings, key=lambda f: order.get(f.type, 99))


def _build_findings(evidence, prioritized) -> list[Finding]:
    out: list[Finding] = []
    for f in prioritized[:4]:  # strongest independent signals, no repeats
        out.append(Finding(type="FACT", text=f.description,
                           evidence_ids=[f.evidence_id]))
        if f.type in ("SHARED_DEVICE", "SHARED_IP", "SHARED_PAYMENT_INSTRUMENT"):
            kind = f.type.replace("SHARED_", "").replace("_", " ").lower()
            out.append(Finding(
                type="INFERENCE",
                text=(
                    f"Multiple accounts on the same {kind} within the "
                    "campaign window is consistent with coordinated activity."
                ),
                evidence_ids=[f.evidence_id],
            ))
            out.append(Finding(
                type="UNCERTAINTY",
                text=(
                    "Shared usage can be legitimate (offices, households, "
                    "sequential device owners); it does not establish fraud."
                ),
                evidence_ids=[f.evidence_id],
            ))
    return out


def _build_executive_summary(evidence) -> str:
    rec = evidence.recommended_action or {}
    action_types = rec.get("action_types", []) or []
    rate = evidence.collateral_metrics.get("recommended_fraud_containment_rate", 0.0)
    if evidence.risk_level in ("HIGH", "CRITICAL") and action_types:
        return (
            f"Campaign {evidence.campaign_id} is assessed as "
            f"{evidence.risk_level} risk ({evidence.campaign_score:.1f}/100) "
            f"with confidence {evidence.confidence:.2f}. The deterministic "
            f"containment heuristic recommends {', '.join(action_types)} "
            f"containing {rate:.0%} of observed suspicious exposure."
        )
    if not action_types:
        return (
            f"Campaign {evidence.campaign_id} is assessed as "
            f"{evidence.risk_level} risk ({evidence.campaign_score:.1f}/100) "
            "with **NO_SAFE_ACTION**: no bounded strategy reaches the "
            "configured fraud-containment threshold without exceeding the "
            "legitimate-user collateral cap."
        )
    return (
        f"Campaign {evidence.campaign_id} is assessed as {evidence.risk_level} "
        f"risk ({evidence.campaign_score:.1f}/100)."
    )


def _build_risk_assessment(evidence) -> str:
    d = evidence.risk_dimensions
    return (
        "Risk dimensions (deterministic): "
        f"transaction={d.transaction_risk:.2f}, "
        f"relationship={d.relationship_risk:.2f}, "
        f"temporal={d.temporal_risk:.2f}, "
        f"concentration={d.concentration_risk:.2f}, "
        f"behavioral={d.behavioral_risk:.2f}."
    )


def _build_collateral_warning(evidence) -> str:
    m = evidence.collateral_metrics
    if not m:
        return ""
    return (
        "Legitimate collateral (simulated): the recommended strategy affects "
        f"{m.get('recommended_legitimate_users_affected', 0)} legitimate "
        f"user(s), collateral level "
        f"{m.get('recommended_collateral_level', 'LOW')}. Blocking an entire "
        "shared entity can affect legitimate customers; the deterministic "
        "optimizer chose the strategy with the best "
        "containment-to-collateral trade-off."
    )


def _build_questions(evidence) -> list[str]:
    q = [
        "Is the shared infrastructure (device/IP/payment instrument) known to "
        "be shared by legitimate users in normal operation?",
        "Were there manual chargebacks or merchant reports that corroborate "
        "or contradict the suspicious activity?",
    ]
    if not evidence.recommended_action:
        q.insert(0, (
            "Should the campaign be escalated to manual review given that no "
            "safe automated containment action exists?"
        ))
    return q


def _evidence_hash(evidence: InvestigationEvidence) -> str:
    from engine.investigator.evidence import evidence_hash

    return evidence_hash(evidence)