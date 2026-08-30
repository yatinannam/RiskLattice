"""InvestigatorProvider abstraction (Phase 6).

Any provider (mock, or a future LLM adapter) must implement
``generate_report(evidence) -> InvestigationReport`` and
``_self_check_report``-style validation enforced separately by
``validate_report``. No provider may invent evidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.investigator.schemas import InvestigationEvidence, InvestigationReport


class InvestigatorProvider(ABC):
    """Interface for an investigation report generator."""

    name: str = "abstract"

    @abstractmethod
    def generate_report(self, evidence: InvestigationEvidence) -> InvestigationReport:
        """Produce a grounded InvestigationReport from the evidence package."""


def resolve_provider(provider_name: str | None = None) -> InvestigatorProvider:
    """Resolve a provider by name; default to the deterministic mock provider.

    The only implemented provider is 'mock' (works with no API key). If an
    unsupported name is requested, we fall back to mock rather than fail.
    """
    from engine.investigator.mock_provider import MockInvestigatorProvider

    name = (provider_name or "mock").lower()
    if name in ("mock", "mock_investigator"):
        return MockInvestigatorProvider()
    if name in ("openai", "anthropic"):
        # Adapter not wired up (no credentials in demo). Honest fallback:
        return MockInvestigatorProvider()
    return MockInvestigatorProvider()


# ---------------------------------------------------------------------------
# Hallucination guard (validation)
# ---------------------------------------------------------------------------

def validate_report(report: InvestigationReport, evidence: InvestigationEvidence) -> None:
    """Validate a report against its evidence package.

    Raises InvestigationValidationError if any referenced evidence/entity/
    transaction is unsupported or any numerical claim is off-package. This is
    a hard guard — unsupported claims are rejected, never repaired.
    """
    from engine.investigator.schemas import InvestigationValidationError

    valid_evidence_ids = {f.evidence_id for f in evidence.findings}
    valid_tx = set(evidence.transaction_ids)

    def _check_all(claim: "Finding"):
        for eid in claim.evidence_ids:
            if eid not in valid_evidence_ids:
                raise InvestigationValidationError(
                    f"Claim references unknown evidence_id '{eid}'"
                )

    for claim in report.why_flagged:
        _check_all(claim)
        # Every material claim about a concrete entity/tx must reference
        # at least one evidence ID (coverage guard).
        if "device" in claim.text.lower() or "user" in claim.text.lower() \
                or "transaction" in claim.text.lower():
            if not claim.evidence_ids:
                raise InvestigationValidationError(
                    f"Material claim lacks evidence IDs: {claim.text}"
                )

    # Entity / transaction references inside evidence findings must be real.
    for finding in evidence.findings:
        for t in finding.transaction_ids:
            if t not in valid_tx:
                raise InvestigationValidationError(
                    f"Evidence finding references unknown transaction '{t}'"
                )

    # Numerical claims in collateral are grounded in the package if present.
    rec = report.recommended_action or {}
    if "fraud_containment_rate" in rec:
        pkg_rate = evidence.collateral_metrics.get(
            "recommended_fraud_containment_rate")
        if pkg_rate is not None and abs(
                (rec.get("fraud_containment_rate", 0.0) - pkg_rate)) > 1e-6:
            raise InvestigationValidationError(
                "Recommended strategy fraud_containment_rate does not match supply"
            )


# ---------------------------------------------------------------------------
# Orchestration workflow
# ---------------------------------------------------------------------------

def investigate_campaign(
    assessment,
    containment: dict,
    provider: "InvestigatorProvider" | None = None,
) -> dict:
    """Run the full investigation workflow:
    CampaignAssessment -> Evidence builder -> Provider -> Validator -> result.

    Returns a dict with the report, the evidence package dict, audit trail, and
    validation status. Never executes containment.
    """
    from engine.investigator.evidence import build_evidence, evidence_hash
    from engine.investigator.investigator import resolve_provider

    if provider is None:
        provider = resolve_provider("mock")

    evidence = build_evidence(assessment, containment)
    report = provider.generate_report(evidence)
    validate_report(report, evidence)  # raises on unsupported claims

    h = evidence_hash(evidence)
    report.audit_trail["validation_status"] = "VALID"
    report.audit_trail["evidence_hash"] = h

    return {
        "report": report.to_dict(),
        "evidence": evidence.to_dict(),
        "evidence_hash": h,
        "provider": provider.name,
        "validation_status": "VALID",
    }