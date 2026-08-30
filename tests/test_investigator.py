"""Tests for the AI investigator layer (Phase 6).

Covers evidence determinism, grounding, hallucination rejection, mock-provider
determinism, FACT/INFERENCE/UNCERTAINTY, NO_SAFE_ACTION, collateral, multiple
actions, no secrets, and the full workflow — all offline (no API key).
"""

from __future__ import annotations

import pytest

from engine.investigator.evidence import build_evidence, evidence_hash
from engine.investigator.investigator import (
    investigate_campaign,
    resolve_provider,
    validate_report,
)
from engine.investigator.mock_provider import MockInvestigatorProvider
from engine.investigator.schemas import (
    EvidenceFinding,
    Finding,
    InvestigationReport,
    InvestigationValidationError,
)


def _make_assessment():
    from types import SimpleNamespace

    return SimpleNamespace(
        campaign_id="CAMP_0002",
        risk_score=76.42,
        risk_level="HIGH",
        confidence=0.9,
        transaction_ids=["T1", "T2", "T3", "T4", "T5"],
        user_ids=["U1", "U2", "U3"],
        device_ids=["DEV_007"],
        ip_ids=["IP_9"],
        payment_instrument_ids=["PI_1"],
        transaction_count=5,
        user_count=3,
        device_count=1,
        ip_count=1,
        payment_instrument_count=1,
        estimated_exposure=1500.0,
        high_risk_transaction_count=3,
        transaction_risk=0.8,
        relationship_risk=0.7,
        temporal_risk=0.6,
        concentration_risk=0.4,
        behavioral_risk=0.3,
        evidence=[type("E", (), {
            "type": "shared_device",
            "severity": "high",
            "description": "3 users share DEV_007 in this campaign",
            "entities": ["DEV_007"],
            "supporting_transactions": ["T1", "T2"],
        })()],
    )


def _make_containment():
    return {
        "recommendation": "CONTAIN",
        "recommended_strategy": {
            "action_types": ["RESTRICT_PAYMENT_INSTRUMENT"],
            "fraud_containment_rate": 0.9,
            "fraud_exposure_contained": 1200.0,
            "legitimate_users_affected": 1,
            "legitimate_transactions_affected": 1,
            "collateral_level": "LOW",
        },
        "alternative_strategies": [
            {"action_types": ["BLOCK_USER"], "fraud_containment_rate": 0.96,
             "legitimate_users_affected": 3, "collateral_level": "HIGH"},
        ],
        "expected_fraud_containment": 0.9,
        "expected_fraud_exposure_contained": 1200.0,
        "expected_legitimate_users_affected": 1,
        "collateral_level": "LOW",
    }


@pytest.fixture
def evidence():
    return build_evidence(_make_assessment(), _make_containment())


@pytest.fixture
def report(evidence):
    return MockInvestigatorProvider().generate_report(evidence)


# ---------------------------------------------------------------------------
# Determinism & grounding
# ---------------------------------------------------------------------------

def test_evidence_package_deterministic():
    e1 = build_evidence(_make_assessment(), _make_containment())
    e2 = build_evidence(_make_assessment(), _make_containment())
    assert e1.to_dict() == e2.to_dict()
    assert evidence_hash(e1) == evidence_hash(e2)


def test_evidence_only_real_entities(evidence):
    all_entities = (set(evidence.user_ids) | set(evidence.device_ids)
                    | set(evidence.ip_ids) | set(evidence.payment_instrument_ids))
    for f in evidence.findings:
        for e in f.entity_ids:
            assert e in all_entities


def test_evidence_only_real_transactions(evidence):
    valid = set(evidence.transaction_ids)
    for f in evidence.findings:
        for t in f.transaction_ids:
            assert t in valid


def test_ground_truth_not_in_investigator_input(evidence):
    d = evidence.to_dict()
    for key in ("is_fraud", "fraud_campaign_id", "scenario"):
        assert key not in d


def test_all_material_claims_have_evidence_ids(report):
    for finding in report.why_flagged:
        assert finding.evidence_ids, "material claim lacks evidence_ids"


# ---------------------------------------------------------------------------
# Hallucination guard
# ---------------------------------------------------------------------------

def test_invalid_evidence_id_rejected(evidence):
    bad = InvestigationReport(
        campaign_id=evidence.campaign_id,
        executive_summary="x",
        why_flagged=[Finding(type="FACT", text="device related claim XYZ",
                             evidence_ids=["EVID_999"])],
    )
    with pytest.raises(InvestigationValidationError):
        validate_report(bad, evidence)


def test_invalid_transaction_id_rejected(evidence):
    evidence.findings.append(EvidenceFinding(
        evidence_id="EVID_X", type="X", description="d", source="s",
        transaction_ids=["NO_SUCH_TX"], severity="LOW"))
    with pytest.raises(InvestigationValidationError):
        validate_report(MockInvestigatorProvider().generate_report(evidence),
                        evidence)


def test_material_claim_without_evidence_id_rejected(evidence):
    bad = InvestigationReport(
        campaign_id=evidence.campaign_id,
        executive_summary="x",
        why_flagged=[Finding(type="FACT",
                             text="the user U1 shows suspicious activity",
                             evidence_ids=[])],
    )
    with pytest.raises(InvestigationValidationError):
        validate_report(bad, evidence)


def test_unsupported_numerical_value_rejected(evidence):
    bad = InvestigationReport(
        campaign_id=evidence.campaign_id,
        executive_summary="x",
        recommended_action={"action_types": ["RESTRICT_PAYMENT_INSTRUMENT"],
                            "fraud_containment_rate": 0.5},
    )
    with pytest.raises(InvestigationValidationError):
        validate_report(bad, evidence)


# ---------------------------------------------------------------------------
# Mock provider behavior
# ---------------------------------------------------------------------------

def test_mock_provider_deterministic(evidence):
    p = MockInvestigatorProvider()
    r1 = p.generate_report(evidence)
    r2 = p.generate_report(evidence)
    assert r1.to_dict() == r2.to_dict()


def test_fact_inference_uncertainty_supported(report):
    types = {f.type for f in report.why_flagged}
    assert types <= {"FACT", "INFERENCE", "UNCERTAINTY"}
    assert "FACT" in types


def test_no_secrets_included(report):
    blob = report.to_dict().__str__().lower()
    for forbidden in ("card", "cvv", "password", "api_key", "token", "secret"):
        assert forbidden not in blob


# ---------------------------------------------------------------------------
# Content quality
# ---------------------------------------------------------------------------

def test_collateral_included(report):
    assert report.collateral_warning
    assert "legitimate" in report.collateral_warning.lower()


def test_multiple_actions_compared(report):
    assert report.alternative_actions, "expected alternative actions"
    rec_types = report.recommended_action.get("action_types", [])
    alt_types = report.alternative_actions[0].get("action_types", [])
    assert rec_types != alt_types


def test_no_safe_action_explained():
    ev = build_evidence(_make_assessment(),
                        {"recommendation": "NO_SAFE_ACTION",
                         "recommended_strategy": None})
    r = MockInvestigatorProvider().generate_report(ev)
    assert "NO_SAFE_ACTION" in r.executive_summary
    assert any("no safe automated" in q.lower()
               for q in r.questions_for_reviewer)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

def test_full_investigation_workflow():
    res = investigate_campaign(_make_assessment(), _make_containment())
    assert res["validation_status"] == "VALID"
    assert res["provider"] == "mock"
    assert res["report"]["campaign_id"] == "CAMP_0002"
    assert res["report"]["audit_trail"]["validation_status"] == "VALID"


def test_resolve_provider_offline_no_key():
    assert resolve_provider("openai").name == "mock"   # honest fallback
    assert resolve_provider(None).name == "mock"