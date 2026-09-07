"""Mappers from prepared engine state to API response models.

No business logic lives here — this is pure serialization. Ground-truth fields
are never passed through.
"""

from __future__ import annotations

from app.schemas import (
    AuditEventOut,
    CampaignDetail,
    CampaignSummary,
    ContainmentOptionOut,
    ContainmentOut,
    EvidenceItemOut,
    FindingOut,
    GraphEdgeOut,
    GraphNodeOut,
    GraphResponse,
    InvestigationOut,
    OverviewOut,
    RiskDimension,
    SimulationOut,
)

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _recommended_action_label(containment: dict) -> str:
    rec = containment.get("recommended_strategy") or {}
    types = rec.get("action_types") or []
    return ", ".join(types) if types else "NO_SAFE_ACTION"


def summary_from_assessment(assessment, containment: dict) -> CampaignSummary:
    rec = containment.get("recommended_strategy") or {}
    return CampaignSummary(
        campaign_id=assessment.campaign_id,
        risk_score=assessment.risk_score,
        risk_level=assessment.risk_level,
        confidence=assessment.confidence,
        transaction_count=assessment.transaction_count,
        user_count=assessment.user_count,
        device_count=assessment.device_count,
        ip_count=assessment.ip_count,
        payment_instrument_count=assessment.payment_instrument_count,
        exposure=assessment.estimated_exposure,
        recommended_action=_recommended_action_label(containment),
        collateral_level=rec.get("collateral_level", "LOW") if rec else "LOW",
    )


def detail_from_assessment(assessment, containment: dict) -> CampaignDetail:
    base = summary_from_assessment(assessment, containment)
    dimensions = [
        RiskDimension(name="transaction", value=assessment.transaction_risk),
        RiskDimension(name="relationship", value=assessment.relationship_risk),
        RiskDimension(name="temporal", value=assessment.temporal_risk),
        RiskDimension(name="concentration", value=assessment.concentration_risk),
        RiskDimension(name="behavioral", value=assessment.behavioral_risk),
    ]
    evidence = [
        EvidenceItemOut(
            type=item.type,
            severity=item.severity,
            description=str(item.description),
            entities=list(item.entities),
            supporting_transactions=list(item.supporting_transactions),
        )
        for item in assessment.evidence
    ]
    return CampaignDetail(
        **{k: getattr(base, k) for k in base.model_fields_set},
        start_time=assessment.start_time,
        end_time=assessment.end_time,
        duration_seconds=assessment.duration_seconds,
        high_risk_transaction_count=assessment.high_risk_transaction_count,
        risk_dimensions=dimensions,
        evidence=evidence,
        transaction_ids=list(assessment.transaction_ids),
        user_ids=list(getattr(assessment, "user_ids", [])),
        device_ids=list(getattr(assessment, "device_ids", [])),
        ip_ids=list(getattr(assessment, "ip_ids", [])),
        payment_instrument_ids=list(
            getattr(assessment, "payment_instrument_ids", [])),
    )


def graph_response(campaign_id: str, raw: dict) -> GraphResponse:
    """Map an engine-serialized graph ({"nodes", "edges"}) to API models."""
    degree: dict[str, int] = {}
    for edge in raw.get("edges", []):
        weight = edge.get("weight", 1)
        for node_id in (edge["source"], edge["target"]):
            degree[node_id] = degree.get(node_id, 0) + weight

    nodes = [
        GraphNodeOut(
            id=node["id"],
            type=node["type"],
            degree=degree.get(node["id"], 0),
            transaction_count=1 if node["type"] == "TRANSACTION" else 0,
        )
        for node in raw.get("nodes", [])
    ]
    edges = [
        GraphEdgeOut(
            source=edge["source"],
            target=edge["target"],
            type=edge.get("type", "RELATED"),
            weight=edge.get("weight", 1),
        )
        for edge in raw.get("edges", [])
    ]
    return GraphResponse(campaign_id=campaign_id, nodes=nodes, edges=edges)


def investigation_out(campaign_id: str, inv: dict) -> InvestigationOut:
    report = inv.get("report", {})
    evidence = inv.get("evidence", {})
    why_flagged = [
        FindingOut(type=f.get("type", "FACT"), text=f.get("text", ""),
                   evidence_ids=list(f.get("evidence_ids", [])))
        for f in report.get("why_flagged", [])
    ]
    return InvestigationOut(
        campaign_id=report.get("campaign_id", campaign_id),
        executive_summary=report.get("executive_summary", ""),
        why_flagged=why_flagged,
        risk_assessment=report.get("risk_assessment", ""),
        recommended_action=report.get("recommended_action", {}),
        alternative_actions=report.get("alternative_actions", []),
        collateral_warning=report.get("collateral_warning", ""),
        uncertainty=report.get("uncertainty", []),
        questions_for_reviewer=report.get("questions_for_reviewer", []),
        evidence_count=len(evidence.get("findings", [])),
        provider=inv.get("provider", "mock"),
        validation_status=inv.get("validation_status", "ERROR"),
        evidence_hash=inv.get("evidence_hash", ""),
    )


def _containment_option_out(raw: dict) -> ContainmentOptionOut:
    return ContainmentOptionOut(
        action_ids=list(raw.get("action_ids", [])),
        action_types=list(raw.get("action_types", [])),
        fraud_containment_rate=raw.get("fraud_containment_rate", 0.0),
        fraud_exposure_contained=raw.get("fraud_exposure_contained", 0.0),
        legitimate_users_affected=raw.get("legitimate_users_affected", 0),
        legitimate_transactions_affected=raw.get(
            "legitimate_transactions_affected", 0),
        collateral_level=raw.get("collateral_level", "LOW"),
        collateral_risk=raw.get("collateral_risk", 0.0),
        action_count=raw.get("action_count", len(raw.get("action_types", []))),
        total_cost=raw.get("total_cost", 0.0),
    )


def containment_out(campaign_id: str, containment: dict) -> ContainmentOut:
    recommended = containment.get("recommended_strategy")
    return ContainmentOut(
        campaign_id=campaign_id,
        recommendation=containment.get("recommendation", "NO_SAFE_ACTION"),
        recommended_action=(_containment_option_out(recommended)
                            if recommended else None),
        alternative_actions=[
            _containment_option_out(s)
            for s in containment.get("alternative_strategies", [])
        ],
        collateral_metrics={
            k: containment.get(k) for k in (
                "expected_fraud_containment",
                "expected_fraud_exposure_contained",
                "expected_legitimate_users_affected",
                "collateral_risk",
                "collateral_level",
            )
        },
        reason=containment.get("reason", ""),
        constraints=containment.get("audit_record", {}).get("constraints", {}),
        test_mode=True,
    )


def simulation_out(campaign_id: str, action_type: str, target_id: str,
                   sim) -> SimulationOut:
    return SimulationOut(
        campaign_id=campaign_id,
        action_type=action_type,
        target_id=target_id,
        fraud_transactions_affected=sim.fraud_transactions_affected,
        fraud_amount_affected=sim.fraud_amount_affected,
        legitimate_transactions_affected=sim.legitimate_transactions_affected,
        legitimate_amount_affected=sim.legitimate_amount_affected,
        users_affected=sim.users_affected,
        fraud_containment_rate=sim.fraud_containment_rate,
        collateral_level=sim.collateral_level,
        test_mode=True,
    )


def overview_out(state, summaries: list[CampaignSummary]) -> OverviewOut:
    risk_dist: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    high_critical = 0
    for assessment in state.assessments:
        risk_dist[assessment.risk_level] = (
            risk_dist.get(assessment.risk_level, 0) + 1)
        if RISK_ORDER[assessment.risk_level] >= RISK_ORDER["HIGH"]:
            high_critical += 1

    no_safe = sum(1 for c in state.containments.values()
                  if c.get("recommendation") == "NO_SAFE_ACTION")
    contain_rates = [
        c.get("recommended_strategy", {}).get("fraud_containment_rate", 0.0)
        for c in state.containments.values()
        if c.get("recommended_strategy")
    ]
    avg_contain = (sum(contain_rates) / len(contain_rates)) if contain_rates else 0.0
    total_exposure = sum(
        c.get("recommended_strategy", {}).get("fraud_exposure_contained", 0.0)
        for c in state.containments.values()
        if c.get("recommended_strategy")
    )

    return OverviewOut(
        dataset=state.dataset_name,
        active_campaigns=len(state.assessments),
        high_critical_campaigns=high_critical,
        fraud_exposure=total_exposure,
        average_containment=round(avg_contain, 4),
        no_safe_action_count=no_safe,
        risk_distribution=risk_dist,
        recent_campaigns=summaries[:10],
    )


def audit_events(state) -> list[AuditEventOut]:
    """Deterministic chronological audit stream derived from prepared state."""
    events: list[AuditEventOut] = [AuditEventOut(
        timestamp="startup",
        campaign="-",
        event=f"Dataset '{state.dataset_name}' loaded and pipeline prepared "
              f"({len(state.assessments)} campaigns, {state.startup_seconds}s)",
        provider="engine",
    )]
    for assessment in state.assessments[:25]:
        cid = assessment.campaign_id
        inv = state.investigations.get(cid, {})  # cached only; never force-build
        containment = state.containment_for(cid)
        ts = assessment.start_time
        events.append(AuditEventOut(
            timestamp=ts, campaign=cid, event="Campaign assessed",
            provider="risk_engine", validation=assessment.risk_level,
            action=f"score {assessment.risk_score:.1f}",
        ))
        events.append(AuditEventOut(
            timestamp=ts, campaign=cid, event="Evidence assembled",
            provider="engine", action=f"{len(assessment.evidence)} evidence items",
        ))
        events.append(AuditEventOut(
            timestamp=ts, campaign=cid, event="Containment simulated",
            provider="containment",
            validation=containment.get("recommendation", "NO_SAFE_ACTION"),
            action=str((containment.get("recommended_strategy") or {})
                       .get("action_types", [])),
        ))
        events.append(AuditEventOut(
            timestamp=ts, campaign=cid, event="Investigation created",
            provider=inv.get("provider", "mock"),
            validation=inv.get("validation_status", "-"),
        ))
        events.append(AuditEventOut(
            timestamp=ts, campaign=cid, event="Recommendation generated",
            provider="containment",
            validation=containment.get("recommendation", "-"),
        ))
    return events