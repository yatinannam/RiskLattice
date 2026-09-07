"""Pydantic response models for the RiskLattice API.

These models are the ONLY contract between the API layer and clients. They
deliberately never include ground-truth labels (is_fraud / fraud_campaign_id /
scenario) or any raw payment/credential data.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    dataset: str
    mode: str = "TEST_MODE"


class CampaignSummary(BaseModel):
    campaign_id: str
    risk_score: float
    risk_level: str
    confidence: float
    transaction_count: int
    user_count: int
    device_count: int
    ip_count: int
    payment_instrument_count: int
    exposure: float
    recommended_action: str = "NO_SAFE_ACTION"
    collateral_level: str = "LOW"


class RiskDimension(BaseModel):
    name: str
    value: float


class EvidenceItemOut(BaseModel):
    type: str
    severity: str
    description: str
    entities: list[str]
    supporting_transactions: list[str]


class CampaignDetail(CampaignSummary):
    start_time: str
    end_time: str
    duration_seconds: int
    high_risk_transaction_count: int
    risk_dimensions: list[RiskDimension]
    evidence: list[EvidenceItemOut]
    transaction_ids: list[str]
    user_ids: list[str]
    device_ids: list[str]
    ip_ids: list[str]
    payment_instrument_ids: list[str]


class GraphNodeOut(BaseModel):
    id: str
    type: str
    degree: int = 0
    transaction_count: int = 0


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    type: str
    weight: int = 1


class GraphResponse(BaseModel):
    campaign_id: str
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]


class FindingOut(BaseModel):
    type: str  # FACT | INFERENCE | UNCERTAINTY
    text: str
    evidence_ids: list[str]


class InvestigationOut(BaseModel):
    campaign_id: str
    executive_summary: str
    why_flagged: list[FindingOut]
    risk_assessment: str
    recommended_action: dict[str, Any]
    alternative_actions: list[dict[str, Any]]
    collateral_warning: str
    uncertainty: list[str]
    questions_for_reviewer: list[str]
    evidence_count: int
    provider: str
    validation_status: str
    evidence_hash: str


class ContainmentOptionOut(BaseModel):
    action_ids: list[str] = Field(default_factory=list)
    action_types: list[str] = Field(default_factory=list)
    fraud_containment_rate: float = 0.0
    fraud_exposure_contained: float = 0.0
    legitimate_users_affected: int = 0
    legitimate_transactions_affected: int = 0
    collateral_level: str = "LOW"
    collateral_risk: float = 0.0
    action_count: int = 0
    total_cost: float = 0.0


class ContainmentOut(BaseModel):
    campaign_id: str
    recommendation: str  # CONTAIN | NO_SAFE_ACTION
    recommended_action: ContainmentOptionOut | None = None
    alternative_actions: list[ContainmentOptionOut]
    collateral_metrics: dict[str, Any]
    reason: str
    constraints: dict[str, Any]
    test_mode: bool = True


class SimulateRequest(BaseModel):
    action_type: str
    target_id: str


class SimulationOut(BaseModel):
    campaign_id: str
    action_type: str
    target_id: str
    fraud_transactions_affected: int
    fraud_amount_affected: float
    legitimate_transactions_affected: int
    legitimate_amount_affected: float
    users_affected: int
    fraud_containment_rate: float
    collateral_level: str
    test_mode: bool = True
    message: str = "SIMULATION / TEST MODE - no real action executed"


class AuditEventOut(BaseModel):
    timestamp: str
    campaign: str
    event: str
    provider: str = "engine"
    validation: str = "-"
    action: str = "-"


class OverviewOut(BaseModel):
    dataset: str
    active_campaigns: int
    high_critical_campaigns: int
    fraud_exposure: float
    average_containment: float
    no_safe_action_count: int
    risk_distribution: dict[str, int]
    recent_campaigns: list[CampaignSummary]