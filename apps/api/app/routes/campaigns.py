"""Campaign listing, detail, and graph endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_state
from app.schemas import CampaignDetail, CampaignSummary, GraphResponse
from app.services.app_state import AppState
from app.services.serializers import (
    detail_from_assessment,
    graph_response,
    summary_from_assessment,
)

router = APIRouter(prefix="/api", tags=["campaigns"])

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _campaign_graph(state: AppState, assessment) -> GraphResponse:
    """Build a small campaign subgraph (campaign transactions + entities)."""
    from engine.graph.graph_builder import build_graph, serialize_graph

    subset = state.df[state.df["transaction_id"].isin(assessment.transaction_ids)]
    sub = build_graph(subset) if len(subset) else None
    raw = serialize_graph(sub) if sub else {"nodes": [], "edges": []}
    return graph_response(assessment.campaign_id, raw)


@router.get(
    "/campaigns",
    response_model=list[CampaignSummary],
    summary="List assessed campaigns",
    description=(
        "All assessed campaigns with filters (risk level, minimum score, "
        "recommended action) and sorting (risk score, exposure, transaction "
        "count). Ground-truth labels are never exposed."
    ),
)
def list_campaigns(
    risk_level: str | None = Query(default=None),
    min_score: float | None = Query(default=None),
    recommended_action: str | None = Query(default=None),
    sort_by: str = Query(default="risk_score",
                         pattern="^(risk_score|exposure|transaction_count)$"),
    state: AppState = Depends(get_state),
) -> list[CampaignSummary]:
    summaries = [
        summary_from_assessment(a, state.containment_for(a.campaign_id))
        for a in state.assessments
    ]
    if risk_level:
        summaries = [s for s in summaries if s.risk_level == risk_level]
    if min_score is not None:
        summaries = [s for s in summaries if s.risk_score >= min_score]
    if recommended_action:
        summaries = [
            s for s in summaries
            if recommended_action.upper() in s.recommended_action.upper()
        ]

    key = {"risk_score": lambda s: (-s.risk_score, s.campaign_id),
           "exposure": lambda s: (-s.exposure, s.campaign_id),
           "transaction_count": lambda s: (-s.transaction_count, s.campaign_id)}
    return sorted(summaries, key=key[sort_by])


@router.get(
    "/campaigns/{campaign_id}",
    response_model=CampaignDetail,
    summary="Campaign detail",
    description="Full assessment detail: risk dimensions, evidence items, entity IDs.",
)
def campaign_detail(campaign_id: str,
                    state: AppState = Depends(get_state)) -> CampaignDetail:
    assessment = state.find_assessment(campaign_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return detail_from_assessment(
        assessment, state.containment_for(campaign_id))


@router.get(
    "/campaigns/{campaign_id}/graph",
    response_model=GraphResponse,
    summary="Campaign relationship graph",
    description="Typed relationship nodes/edges for the campaign.",
)
def campaign_graph(campaign_id: str,
                   state: AppState = Depends(get_state)) -> GraphResponse:
    assessment = state.find_assessment(campaign_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return _campaign_graph(state, assessment)