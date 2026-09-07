"""Risk overview endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_state
from app.schemas import OverviewOut
from app.services.app_state import AppState
from app.services.serializers import (
    overview_out,
    summary_from_assessment,
)

router = APIRouter(prefix="/api", tags=["overview"])


@router.get(
    "/overview",
    response_model=OverviewOut,
    summary="Risk overview",
    description=(
        "Dense risk overview: campaign counts, exposure, containment rate, "
        "risk distribution, and the most recent high-risk campaigns. No "
        "ground-truth labels are exposed."
    ),
)
def overview(state: AppState = Depends(get_state)) -> OverviewOut:
    summaries = [
        summary_from_assessment(a, state.containment_for(a.campaign_id))
        for a in state.assessments
    ]
    return overview_out(state, summaries)