"""Containment endpoints: recommended strategies and bounded simulation.

Simulation is TEST-MODE only. No real payment action is ever executed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_state
from app.schemas import (
    ContainmentOut,
    SimulateRequest,
    SimulationOut,
)
from app.services.app_state import AppState
from app.services.serializers import containment_out, simulation_out

router = APIRouter(prefix="/api", tags=["containment"])


@router.get(
    "/campaigns/{campaign_id}/containment",
    response_model=ContainmentOut,
    summary="Campaign containment strategies",
    description=(
        "Recommended and alternative containment strategies with estimated "
        "fraud containment, legitimate-user collateral, and collateral level. "
        "All values come from the deterministic containment engine."
    ),
)
def containment(campaign_id: str,
                state: AppState = Depends(get_state)) -> ContainmentOut:
    assessment = state.find_assessment(campaign_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return containment_out(campaign_id, state.containment_for(campaign_id))


@router.post(
    "/campaigns/{campaign_id}/containment/simulate",
    response_model=SimulationOut,
    summary="Simulate a single containment action",
    description=(
        "Simulates a bounded action (BLOCK_USER / RESTRICT_DEVICE / "
        "RESTRICT_PAYMENT_INSTRUMENT / BLOCK_TRANSACTION) against the whole "
        "dataset and returns estimated impact. SIMULATION / TEST MODE only."
    ),
)
def simulate(campaign_id: str, body: SimulateRequest,
             state: AppState = Depends(get_state)) -> SimulationOut:
    from engine.containment.actions import ActionType, TargetType

    assessment = state.find_assessment(campaign_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Campaign not found")

    try:
        action_type = ActionType(body.action_type)
    except ValueError as exc:
        raise HTTPException(status_code=400,
                            detail=f"invalid action_type: {body.action_type}") from exc

    allowed_target = {
        ActionType.BLOCK_USER: TargetType.USER,
        ActionType.REVIEW_USER: TargetType.USER,
        ActionType.RESTRICT_DEVICE: TargetType.DEVICE,
        ActionType.REVIEW_DEVICE: TargetType.DEVICE,
        ActionType.RESTRICT_PAYMENT_INSTRUMENT: TargetType.PAYMENT_INSTRUMENT,
        ActionType.REVIEW_PAYMENT_INSTRUMENT: TargetType.PAYMENT_INSTRUMENT,
        ActionType.BLOCK_TRANSACTION: TargetType.TRANSACTION,
        ActionType.REVIEW_TRANSACTION: TargetType.TRANSACTION,
    }
    target_type = allowed_target.get(action_type)
    if target_type is None:
        raise HTTPException(status_code=400,
                            detail="MONITOR/NO_ACTION are not simulatable here")

    # Validate the target belongs to this campaign's candidate actions.
    candidates = state.optimizer._generate_actions(assessment)
    match = [
        a for a in candidates
        if a.action_type == action_type and a.target_id == body.target_id
    ]
    if not match:
        raise HTTPException(
            status_code=400,
            detail=(
                f"action {body.action_type} {body.target_id} is not a valid "
                "candidate for this campaign"
            ),
        )

    sim = state.optimizer._simulate_action(
        match[0], set(assessment.transaction_ids))
    return simulation_out(campaign_id, body.action_type, body.target_id, sim)