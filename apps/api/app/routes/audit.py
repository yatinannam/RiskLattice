"""Audit trail endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_state
from app.schemas import AuditEventOut
from app.services.app_state import AppState
from app.services.serializers import audit_events

router = APIRouter(prefix="/api", tags=["audit"])


@router.get(
    "/audit",
    response_model=list[AuditEventOut],
    summary="Audit trail",
    description=(
        "Chronological decision/evidence/containment events derived from the "
        "prepared state. All actions remain SIMULATED / TEST MODE."
    ),
)
def audit(state: AppState = Depends(get_state)) -> list[AuditEventOut]:
    return audit_events(state)