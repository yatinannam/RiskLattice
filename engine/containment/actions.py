"""Action model and semantics for RiskLattice containment (Phase 5).

Actions are structured, immutable descriptions of a candidate intervention.
They are NEVER executed: every consequential action is SIMULATED / TEST-MODE
until explicit merchant approval (which is out of scope until the API/dashboard
phases).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Allowed intervention types."""

    BLOCK_TRANSACTION = "BLOCK_TRANSACTION"
    REVIEW_TRANSACTION = "REVIEW_TRANSACTION"
    BLOCK_USER = "BLOCK_USER"
    REVIEW_USER = "REVIEW_USER"
    RESTRICT_DEVICE = "RESTRICT_DEVICE"
    REVIEW_DEVICE = "REVIEW_DEVICE"
    RESTRICT_PAYMENT_INSTRUMENT = "RESTRICT_PAYMENT_INSTRUMENT"
    REVIEW_PAYMENT_INSTRUMENT = "REVIEW_PAYMENT_INSTRUMENT"
    MONITOR_CAMPAIGN = "MONITOR_CAMPAIGN"
    NO_ACTION = "NO_ACTION"


class TargetType(str, Enum):
    USER = "USER"
    DEVICE = "DEVICE"
    IP = "IP"
    PAYMENT_INSTRUMENT = "PAYMENT_INSTRUMENT"
    TRANSACTION = "TRANSACTION"
    CAMPAIGN = "CAMPAIGN"


# Which target node types each action type may address.
ALLOWED_TARGETS: dict[ActionType, set[TargetType]] = {
    ActionType.BLOCK_TRANSACTION: {TargetType.TRANSACTION},
    ActionType.REVIEW_TRANSACTION: {TargetType.TRANSACTION},
    ActionType.BLOCK_USER: {TargetType.USER},
    ActionType.REVIEW_USER: {TargetType.USER},
    ActionType.RESTRICT_DEVICE: {TargetType.DEVICE},
    ActionType.REVIEW_DEVICE: {TargetType.DEVICE},
    ActionType.RESTRICT_PAYMENT_INSTRUMENT: {TargetType.PAYMENT_INSTRUMENT},
    ActionType.REVIEW_PAYMENT_INSTRUMENT: {TargetType.PAYMENT_INSTRUMENT},
    ActionType.MONITOR_CAMPAIGN: {TargetType.CAMPAIGN},
    ActionType.NO_ACTION: {TargetType.CAMPAIGN},
}


# Semantic effect on transactions related to the target.
# "block" -> affected transactions are considered contained.
# "review" -> affected transactions enter a review queue (not blocked).
# "monitor" / "no_action" -> observation only.
class EffectKind(str, Enum):
    BLOCK = "block"
    REVIEW = "review"
    MONITOR = "monitor"
    NONE = "none"


ACTION_EFFECT: dict[ActionType, EffectKind] = {
    ActionType.BLOCK_TRANSACTION: EffectKind.BLOCK,
    ActionType.REVIEW_TRANSACTION: EffectKind.REVIEW,
    ActionType.BLOCK_USER: EffectKind.BLOCK,
    ActionType.REVIEW_USER: EffectKind.REVIEW,
    ActionType.RESTRICT_DEVICE: EffectKind.BLOCK,
    ActionType.REVIEW_DEVICE: EffectKind.REVIEW,
    ActionType.RESTRICT_PAYMENT_INSTRUMENT: EffectKind.BLOCK,
    ActionType.REVIEW_PAYMENT_INSTRUMENT: EffectKind.REVIEW,
    ActionType.MONITOR_CAMPAIGN: EffectKind.MONITOR,
    ActionType.NO_ACTION: EffectKind.NONE,
}


@dataclass(frozen=True)
class CostConfig:
    """Demo action costs (NOT Razorpay economics). Clearly labeled."""

    transaction_block_cost: float = 20.0
    transaction_review_cost: float = 5.0
    user_block_cost: float = 25.0
    user_review_cost: float = 8.0
    device_restrict_cost: float = 30.0
    device_review_cost: float = 10.0
    payment_restrict_cost: float = 30.0
    payment_review_cost: float = 10.0
    monitor_cost: float = 2.0
    no_action_cost: float = 0.0


DEFAULT_COST_CONFIG = CostConfig()


def action_cost_for(action_type: ActionType, costs: CostConfig = DEFAULT_COST_CONFIG) -> float:
    cost = {
        ActionType.BLOCK_TRANSACTION: costs.transaction_block_cost,
        ActionType.REVIEW_TRANSACTION: costs.transaction_review_cost,
        ActionType.BLOCK_USER: costs.user_block_cost,
        ActionType.REVIEW_USER: costs.user_review_cost,
        ActionType.RESTRICT_DEVICE: costs.device_restrict_cost,
        ActionType.REVIEW_DEVICE: costs.device_review_cost,
        ActionType.RESTRICT_PAYMENT_INSTRUMENT: costs.payment_restrict_cost,
        ActionType.REVIEW_PAYMENT_INSTRUMENT: costs.payment_review_cost,
        ActionType.MONITOR_CAMPAIGN: costs.monitor_cost,
        ActionType.NO_ACTION: costs.no_action_cost,
    }.get(action_type, 0.0)
    return float(cost)


@dataclass(frozen=True)
class Action:
    """A single candidate intervention (simulated/test-mode only)."""

    action_id: str
    action_type: ActionType
    target_id: str
    target_type: TargetType
    campaign_id: str
    reason: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["action_type"] = self.action_type.value
        out["target_type"] = self.target_type.value
        return out