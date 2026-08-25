"""Containment optimizer (Phase 5).

Bounded search heuristic (documented — NOT claimed optimal):

  1. Generate candidate actions from the campaign's own entities (users,
     devices, payment instruments) and its high-risk transactions; never from
     unrelated entities.
  2. Evaluate each individual action via whole-dataset simulation.
  3. Evaluate small combinations (max size 3) over the top-K highest-leverage
     entities to bound the search.
  4. Apply configurable constraints (max legit users, min fraud containment,
     max action count). If nothing satisfies them -> NO_SAFE_ACTION.
  5. Remove dominated strategies.
  6. Recommend the best feasible strategy; produce an audit record.

The optimizer's own estimates use risk-derived labels only; ground-truth is
used exclusively by the dedicated evaluation helpers.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from engine.containment.actions import (
    Action,
    ActionType,
    CostConfig,
    DEFAULT_COST_CONFIG,
    TargetType,
    action_cost_for,
)
from engine.containment.simulation import (
    ContainmentIndex,
    SimResult,
    collateral_risk_score,
    simulate_action,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Configurable constraints (documented, not hidden magic numbers)
# ---------------------------------------------------------------------------
MAX_LEGITIMATE_USERS_AFFECTED = 5
MIN_FRAUD_CONTAINMENT = 0.70
MAX_ACTIONS = 3
TOP_K_ENTITIES = 10  # bounding limit for combination search


# Containment-score heuristic weights (documented, not optimized).
W_LEGIT_PENALTY = 0.35
W_COST_PENALTY = 0.15
W_COLLATERAL_PENALTY = 0.20


@dataclass
class Strategy:
    """One candidate intervention strategy (a set of actions)."""

    action_ids: list[str] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    affected_transactions: set[str] = field(default_factory=set)
    fraud_containment_rate: float = 0.0
    fraud_exposure_contained: float = 0.0
    legitimate_users_affected: int = 0
    legitimate_transactions_affected: int = 0
    legitimate_amount_affected: float = 0.0
    collateral_risk: float = 0.0
    collateral_level: str = "LOW"
    action_count: int = 0
    total_cost: float = 0.0
    containment_score: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "action_ids": self.action_ids,
            "action_types": [a.action_type.value for a in self.actions],
            "fraud_containment_rate": round(self.fraud_containment_rate, 4),
            "fraud_exposure_contained": round(self.fraud_exposure_contained, 2),
            "legitimate_users_affected": self.legitimate_users_affected,
            "legitimate_transactions_affected": self.legitimate_transactions_affected,
            "legitimate_amount_affected": round(self.legitimate_amount_affected, 2),
            "collateral_risk": round(self.collateral_risk, 4),
            "collateral_level": self.collateral_level,
            "action_count": self.action_count,
            "total_cost": round(self.total_cost, 2),
            "containment_score": round(self.containment_score, 4),
        }


class ContainmentOptimizer:
    """Evaluate candidate strategies and recommend one (bounded heuristic)."""

    def __init__(
        self,
        transactions,
        index: ContainmentIndex,
        risk_scores: dict[str, float],
        max_legit_users: int = MAX_LEGITIMATE_USERS_AFFECTED,
        min_fraud_containment: float = MIN_FRAUD_CONTAINMENT,
        max_actions: int = MAX_ACTIONS,
        top_k: int = TOP_K_ENTITIES,
        costs: CostConfig = DEFAULT_COST_CONFIG,
    ) -> None:
        self.transactions = transactions
        self.index = index
        self.risk_scores = risk_scores
        self.max_legit_users = max_legit_users
        self.min_fraud_containment = min_fraud_containment
        self.max_actions = max_actions
        self.top_k = top_k
        self.costs = costs

    # ------------------------------------------------------------------
    # Action generation (campaign entities only)
    # ------------------------------------------------------------------

    def _generate_actions(self, assessment) -> list[Action]:
        """Candidate actions from the campaign's own entities (bounded)."""
        campaign_id = assessment.campaign_id
        actions: list[Action] = []
        seq = 0

        def _uid() -> str:
            nonlocal seq
            seq += 1
            return f"ACT_{campaign_id}_{seq:03d}"

        # User-level actions.
        for uid in assessment.user_ids:
            actions.append(Action(
                action_id=_uid(), action_type=ActionType.BLOCK_USER,
                target_id=uid, target_type=TargetType.USER,
                campaign_id=campaign_id,
                reason="User is a member of the assessed campaign",
                evidence=["campaign membership"],
            ))

        # Device-level actions.
        for dev in assessment.device_ids:
            actions.append(Action(
                action_id=_uid(), action_type=ActionType.RESTRICT_DEVICE,
                target_id=dev, target_type=TargetType.DEVICE,
                campaign_id=campaign_id,
                reason="Device is part of the campaign's shared infrastructure",
                evidence=["campaign membership"],
            ))

        # Payment-instrument actions.
        for pi in assessment.payment_instrument_ids:
            actions.append(Action(
                action_id=_uid(), action_type=ActionType.RESTRICT_PAYMENT_INSTRUMENT,
                target_id=pi, target_type=TargetType.PAYMENT_INSTRUMENT,
                campaign_id=campaign_id,
                reason="Payment instrument is part of the campaign",
                evidence=["campaign membership"],
            ))

        # Transaction-level actions (top high-risk transactions only, bounded).
        high_risk_tx = [t for t in assessment.transaction_ids
                        if self.index.is_suspicious(t)]
        for tx_id in high_risk_tx[:self.top_k]:
            actions.append(Action(
                action_id=_uid(), action_type=ActionType.BLOCK_TRANSACTION,
                target_id=tx_id, target_type=TargetType.TRANSACTION,
                campaign_id=campaign_id,
                reason="High-risk transaction in the campaign",
                evidence=["elevated transaction risk"],
            ))

        return actions

    # ------------------------------------------------------------------
    # Strategy evaluation
    # ------------------------------------------------------------------

    def _simulate_action(self, action: Action, campaign_tx: set[str]) -> SimResult:
        return simulate_action(
            action, self.transactions, self.risk_scores, None, campaign_tx, self.index
        )

    def _evaluate_strategy(self, actions: list[Action], campaign_tx: set[str]) -> Strategy:
        affected: set[str] = set()
        total_cost = 0.0
        for action in actions:
            res = self._simulate_action(action, campaign_tx)
            affected |= set(res.affected_transactions)
            total_cost += action_cost_for(action.action_type, self.costs)

        campaign_susp = {t for t in campaign_tx if self.index.is_suspicious(t)}
        fraud_hit = {t for t in affected if self.index.is_suspicious(t)}
        containment = len(fraud_hit & campaign_tx) / max(len(campaign_susp), 1)

        legit_tx = affected - fraud_hit
        amounts = self.index.amounts
        legit_amount = sum(amounts.get(t, 0.0) for t in legit_tx)

        legit_users = set()
        for t in legit_tx:
            u = self.index.transaction_entities.get(t, {}).get("USER")
            if u:
                legit_users.add(u)

        # Collateral is computed on the union for transparency.
        collateral = collateral_risk_score(
            legit_count=len(legit_tx),
            legit_user_count=len(legit_users),
            fraud_count=len(fraud_hit),
        )

        exposure = sum(amounts.get(t, 0.0) for t in fraud_hit)
        score = self._containment_score(containment, legit_tx, legit_users, total_cost, collateral)

        return Strategy(
            action_ids=[a.action_id for a in actions],
            actions=actions,
            affected_transactions=affected,
            fraud_containment_rate=containment,
            fraud_exposure_contained=exposure,
            legitimate_users_affected=len(legit_users),
            legitimate_transactions_affected=len(legit_tx),
            legitimate_amount_affected=round(legit_amount, 2),
            collateral_risk=round(collateral, 4),
            collateral_level="HIGH" if collateral >= 0.6 else ("MEDIUM" if collateral >= 0.3 else "LOW"),
            action_count=len(actions),
            total_cost=round(total_cost, 2),
            containment_score=score,
        )

    def _containment_score(self, containment_rate: float, legit_tx: set[str],
                           legit_users: set[str], cost: float, collateral: float) -> float:
        """Transparent heuristic: containment - impact - cost - collateral."""
        legit_penalty = min(1.0, (len(legit_users) / 5.0) * 0.5 + (len(legit_tx) / 30.0))
        cost_penalty = min(1.0, cost / 100.0)
        score = (
            containment_rate
            - W_LEGIT_PENALTY * legit_penalty
            - W_COST_PENALTY * cost_penalty
            - W_COLLATERAL_PENALTY * collateral
        )
        return max(-1.0, min(1.0, score))

    def _is_feasible(self, strategy: Strategy) -> bool:
        return (
            strategy.fraud_containment_rate >= self.min_fraud_containment
            and strategy.legitimate_users_affected <= self.max_legit_users
            and 1 <= strategy.action_count <= self.max_actions
        )

    def _dominates(self, a: Strategy, b: Strategy) -> bool:
        """a dominates b if it is >= on containment and <= on every impact/cost
        dimension, with at least one strictly better."""
        better = (
            a.fraud_containment_rate >= b.fraud_containment_rate
            and a.legitimate_users_affected <= b.legitimate_users_affected
            and a.legitimate_transactions_affected <= b.legitimate_transactions_affected
            and a.collateral_risk <= b.collateral_risk
            and a.action_count <= b.action_count
        )
        strictly = (
            a.fraud_containment_rate > b.fraud_containment_rate
            or a.legitimate_users_affected < b.legitimate_users_affected
            or a.legitimate_transactions_affected < b.legitimate_transactions_affected
            or a.collateral_risk < b.collateral_risk
            or a.action_count < b.action_count
        )
        return better and strictly

    def _prune_dominated(self, strategies: list[Strategy]) -> list[Strategy]:
        kept: list[Strategy] = []
        for s in strategies:
            if any(self._dominates(o, s) for o in strategies if o is not s):
                continue
            kept.append(s)
        return kept

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(self, assessment) -> dict:
        """Return a recommendation dict with an audit trail.

        The returned dict is SIMULATED / TEST-MODE only; it never executes
        any payment action.
        """
        campaign_tx = set(assessment.transaction_ids)
        actions = self._generate_actions(assessment)

        # Bounded combination search over top-K highest-leverage entities.
        pool = self._top_k_actions(actions, campaign_tx)

        strategies: list[Strategy] = []
        singles = [self._evaluate_strategy([a], campaign_tx) for a in pool]
        strategies.extend(singles)

        # Pair and triple combinations from the pool (bounded).
        for n_comb in (2, 3):
            for combo in combinations(pool, n_comb):
                if len(combo) > self.max_actions:
                    continue
                strategies.append(self._evaluate_strategy(list(combo), campaign_tx))

        # Deduplicate identical action sets.
        seen: set[tuple] = set()
        unique: list[Strategy] = []
        for s in strategies:
            key = tuple(s.action_ids)
            if key in seen:
                continue
            seen.add(key)
            unique.append(s)

        feasible = [s for s in unique if self._is_feasible(s)]
        if not feasible:
            return {
                "campaign_id": assessment.campaign_id,
                "recommended_strategy": None,
                "recommendation": "NO_SAFE_ACTION",
                "reason": (
                    f"No strategy reached {self.min_fraud_containment:.0%} fraud "
                    f"containment while affecting at most "
                    f"{self.max_legit_users} legitimate users within max "
                    f"{self.max_actions} actions on this dataset."
                ),
                "constraints_satisfied": False,
                "audit_record": self._audit_record(assessment, "NO_SAFE_ACTION"),
            }

        # Best by containment score (documented heuristic), ties -> fewer actions.
        best = max(feasible, key=lambda s: (s.containment_score, -s.action_count))
        pruned = self._prune_dominated(unique)
        alternatives = [s for s in pruned if s.action_ids != best.action_ids]
        alternatives.sort(key=lambda s: -s.containment_score)

        return {
            "campaign_id": assessment.campaign_id,
            "recommended_strategy": best.describe(),
            "recommendation": "CONTAIN",
            "alternative_strategies": [s.describe() for s in alternatives[:4]],
            "expected_fraud_containment": best.fraud_containment_rate,
            "expected_fraud_exposure_contained": best.fraud_exposure_contained,
            "expected_legitimate_users_affected": best.legitimate_users_affected,
            "collateral_risk": best.collateral_risk,
            "collateral_level": best.collateral_level,
            "confidence": assessment.confidence,
            "constraints_satisfied": True,
            "reason": "Best feasible strategy under the bounded containment heuristic.",
            "audit_record": self._audit_record(assessment, best.describe()),
        }

    def _top_k_actions(self, actions: list[Action], campaign_tx: set[str]) -> list[Action]:
        """Rank actions by affected suspicious tx volume and bound the pool."""
        ranked = []
        for action in actions:
            res = self._simulate_action(action, campaign_tx)
            ranked.append((res.fraud_transactions_affected, res.fraud_exposure_contained, action))
        ranked.sort(key=lambda t: (-t[0], -t[1], t[2].action_id))
        return [a for _, _, a in ranked[: self.top_k]]

    def _audit_record(self, assessment, selected) -> dict:
        return {
            "decision_id": f"DEC_{uuid.uuid4().hex[:10]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "campaign_id": assessment.campaign_id,
            "constraints": {
                "max_legitimate_users": self.max_legit_users,
                "min_fraud_containment": self.min_fraud_containment,
                "max_actions": self.max_actions,
            },
            "candidate_action_count": len(self._generate_actions(assessment)),
            "selected_action": selected,
            "risk_score": assessment.risk_score,
            "evidence_types": [e.type for e in assessment.evidence],
            "reason": "SIMULATED / TEST-MODE containment recommendation",
            "approval_required": True,
            "execution_status": "SIMULATED",
        }