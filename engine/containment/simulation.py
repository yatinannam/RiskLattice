"""Containment simulation (Phase 5).

simulate_action evaluates an action against the ENTIRE usable dataset so
legitimate collateral is measured across all historical transactions associated
with the affected entities — not just inside the campaign.

Ground-truth isolation: the simulator's own "fraud vs legitimate" split is
derived from Phase-2 risk scores (risk >= threshold = suspicious), NEVER from
ground-truth labels. Ground truth is consumed only by the dedicated
``evaluate_strategy_ground_truth`` helper.

Simulation output fields:
  fraud_containment_rate     = suspicious tx affected / campaign suspicious tx
  fraud_exposure_contained   = suspicious amount affected (exposure estimate,
                               NOT money recovered)
  legitimate_impact_rate     = legit tx affected / all related legit tx
  collateral_risk            = transparent 0..1 score
"""

from __future__ import annotations

import sys
from pathlib import Path

from engine.containment.actions import Action, TargetType

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

HIGH_RISK_TX_THRESHOLD = 0.5


class ContainmentIndex:
    """Whole-dataset reverse lookup, built once (no repeated full scans)."""

    def __init__(self, transactions, risk_scores: dict[str, float],
                 gt_is_fraud: dict[str, bool] | None = None) -> None:
        self.amounts: dict[str, float] = {}
        self.risk: dict[str, float] = dict(risk_scores)
        self.entity_map: dict[str, dict[str, list[str]]] = {
            "USER": {}, "DEVICE": {}, "IP": {}, "PAYMENT_INSTRUMENT": {},
            "TRANSACTION": {},
        }
        self.transaction_entities: dict[str, dict[str, str]] = {}
        self.gt_is_fraud: dict[str, bool] = gt_is_fraud or {}
        self._build(transactions)

    def _build(self, transactions) -> None:
        for row in transactions.itertuples(index=False):
            tx_id = row.transaction_id
            self.amounts[tx_id] = float(row.amount)
            tx_ent = {
                "USER": row.user_id,
                "DEVICE": row.device_id,
                "IP": row.ip_id,
                "PAYMENT_INSTRUMENT": row.payment_instrument_id,
            }
            self.transaction_entities[tx_id] = tx_ent
            for kind, eid in tx_ent.items():
                self.entity_map[kind].setdefault(eid, []).append(tx_id)
            self.entity_map["TRANSACTION"].setdefault(tx_id, [tx_id])
        for kind in self.entity_map:
            for eid in self.entity_map[kind]:
                self.entity_map[kind][eid].sort()

    def lookup(self, target_type: str, target_id: str) -> list[str]:
        return list(self.entity_map.get(target_type, {}).get(target_id, []))

    def is_suspicious(self, tx_id: str) -> bool:
        """Optimiser-view label derived from risk score (never ground truth)."""
        return self.risk.get(tx_id, 0.0) >= HIGH_RISK_TX_THRESHOLD

    def is_fraud_gt(self, tx_id: str) -> bool:
        """Ground-truth label; only used by evaluation helpers."""
        return bool(self.gt_is_fraud.get(tx_id, False))


class SimResult:
    """Result of simulating an action across the whole dataset."""

    __slots__ = (
        "action_id", "fraud_transactions_affected", "fraud_amount_affected",
        "legitimate_transactions_affected", "legitimate_amount_affected",
        "users_affected", "devices_affected", "ips_affected",
        "payment_instruments_affected", "fraud_containment_rate",
        "legitimate_impact_rate", "collateral_risk", "collateral_level",
        "fraud_exposure_contained", "affected_transactions",
    )

    def __init__(self, action_id: str, **kwargs) -> None:
        self.action_id = action_id
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        return {slot: getattr(self, slot) for slot in self.__slots__}


def _affected_for_action(action: Action, index: ContainmentIndex) -> set[str]:
    """Transactions historically associated with the action target (whole set)."""
    return set(index.lookup(action.target_type.value, action.target_id))


def _empty_result(action_id: str) -> SimResult:
    return SimResult(
        action_id=action_id,
        fraud_transactions_affected=0, fraud_amount_affected=0.0,
        legitimate_transactions_affected=0, legitimate_amount_affected=0.0,
        users_affected=0, devices_affected=0, ips_affected=0,
        payment_instruments_affected=0, fraud_containment_rate=0.0,
        legitimate_impact_rate=0.0, collateral_risk=0.0, collateral_level="LOW",
        fraud_exposure_contained=0.0, affected_transactions=[],
    )


def simulate_action(
    action: Action,
    transactions,
    risk_scores: dict[str, float],
    graph,
    campaign_transaction_ids: set[str],
    index: ContainmentIndex,
) -> SimResult:
    """Simulate one action and return whole-dataset impact estimates.

    The fraud/legit split uses ``index.is_suspicious`` (risk-derived), so the
    optimizer never reads ground-truth labels.
    """
    affected = _affected_for_action(action, index)
    if not affected:
        return _empty_result(action.action_id)

    amounts = index.amounts
    fraud_affected = {t for t in affected if index.is_suspicious(t)}
    legit_affected = affected - fraud_affected

    fraud_amount = sum(amounts.get(t, 0.0) for t in fraud_affected)
    legit_amount = sum(amounts.get(t, 0.0) for t in legit_affected)

    campaign_susp = {t for t in campaign_transaction_ids if index.is_suspicious(t)}
    containment_rate = len(fraud_affected & campaign_transaction_ids) / max(len(campaign_susp), 1)

    legit_rate = len(legit_affected) / max(len(affected), 1)

    users = {index.transaction_entities[t].get("USER") for t in affected if t in index.transaction_entities}
    devices = {index.transaction_entities[t].get("DEVICE") for t in affected if t in index.transaction_entities}
    ips = {index.transaction_entities[t].get("IP") for t in affected if t in index.transaction_entities}
    pis = {index.transaction_entities[t].get("PAYMENT_INSTRUMENT") for t in affected if t in index.transaction_entities}
    users.discard(None); devices.discard(None); ips.discard(None); pis.discard(None)

    collateral = collateral_risk_score(
        legit_count=len(legit_affected),
        legit_user_count=len(users),
        fraud_count=len(fraud_affected),
        unrelated_entity_count=len(devices | ips | pis),
    )

    return SimResult(
        action_id=action.action_id,
        fraud_transactions_affected=len(fraud_affected),
        fraud_amount_affected=round(fraud_amount, 2),
        legitimate_transactions_affected=len(legit_affected),
        legitimate_amount_affected=round(legit_amount, 2),
        users_affected=len(users),
        devices_affected=len(devices),
        ips_affected=len(ips),
        payment_instruments_affected=len(pis),
        fraud_containment_rate=round(containment_rate, 4),
        legitimate_impact_rate=round(legit_rate, 4),
        collateral_risk=round(collateral, 4),
        collateral_level=_collateral_level(collateral),
        fraud_exposure_contained=round(fraud_amount, 2),
        affected_transactions=sorted(affected),
    )


def collateral_risk_score(
    legit_count: int,
    legit_user_count: int,
    fraud_count: int,
    unrelated_entity_count: int = 0,
) -> float:
    """Transparent 0..1 collateral-risk score (documented heuristic).

    = 0.40*legit_tx_norm + 0.25*legit_user_norm + 0.20*unrelated_ent_norm
      + 0.15*legit_proportion
    where
      legit_tx_norm       = min(1, legit_count / 10)
      legit_user_norm     = min(1, legit_user_count / 5)
      unrelated_ent_norm  = min(1, unrelated_entity_count / 5)
      legit_proportion    = legit_count / max(fraud_count + legit_count, 1)
    """
    tx_norm = min(1.0, legit_count / 10.0)
    user_norm = min(1.0, legit_user_count / 5.0)
    entity_norm = min(1.0, unrelated_entity_count / 5.0)
    proportion = legit_count / max((fraud_count + legit_count), 1)
    score = 0.40 * tx_norm + 0.25 * user_norm + 0.20 * entity_norm + 0.15 * proportion
    return max(0.0, min(1.0, score))


def _collateral_level(score: float) -> str:
    if score >= 0.6:
        return "HIGH"
    if score >= 0.3:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Ground-truth evaluation (dedicated; NEVER used to choose actions)
# ---------------------------------------------------------------------------

def evaluate_strategy_ground_truth(
    affected_tx_ids: set[str],
    campaign_transaction_ids: set[str],
    index: ContainmentIndex,
) -> dict:
    """Recompute actual fraud/legit impact using ground-truth labels.

    Returns actual (not estimated) counts/amounts for evaluation only.
    """
    if not index.gt_is_fraud:
        return {"error": "no_ground_truth_provided"}

    fraud_actual = {t for t in affected_tx_ids if index.is_fraud_gt(t)}
    legit_actual = {t for t in affected_tx_ids if not index.is_fraud_gt(t)}
    amounts = index.amounts

    campaign_fraud_gt = {t for t in campaign_transaction_ids if index.is_fraud_gt(t)}
    real_contain = len(fraud_actual & campaign_transaction_ids) / max(len(campaign_fraud_gt), 1)

    return {
        "actual_fraud_transactions_affected": len(fraud_actual),
        "actual_fraud_amount_affected": round(sum(amounts.get(t, 0.0) for t in fraud_actual), 2),
        "actual_legitimate_transactions_affected": len(legit_actual),
        "actual_legitimate_amount_affected": round(sum(amounts.get(t, 0.0) for t in legit_actual), 2),
        "actual_fraud_containment_rate": round(real_contain, 4),
    }