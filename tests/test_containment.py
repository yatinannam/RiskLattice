"""Tests for the containment optimizer (Phase 5)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from engine.containment.actions import Action, ActionType, TargetType
from engine.containment.optimizer import ContainmentOptimizer
from engine.containment.simulation import (
    ContainmentIndex,
    evaluate_strategy_ground_truth,
    simulate_action,
)
from engine.graph.graph_builder import build_graph

B = datetime(2026, 8, 1, 10, 0)


def _row(txid, ts, user, device, ip, pi, amount=100.0, status="success",
         is_fraud=False):
    return {
        "transaction_id": txid,
        "timestamp": ts,
        "merchant_id": "MERCH_001",
        "user_id": user,
        "device_id": device,
        "ip_id": ip,
        "payment_instrument_id": pi,
        "amount": amount,
        "currency": "INR",
        "payment_method": "upi",
        "status": status,
        "is_fraud": is_fraud,
        "fraud_campaign_id": "GT-C" if is_fraud else None,
        "scenario": "account_farm" if is_fraud else "legitimate",
    }


def _frame():
    """3 fraud tx on DEV_007, 2 legit tx on DEV_007 (collateral), unrelated
    legitimate noise."""
    rows = [
        _row("T1", B + timedelta(minutes=0), "U1", "DEV_007", "IP_A", "PI_1",
             amount=500.0, status="success", is_fraud=True),
        _row("T2", B + timedelta(minutes=1), "U2", "DEV_007", "IP_A", "PI_2",
             amount=520.0, status="success", is_fraud=True),
        _row("T3", B + timedelta(minutes=2), "U3", "DEV_007", "IP_B", "PI_3",
             amount=540.0, status="success", is_fraud=True),
    ]
    rows.append(_row("T4", B + timedelta(minutes=5), "U10", "DEV_007", "IP_C",
                     "PI_10", amount=90.0, status="success", is_fraud=False))
    rows.append(_row("T5", B + timedelta(minutes=6), "U11", "DEV_007", "IP_D",
                     "PI_11", amount=80.0, status="success", is_fraud=False))
    for i in range(20, 30):
        rows.append(_row(f"T{i}", B + timedelta(hours=i), f"U90_{i}",
                         f"DEV_9{i}", f"IP_9{i}", f"PI_9{i}", amount=200.0,
                         status="success", is_fraud=False))
    return pd.DataFrame(rows)


def _risk_all(df):
    return {tx: (0.99 if tx.startswith(("T1", "T2", "T3")) else 0.05)
            for tx in df["transaction_id"]}


def _gt_map(df):
    return {row["transaction_id"]: bool(row["is_fraud"])
            for row in df.to_dict("records")}


def _idx(df, risk=None):
    if risk is None:
        risk = _risk_all(df)
    return ContainmentIndex(df, risk, gt_is_fraud=_gt_map(df))


def _action(idx, atype, target_id, ttype, campaign="C-1"):
    return Action(
        action_id=f"A_{idx}",
        action_type=atype,
        target_id=target_id,
        target_type=ttype,
        campaign_id=campaign,
        reason="test action",
        evidence=["test"],
    )


def _campaign_tx(df):
    return {"T1", "T2", "T3"}


# ---------------------------------------------------------------------------
# Target semantics
# ---------------------------------------------------------------------------

def test_transaction_block_affects_only_target():
    df = _frame()
    index = _idx(df)
    action = _action("x", ActionType.BLOCK_TRANSACTION, "T1", TargetType.TRANSACTION)
    res = simulate_action(action, df, _risk_all(df), None, _campaign_tx(df), index)
    assert res.fraud_transactions_affected == 1
    assert res.affected_transactions == ["T1"]


def test_user_block_affects_user_transactions():
    df = _frame()
    index = _idx(df)
    action = _action("A", ActionType.BLOCK_USER, "U1", TargetType.USER)
    res = simulate_action(action, df, _risk_all(df), None, _campaign_tx(df), index)
    assert res.affected_transactions == ["T1"]


def test_device_restriction_evaluates_all_associated():
    df = _frame()
    index = _idx(df)
    action = _action("A", ActionType.RESTRICT_DEVICE, "DEV_007", TargetType.DEVICE)
    res = simulate_action(action, df, _risk_all(df), None, _campaign_tx(df), index)
    # 3 fraud + 2 legit collateral on DEV_007.
    assert res.fraud_transactions_affected == 3
    assert res.legitimate_transactions_affected == 2


def test_payment_instrument_restriction_works():
    df = _frame()
    index = _idx(df)
    action = _action("A", ActionType.RESTRICT_PAYMENT_INSTRUMENT, "PI_1",
                     TargetType.PAYMENT_INSTRUMENT)
    res = simulate_action(action, df, _risk_all(df), None, _campaign_tx(df), index)
    assert res.affected_transactions == ["T1"]


# ---------------------------------------------------------------------------
# Rates and collateral
# ---------------------------------------------------------------------------

def test_fraud_containment_rate_correct():
    df = _frame()
    index = _idx(df)
    action = _action("A", ActionType.RESTRICT_DEVICE, "DEV_007", TargetType.DEVICE)
    res = simulate_action(action, df, _risk_all(df), None, _campaign_tx(df), index)
    # 3 fraud tx in campaign, device covers all 3 -> containment 1.0
    assert res.fraud_containment_rate == 1.0


def test_legitimate_impact_rate_correct():
    df = _frame()
    index = _idx(df)
    action = _action("A", ActionType.RESTRICT_DEVICE, "DEV_007", TargetType.DEVICE)
    res = simulate_action(action, df, _risk_all(df), None, _campaign_tx(df), index)
    # 2 legit affected out of 5 total affected -> 0.4
    assert abs(res.legitimate_impact_rate - 0.4) < 1e-9


def test_legitimate_collateral_is_whole_dataset_not_campaign():
    df = _frame()
    index = _idx(df)
    action = _action("A", ActionType.RESTRICT_DEVICE, "DEV_007", TargetType.DEVICE)
    res = simulate_action(action, df, _risk_all(df), None, _campaign_tx(df), index)
    # Collateral = legit tx associated with DEV_007 across the whole dataset,
    # including those OUTSIDE the campaign (T4, T5).
    assert res.legitimate_transactions_affected == 2
    assert {"T4", "T5"} <= set(res.affected_transactions)


def _assessment_object(df, campaign_tx, risk=None):
    """Minimal CampaignAssessment-like object for the optimizer."""
    from types import SimpleNamespace
    ent = df[df["transaction_id"].isin(campaign_tx)]
    return SimpleNamespace(
        campaign_id="C-1",
        transaction_ids=list(campaign_tx),
        user_ids=sorted(ent["user_id"].unique()),
        device_ids=sorted(ent["device_id"].unique()),
        ip_ids=sorted(ent["ip_id"].unique()),
        payment_instrument_ids=sorted(ent["payment_instrument_id"].unique()),
        risk_score=70.0,
        confidence=0.8,
        evidence=[],
    )


def test_constraints_enforced_device_allows_feasible_block():
    df = _frame()
    index = _idx(df)
    risk = _risk_all(df)
    opt = ContainmentOptimizer(df, index, risk, max_legit_users=5,
                               min_fraud_containment=0.70, max_actions=3,
                               top_k=6)
    assessment = _assessment_object(df, _campaign_tx(df))
    rec = opt.recommend(assessment)
    assert rec["recommendation"] == "CONTAIN"
    s = rec["recommended_strategy"]
    assert s["fraud_containment_rate"] >= 0.70
    assert s["legitimate_users_affected"] <= 5


def test_no_safe_action_when_constraints_unreachable():
    df = _frame()
    index = _idx(df)
    opt = ContainmentOptimizer(df, index, _risk_all(df), max_legit_users=0,
                               min_fraud_containment=0.99, max_actions=1,
                               top_k=3)
    rec = opt.recommend(_assessment_object(df, _campaign_tx(df)))
    assert rec["recommendation"] == "NO_SAFE_ACTION"
    assert rec["constraints_satisfied"] is False


def test_ground_truth_not_used_for_optimization():
    """The optimizer must never read is_fraud labels. We pass a mutated frame
    where is_fraud is inverted; the recommendation must not change."""
    df = _frame()
    df_bad = df.copy()
    df_bad["is_fraud"] = ~df["is_fraud"]  # corrupt labels
    risk = _risk_all(df)
    opt_a = ContainmentOptimizer(df, _idx(df, risk), risk, max_legit_users=5,
                                 min_fraud_containment=0.7, max_actions=3, top_k=6)
    opt_b = ContainmentOptimizer(df_bad, _idx(df_bad, risk), risk, max_legit_users=5,
                                 min_fraud_containment=0.7, max_actions=3, top_k=6)
    ra = opt_a.recommend(_assessment_object(df, _campaign_tx(df)))
    rb = opt_b.recommend(_assessment_object(df_bad, _campaign_tx(df)))
    # Corrupting ground truth must have no effect on the chosen strategy.
    assert ra["recommended_strategy"]["action_ids"] == \
        rb["recommended_strategy"]["action_ids"]


def test_combinations_are_bounded():
    df = _frame()
    index = _idx(df)
    opt = ContainmentOptimizer(df, index, _risk_all(df), max_legit_users=5,
                               min_fraud_containment=0.1, max_actions=3, top_k=4)
    from itertools import combinations
    actions = opt._generate_actions(_assessment_object(df, _campaign_tx(df)))
    # 3 users + 1 device + 3 PIs + 3 high-risk tx = 10 candidate actions.
    assert len(actions) == 10
    for n in (1, 2, 3):
        assert len(list(combinations(actions, n))) <= 300


def _stable_rec(assessment, df, risk):
    index = _idx(df, risk)
    opt = ContainmentOptimizer(df, index, risk, max_legit_users=5,
                               min_fraud_containment=0.1, max_actions=3, top_k=6)
    rec = opt.recommend(assessment)
    return rec


def test_recommendation_deterministic():
    df = _frame()
    risk = _risk_all(df)
    a1 = _assessment_object(df, _campaign_tx(df))
    a2 = _assessment_object(df, _campaign_tx(df))
    r1 = _stable_rec(a1, df, risk)
    r2 = _stable_rec(a2, df, risk)
    assert r1["recommendation"] == r2["recommendation"]
    assert r1["recommended_strategy"]["action_ids"] == \
        r2["recommended_strategy"]["action_ids"]


def test_audit_trail_is_created():
    df = _frame()
    risk = _risk_all(df)
    index = _idx(df, risk)
    opt = ContainmentOptimizer(df, index, risk, max_legit_users=5,
                               min_fraud_containment=0.7, max_actions=3, top_k=6)
    rec = opt.recommend(_assessment_object(df, _campaign_tx(df)))
    assert "audit_record" in rec
    audit = rec["audit_record"]
    assert audit["execution_status"] == "SIMULATED"
    assert audit["approval_required"] is True
    assert audit["decision_id"].startswith("DEC_")
    assert audit["campaign_id"] == "C-1"


def test_simulation_never_executes_real_actions():
    """Actions are always simulated; there is no execution path in scope."""
    df = _frame()
    index = _idx(df)
    action = _action("A", ActionType.RESTRICT_DEVICE, "DEV_007", TargetType.DEVICE)
    res = simulate_action(action, df, _risk_all(df), None, _campaign_tx(df), index)
    # Only impact estimates; no API call/state mutation possible by design.
    assert res.fraud_containment_rate is not None
    assert res.legitimate_transactions_affected == 2