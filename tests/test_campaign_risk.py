"""Tests for campaign risk intelligence (Phase 4).

Covers determinism, bounded scores, risk-level thresholds, confidence vs risk,
single-signal != critical, multi-signal escalation, temporal density,
exposure = actual amount sum, ground-truth exclusion, deterministic ranking and
deduplication, evidence integrity, and false-negative diagnosis.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from engine.graph.graph_builder import build_graph
from engine.risk.rules import risk_level_for_score
from engine.risk.scoring import temporal_risk_score
from engine.risk.risk_engine import (
    TxIndex,
    analyze_false_negatives,
    assess_campaign,
    deduplicate_candidates,
    rank_campaigns,
    evaluate_against_ground_truth,
)


def _row(txid, ts, user, device, ip, pi, amount=100.0, status="success"):
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
        "is_fraud": False,
        "fraud_campaign_id": None,
        "scenario": "legitimate",
    }


def _bulk(n, prefix, shared_ip=None, shared_dev=None, shared_pi=None,
          step_min=1, amount=100.0, status="success", low_risk=0.05):
    """Build n transactions, optionally sharing an entity. Returns rows+risk."""
    base = datetime(2026, 8, 1, 10, 0)
    rows = []
    risk = {}
    for i in range(n):
        ip = shared_ip if shared_ip else f"{prefix}IP{i}"
        dev = shared_dev if shared_dev else f"{prefix}DEV{i}"
        pi = shared_pi if shared_pi else f"{prefix}PI{i}"
        tx_id = f"{prefix}T{i}"
        rows.append(_row(
            tx_id,
            base + timedelta(minutes=i * step_min),
            f"{prefix}U{i}",
            dev, ip, pi,
            amount=amount + i,
            status=status,
        ))
        risk[tx_id] = low_risk
    return rows, risk


def _candidate_from_frame(df, cid="cand-test", density=0.0):
    return {
        "campaign_candidate_id": cid,
        "transaction_ids": list(df["transaction_id"]),
        "user_ids": sorted(df["user_id"].unique()),
        "device_ids": sorted(df["device_id"].unique()),
        "ip_ids": sorted(df["ip_id"].unique()),
        "payment_instrument_ids": sorted(df["payment_instrument_id"].unique()),
        "start_time": df["timestamp"].min().isoformat(),
        "end_time": df["timestamp"].max().isoformat(),
        "duration_seconds": int((df["timestamp"].max() - df["timestamp"].min()).total_seconds()),
        "relationship_density": density,
        "risk_score_summary": {"mean": 0.5, "max": 0.5},
        "evidence": {},
    }


def _assess(rows, risk, cid="cand-test", density=0.0):
    df = pd.DataFrame(rows)
    graph = build_graph(df)
    cand = _candidate_from_frame(df, cid, density)
    return assess_campaign(cand, graph, TxIndex(df), risk)


# ---------------------------------------------------------------------------
# Determinism / bounds
# ---------------------------------------------------------------------------

def test_risk_score_deterministic():
    rows, risk = _bulk(5, "A")
    a1 = _assess(rows, risk)
    a2 = _assess(rows, risk)
    assert a1.risk_score == a2.risk_score
    assert a1.to_dict() == a2.to_dict()


def test_risk_score_bounded_0_100():
    rows, risk = _bulk(8, "B", shared_ip="IPX", shared_dev="DEVX", shared_pi="PIX",
                       low_risk=0.9)
    a = _assess(rows, risk)
    assert 0.0 <= a.risk_score <= 100.0


def test_component_scores_bounded_0_1():
    rows, risk = _bulk(8, "C", low_risk=0.9)
    a = _assess(rows, risk)
    for name in ("transaction_risk", "relationship_risk", "temporal_risk",
                 "concentration_risk", "behavioral_risk"):
        assert 0.0 <= getattr(a, name) <= 1.0, name


def test_risk_level_follows_documented_thresholds():
    assert risk_level_for_score(10) == "LOW"
    assert risk_level_for_score(29.9) == "LOW"
    assert risk_level_for_score(30) == "MEDIUM"
    assert risk_level_for_score(59.9) == "MEDIUM"
    assert risk_level_for_score(60) == "HIGH"
    assert risk_level_for_score(79.9) == "HIGH"
    assert risk_level_for_score(80) == "CRITICAL"
    assert risk_level_for_score(100) == "CRITICAL"


def test_confidence_is_separate_from_risk():
    rows, risk = _bulk(8, "D")
    a = _assess(rows, risk)
    assert 0.0 <= a.confidence <= 1.0
    assert a.confidence != a.risk_score / 100.0


# ---------------------------------------------------------------------------
# Single signal != critical; multi-signal escalation
# ---------------------------------------------------------------------------

def test_shared_ip_alone_cannot_cause_critical_risk():
    rows, risk = _bulk(10, "E", shared_ip="IP_SHARED", low_risk=0.05, step_min=120)
    a = _assess(rows, risk)
    assert a.risk_score < 80.0
    assert a.risk_level != "CRITICAL"


def test_shared_device_alone_cannot_cause_critical_risk():
    rows, risk = _bulk(10, "F", shared_dev="DEV_SHARED", low_risk=0.05, step_min=120)
    a = _assess(rows, risk)
    assert a.risk_score < 80.0
    assert a.risk_level != "CRITICAL"


def test_multiple_independent_signals_increase_risk():
    rows_a, risk_a = _bulk(8, "G", shared_ip="IPX", low_risk=0.05, step_min=120)
    rows_b, risk_b = _bulk(8, "H", shared_ip="IPX", shared_dev="DEVX",
                           shared_pi="PIX", low_risk=0.5, step_min=1)
    a = _assess(rows_a, risk_a)
    b = _assess(rows_b, risk_b)
    assert b.risk_score > a.risk_score


def test_temporal_density_affects_temporal_risk():
    short = temporal_risk_score(20, 4 * 60)          # 20 tx in 4 min
    long = temporal_risk_score(20, 30 * 24 * 3600)   # 20 tx in 30 days
    assert short > long


def test_exposure_equals_actual_transaction_sum():
    df = pd.DataFrame([
        _row("t1", datetime(2026, 8, 1, 10, 0), "U0", "D0", "I0", "P0", amount=100.0),
        _row("t2", datetime(2026, 8, 1, 10, 1), "U0", "D0", "I1", "P1", amount=250.5),
        _row("t3", datetime(2026, 8, 1, 10, 2), "U1", "D1", "I0", "P2", amount=49.5),
    ])
    graph = build_graph(df)
    cand = _candidate_from_frame(df)
    a = assess_campaign(cand, graph, TxIndex(df), {})
    assert a.estimated_exposure == 400.0


def test_ground_truth_fields_never_change_scoring():
    rows, risk = _bulk(6, "J", shared_ip="IPX", low_risk=0.6)
    df = pd.DataFrame(rows)
    df["is_fraud"] = 1
    df["fraud_campaign_id"] = "GT-1"
    df["scenario"] = "account_farm"
    graph = build_graph(df)
    ti = TxIndex(df)
    cand = _candidate_from_frame(df, "cand-j")
    base = assess_campaign(cand, graph, ti, risk)
    leaked = dict(cand)
    leaked["is_fraud"] = True
    leaked["fraud_campaign_id"] = "GT-1"
    leaked["scenario"] = "account_farm"
    out = assess_campaign(leaked, graph, ti, risk)
    assert base.to_dict() == out.to_dict()


def test_ranking_is_deterministic():
    rows, risk = _bulk(5, "K", low_risk=0.5, shared_ip="IPX")
    a = _assess(rows, risk, cid="K1")
    b = _assess(rows, risk, cid="K2")
    a2 = _assess(rows, risk, cid="K1")
    ranked1 = [x.campaign_id for x in rank_campaigns([a, b])]
    ranked2 = [x.campaign_id for x in rank_campaigns([a2, b])]
    assert ranked1 == ranked2


def test_ranking_filters():
    rows, risk = _bulk(5, "L", low_risk=0.5, amount=1000.0)
    a = _assess(rows, risk, cid="L1")
    high_only = rank_campaigns([a], min_risk_level="HIGH")
    assert all(x.risk_level in ("HIGH", "CRITICAL") for x in high_only)


def test_ranking_is_deterministic_full_rank():
    rows, risk = _bulk(6, "M", low_risk=0.9, shared_ip="IPX")
    a1 = _assess(rows, risk, cid="M1")
    a2 = _assess(rows, risk, cid="M2")
    seq1 = [c.campaign_id for c in rank_campaigns([a1, a2])]
    seq2 = [c.campaign_id for c in rank_campaigns([a1, a2])]
    assert seq1 == seq2


def test_deduplication_is_deterministic():
    base = [
        {"campaign_candidate_id": "c1", "transaction_ids": ["1", "2", "3"],
         "risk_score_summary": {"max": 0.9}},
        {"campaign_candidate_id": "c2", "transaction_ids": ["1", "2"],
         "risk_score_summary": {"max": 0.5}},
    ]
    out1 = deduplicate_candidates(base)
    out2 = deduplicate_candidates(list(base))
    assert [c["campaign_candidate_id"] for c in out1] == \
           [c["campaign_candidate_id"] for c in out2]


def test_evidence_references_real_entities_and_transactions():
    rows, risk = _bulk(8, "E", shared_ip="IPX", shared_dev="DEVX", low_risk=0.4)
    a = _assess(rows, risk)
    df = pd.DataFrame(rows)
    allowed_tx = set(df["transaction_id"])
    allowed_entities = (set(df["user_id"]) | set(df["device_id"])
                        | set(df["ip_id"]) | set(df["payment_instrument_id"]))
    for ev in a.evidence:
        for entity in ev.entities:
            assert entity in allowed_entities
        for tx in ev.supporting_transactions:
            assert tx in allowed_tx


def test_false_negatives_analysis_does_not_change_scoring():
    rows, risk = _bulk(8, "P", shared_ip="IPX", low_risk=0.3)
    df = pd.DataFrame(rows)
    df["is_fraud"] = df["is_fraud"].astype(int)
    # Mark one transaction as ground-truth fraud.
    df.loc[0, "is_fraud"] = 1
    df.loc[0, "fraud_campaign_id"] = "GT-CAMP"
    df.loc[0, "scenario"] = "account_farm"
    for i in range(1, len(df)):
        df.loc[i, "is_fraud"] = 0

    graph = build_graph(df)
    ti = TxIndex(df)
    cand = _candidate_from_frame(df, "cand-fn")
    before = assess_campaign(cand, graph, ti, risk).to_dict()

    preds = {t: 0 for t in df["transaction_id"]}  # model misses everything as 0
    fn_rows = analyze_false_negatives([assess_campaign(cand, graph, ti, risk)],
                                      df, preds, risk_scores=risk,
                                      min_risk_level="LOW")
    assert fn_rows, "expected at least one false negative"
    assert any(r["false_negative_transaction_id"] == df.loc[0, "transaction_id"]
               for r in fn_rows)

    after = assess_campaign(cand, graph, ti, risk).to_dict()
    assert before == after  # diagnosis must not alter scoring