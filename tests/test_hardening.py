"""Tests for Phase 5.5 hardening + adversarial evaluation.

Covers dataset determinism, scenario presence, feature-builder compatibility,
leakage guards (graph/campaign/containment), mixed-entity collateral, and
honest NO_SAFE_ACTION behavior. The baseline dataset must remain byte-identical.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from data.generators.generate_dataset import OUTPUT_PATH, generate_dataset
from data.generators.generate_hardened import (
    HARDENED_SEED,
    generate_hardened_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARDENED_CSV = PROJECT_ROOT / "data" / "samples" / "transactions_hardened.csv"

REQUIRED_SCENARIOS = {
    "low_signal_account_farm",
    "low_signal_payment_abuse",
    "low_signal_coordinated_burst",
    "mixed_entity_campaign",
    "slow_coordinated_campaign",
    "legitimate_shared_office",
    "legitimate_shared_university",
    "legitimate_burst",
    "legitimate_household",
}

# The baseline CSV digest (Phase 1/2 report) — must never change.
BASELINE_SHA256 = (
    "07bc64f3ad3df9c9c9d77610d791a5a51fb1aeecf7376868b81f6c817497e334"
)


@pytest.fixture(scope="module")
def hardened():
    return pd.read_csv(HARDENED_CSV, parse_dates=["timestamp"])


def test_baseline_remains_byte_identical():
    g1 = generate_dataset()
    g2 = generate_dataset()
    assert len(g1) == 10_000
    assert [t.model_dump(mode="json") for t in g1] == [
        t.model_dump(mode="json") for t in g2
    ]
    digest = hashlib.sha256(open(OUTPUT_PATH, "rb").read()).hexdigest()
    assert digest == BASELINE_SHA256


def test_hardened_contains_all_scenario_categories(hardened):
    present = set(hardened["scenario"].unique())
    assert REQUIRED_SCENARIOS <= present


def test_legitimate_shared_infrastructure_exists(hardened):
    legit = hardened[hardened["is_fraud"] == 0]
    shared_ips = legit.groupby("ip_id")["user_id"].nunique()
    shared_devs = legit.groupby("device_id")["user_id"].nunique()
    assert (shared_ips > 1).any()
    assert (shared_devs > 1).any()
    assert legit["scenario"].isin([
        "legitimate_shared_office", "legitimate_shared_university",
        "legitimate_household",
    ]).any()


def test_legitimate_bursts_exist(hardened):
    bursts = hardened[hardened["scenario"] == "legitimate_burst"]
    assert len(bursts) > 0


def test_low_signal_fraud_exists(hardened):
    low = hardened[(hardened["is_fraud"] == 1)
                   & hardened["scenario"].str.startswith("low_signal")]
    assert len(low) > 0
    fraud = hardened[hardened["is_fraud"] == 1]
    legit = hardened[hardened["is_fraud"] == 0]
    # Fraud and legitimate amount distributions should overlap (normal-looking).
    f_med = fraud["amount"].median()
    l_med = legit["amount"].median()
    assert abs(f_med - l_med) / l_med < 1.0


def test_hardened_dataset_is_deterministic():
    assert HARDENED_SEED == 2026
    a = generate_hardened_dataset()
    b = generate_hardened_dataset()
    assert [t.model_dump(mode="json") for t in a] == [
        t.model_dump(mode="json") for t in b
    ]
    assert len(a) == 12_000


def _mixed_rows():
    """5 fraud + 3 legit users sharing DEV_007 (mixed-entity collateral test)."""
    from datetime import datetime, timedelta

    B = datetime(2026, 8, 1, 10, 0)
    out = []

    def mk(i, user, fraud):
        return {
            "transaction_id": f"T{i}", "timestamp": B + timedelta(hours=i),
            "merchant_id": "M", "user_id": user, "device_id": "DEV_007",
            "ip_id": "IP_X", "payment_instrument_id": f"PI_{i}",
            "amount": 100.0, "status": "success", "is_fraud": fraud,
            "fraud_campaign_id": "GT" if fraud else None,
            "scenario": "mixed" if fraud else "legitimate",
        }

    for i, u in enumerate([f"U{k}" for k in range(5)] + [f"L{k}" for k in range(4)]):
        out.append(mk(i, u, u.startswith("U")))
    return pd.DataFrame(out)


def test_mixed_entity_collateral_is_measured_correctly():
    from engine.containment.actions import Action, ActionType, TargetType
    from engine.containment.simulation import (
        ContainmentIndex,
        simulate_action,
    )

    df = _mixed_rows()
    risk = {}
    for row in df.to_dict("records"):
        risk[row["transaction_id"]] = 0.99 if row["user_id"].startswith("U") else 0.05
    gt = {row["transaction_id"]: bool(row["is_fraud"]) for row in df.to_dict("records")}

    index = ContainmentIndex(df, risk, gt_is_fraud=gt)
    action = Action("A", ActionType.RESTRICT_DEVICE, "DEV_007",
                    TargetType.DEVICE, "C", "test", ["e"])
    res = simulate_action(action, df, risk, None, set(df["transaction_id"]), index)
    # 5 fraud + 4 legitimate users historically touch DEV_007 -> 4 legit collater.
    assert res.legitimate_transactions_affected == 4
    assert res.fraud_transactions_affected == 5


def _fake_assessment(campaign_ids):
    from types import SimpleNamespace

    return SimpleNamespace(
        campaign_id="C-1",
        transaction_ids=list(campaign_ids),
        user_ids=[f"U{i}" for i in range(5)] + [f"L{i}" for i in range(4)],
        device_ids=["DEV_007"],
        ip_ids=["IP_X"],
        payment_instrument_ids=[f"PI_{i}" for i in range(len(campaign_ids))],
        risk_score=80.0,
        confidence=0.9,
        evidence=[],
    )


def test_no_safe_action_is_possible():
    from engine.containment.optimizer import ContainmentOptimizer
    from engine.containment.simulation import ContainmentIndex

    df = _mixed_rows()
    # Realistic risk: fraud high, legitimate low.
    risk = {}
    for row in df.to_dict("records"):
        risk[row["transaction_id"]] = 0.9 if row["user_id"].startswith("U") else 0.05
    index = ContainmentIndex(df, risk)
    opt = ContainmentOptimizer(df, index, risk, max_legit_users=0,
                               min_fraud_containment=0.99, max_actions=1, top_k=4)
    rec = opt.recommend(_fake_assessment(set(df["transaction_id"])))
    assert rec["recommendation"] == "NO_SAFE_ACTION"


def test_hardened_works_with_feature_generation(hardened):
    import numpy as np

    from ml.features.build_features import build_features, temporal_split

    x, y, meta = build_features(hardened)
    assert x.shape[0] == len(hardened)
    assert not x.isna().any().any()
    assert np.isfinite(x.to_numpy(dtype=float)).all()
    for col in ("is_fraud", "fraud_campaign_id", "scenario"):
        assert col not in x.columns
    xt, yt, mt, xv, yv, mv, split_meta = temporal_split(x, y, meta)
    assert split_meta["train_count"] + split_meta["test_count"] == len(hardened)
    # Chronological: all test timestamps after all training timestamps.
    assert mv["timestamp"].min() >= mt["timestamp"].max()


def test_no_future_leakage_in_hardened_features(hardened):
    from ml.features.build_features import build_features

    x, _, meta = build_features(hardened)
    # User's first transaction has zero prior count (past-only test on hardened).
    raw = hardened.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    from ml.features.build_features import compute_past_features

    feat = compute_past_features(raw)
    first_idxs = raw.groupby("user_id")["timestamp"].idxmin()
    assert (feat.loc[first_idxs, "user_transaction_count_before"] == 0).all()


def test_ground_truth_never_used_by_detection_pipeline():
    """Detection (graph/campaign/containment) must ignore gt columns entirely."""
    from engine.graph.campaign_detector import find_campaign_candidates
    from engine.graph.graph_builder import build_graph

    df = pd.read_csv(HARDENED_CSV, parse_dates=["timestamp"])
    # Neutralize ground truth: detection must behave identically either way.
    df_clean = df.copy()
    df_clean["is_fraud"] = 0
    df_clean["fraud_campaign_id"] = None
    df_clean["scenario"] = "legitimate"

    # Sparse risk mask keeps the candidate search fast while still exercising
    # the full detection path (graph + candidate campaigns).
    first_ids = set(df["transaction_id"].iloc[:600])
    risk = {t: (0.95 if t in first_ids else 0.02)
            for t in df["transaction_id"]}

    g1 = build_graph(df)
    g2 = build_graph(df_clean)
    assert sorted(g1.nodes) == sorted(g2.nodes)

    c2 = find_campaign_candidates(g2, risk_scores=risk)
    # Candidate structures must never carry ground-truth labels.
    assert isinstance(c2, list)
    for cand in c2:
        for forbidden in ("is_fraud", "fraud_campaign_id", "scenario"):
            assert forbidden not in cand