"""Feature engineering tests for the RiskLattice transaction-level pipeline.

Covers the leakage rules (no ground truth, deterministic ordering, past-only
aggregates, no future influence) plus the data-quality requirement that a
single suspicious signal never automatically implies fraud.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.features import build_features as fb
from ml.features.feature_definitions import ALL_FEATURES, GROUND_TRUTH_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "samples" / "transactions.csv"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def memdata():
    """Public build_features output: (X, y, metadata)."""
    return fb.build_features(str(DEFAULT_CSV))


@pytest.fixture(scope="module")
def sorted_features():
    """Sorted raw frame plus computed past-only features (for entity lookups)."""
    raw = fb.load_dataset(str(DEFAULT_CSV))
    feat = fb.compute_past_features(raw)
    return raw, feat


# --------------------------------------------------------------------------
# Leakage guards
# --------------------------------------------------------------------------

def test_feature_matrix_has_no_ground_truth(memdata):
    x, _, _ = memdata
    fb.ensure_no_ground_truth(x)
    assert not (set(GROUND_TRUTH_COLUMNS) & set(x.columns))


def test_deterministic_column_ordering(memdata):
    x, _, _ = memdata
    assert list(x.columns) == ALL_FEATURES


def test_feature_generation_is_deterministic():
    a = fb.build_features(str(DEFAULT_CSV))[0]
    b = fb.build_features(str(DEFAULT_CSV))[0]
    assert (a.values == b.values).all()
    assert list(a.columns) == list(b.columns)


def test_no_nan_inf_in_matrix(memdata):
    x, _, _ = memdata
    assert not x.isna().any().any()
    assert np.isfinite(x.to_numpy(dtype=float)).all()


# --------------------------------------------------------------------------
# Past-only / no-future-leakage rules
# --------------------------------------------------------------------------

def test_first_user_transaction_has_zero_prior_count(sorted_features):
    raw, feat = sorted_features
    first_idx = raw.groupby("user_id")["timestamp"].idxmin()
    assert (feat.loc[first_idx, "user_transaction_count_before"] == 0).all()


def test_historical_counts_exclude_current(sorted_features):
    """For each user's k-th transaction, count_before equals the number of that
    user's strictly earlier transactions (k), so the current tx is not counted."""
    raw, feat = sorted_features
    merged = pd.concat(
        [raw[["user_id", "timestamp"]], feat[["user_transaction_count_before"]]],
        axis=1,
    ).sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    for _, group in merged.groupby("user_id"):
        positions = group.reset_index(drop=True).index.to_numpy()
        counts = group["user_transaction_count_before"].to_numpy()
        assert (counts == positions).all()


def test_user_velocity_excludes_current(sorted_features):
    """Velocity counts are a subset of prior activity: <= count_before."""
    _, feat = sorted_features
    for col in ("user_transactions_last_5m", "user_transactions_last_1h",
                "user_transactions_last_24h"):
        assert (feat[col] <= feat["user_transaction_count_before"]).all()


def test_future_transactions_do_not_influence_past(sorted_features):
    raw, feat = sorted_features
    # Latest transaction of each user counts all prior (n-1) but not itself.
    last_idx = raw.groupby("user_id")["timestamp"].idxmax()
    user_of_last = raw.loc[last_idx, "user_id"]
    total_per_user = raw["user_id"].value_counts()
    expected = (total_per_user.reindex(user_of_last.to_numpy()) - 1).to_numpy()
    got = feat.loc[last_idx, "user_transaction_count_before"].to_numpy()
    assert (got == expected).all()


# --------------------------------------------------------------------------
# Temporal split correctness
# --------------------------------------------------------------------------

def test_train_test_split_is_chronological(memdata):
    x, y, meta = memdata
    xt, yt, mt, xv, yv, mv, split_meta = fb.temporal_split(x, y, meta)
    assert split_meta["train_count"] + split_meta["test_count"] == len(x)
    assert float(split_meta["train_count"]) >= 0.79 * len(x)
    # All test timestamps occur after all training timestamps.
    assert (mv["timestamp"] >= mt["timestamp"].max()).all()
    assert mv["timestamp"].min() >= mt["timestamp"].max()


test_test_timestamps_after_train = test_train_test_split_is_chronological


# --------------------------------------------------------------------------
# Data quality: legitimate transactions may carry suspicious-looking signals.
# A single signal must not imply fraud.
# --------------------------------------------------------------------------

def test_legitimate_transactions_have_shared_ip_signal(sorted_features):
    raw, feat = sorted_features
    legit_mask = raw["is_fraud"].astype(int).to_numpy() == 0
    assert (feat.loc[legit_mask, "ip_unique_users_before"] >= 2).any()


def test_legitimate_transactions_have_shared_device_signal(sorted_features):
    raw, feat = sorted_features
    legit_mask = raw["is_fraud"].astype(int).to_numpy() == 0
    assert (feat.loc[legit_mask, "device_unique_users_before"] >= 2).any()


def test_legitimate_transactions_include_refunds(sorted_features):
    raw, feat = sorted_features
    legit_mask = raw["is_fraud"].astype(int).to_numpy() == 0
    assert feat.loc[legit_mask, "status_refunded"].sum() > 0


def test_legitimate_transactions_include_high_value(sorted_features):
    raw, feat = sorted_features
    legit_mask = raw["is_fraud"].astype(int).to_numpy() == 0
    assert (feat.loc[legit_mask, "amount"] >= 50_000).any()