"""Build the transaction-level feature matrix for RiskLattice.

The feature computation is **past-only**: for every transaction, aggregate and
velocity features are derived only from transactions that occurred strictly
before the current transaction's timestamp. No future information is ever used,
so the outcome label can never leak backward into a row's own features.

The builder never includes ground-truth columns in the returned X matrix.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta
from math import log
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running directly:  python ml/features/build_features.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.features.feature_definitions import (
    ALL_FEATURES,
    GROUND_TRUTH_COLUMNS,
    METADATA_COLUMNS,
    NO_PRIOR_TIME_SENTINEL,
    VELOCITY_WINDOW_1H,
    VELOCITY_WINDOW_5M,
    VELOCITY_WINDOW_24H,
    LONGEST_VELOCITY_WINDOW,
)

logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.80


class _EntityState:
    """Mutable running aggregate for one entity (user/device/ip/instrument).

    Values held here represent **prior** activity only: callers must read
    features via the snapshot methods BEFORE calling ``observe`` so that the
    current transaction is never counted in its own past-only features.
    """

    __slots__ = ("count", "success", "failed", "refunded", "last_ts", "users", "timestamps", "track_users")

    def __init__(self, track_users: bool) -> None:
        self.count: float = 0.0
        self.success: float = 0.0
        self.failed: float = 0.0
        self.refunded: float = 0.0
        self.last_ts: datetime | None = None
        self.users: set[str] = set() if track_users else None
        self.timestamps: deque = deque()
        self.track_users = track_users

    def _prune(self, current: datetime) -> None:
        """Drop timestamps outside the longest velocity window (24h)."""
        cutoff = current - timedelta(seconds=LONGEST_VELOCITY_WINDOW)
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def count_in_window(self, current: datetime, window_seconds: float) -> float:
        """Count prior timestamps within ``window_seconds`` before ``current``."""
        self._prune(current)
        cutoff = current - timedelta(seconds=window_seconds)
        n = sum(1 for ts in self.timestamps if ts >= cutoff)
        return float(n)

    def time_since_previous(self, current: datetime) -> float:
        """Seconds since last prior transaction; sentinel if none seen."""
        if self.last_ts is None:
            return NO_PRIOR_TIME_SENTINEL
        return (current - self.last_ts).total_seconds()

    def observe(self, timestamp: datetime, status: str, user_id: str | None) -> None:
        """Register the current transaction into running state (after reads)."""
        self.count += 1.0
        if status == "success":
            self.success += 1.0
        elif status == "failed":
            self.failed += 1.0
        elif status == "refunded":
            self.refunded += 1.0
        self.last_ts = timestamp
        self.timestamps.append(timestamp)
        if self.track_users and user_id is not None:
            self.users.add(user_id)


def _rate(part: float, total: float) -> float:
    """Rate helper; returns 0 when there is no prior activity."""
    return part / total if total > 0 else 0.0


def compute_past_features(sorted_df: pd.DataFrame) -> pd.DataFrame:
    """Compute past-only features for a timestamp-sorted frame.

    ``sorted_df`` must be sorted ascending by (timestamp, transaction_id).
    The result contains exactly ALL_FEATURES columns, with no NaN.
    """

    user_state = defaultdict(lambda: _EntityState(track_users=False))
    device_state = defaultdict(lambda: _EntityState(track_users=True))
    ip_state = defaultdict(lambda: _EntityState(track_users=True))
    pi_state = defaultdict(lambda: _EntityState(track_users=True))

    rows: list[dict[str, Any]] = []

    for row in sorted_df.itertuples(index=False):
        ts: datetime = row.timestamp
        uid: str = row.user_id
        device: str = row.device_id
        ip: str = row.ip_id
        pi: str = row.payment_instrument_id
        status: str = row.status
        method: str = row.payment_method
        amount: float = float(row.amount)

        hour = float(ts.hour)
        day_of_week = float(ts.weekday())
        is_weekend = 1.0 if ts.weekday() >= 5 else 0.0

        # Read this row's past-only state BEFORE observing the transaction.
        u = user_state[uid]
        d = device_state[device]
        i = ip_state[ip]
        p = pi_state[pi]

        features: dict[str, Any] = {
            "amount": amount,
            "log_amount": log(amount),
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "payment_method_upi": 1.0 if method == "upi" else 0.0,
            "payment_method_card": 1.0 if method == "card" else 0.0,
            "payment_method_netbanking": 1.0 if method == "netbanking" else 0.0,
            "payment_method_wallet": 1.0 if method == "wallet" else 0.0,
            "status_success": 1.0 if status == "success" else 0.0,
            "status_failed": 1.0 if status == "failed" else 0.0,
            "status_refunded": 1.0 if status == "refunded" else 0.0,
            # User (past-only)
            "user_transaction_count_before": u.count,
            "user_success_count_before": u.success,
            "user_failed_count_before": u.failed,
            "user_refund_count_before": u.refunded,
            "user_success_rate_before": _rate(u.success, u.count),
            "user_failure_rate_before": _rate(u.failed, u.count),
            "user_refund_rate_before": _rate(u.refunded, u.count),
            "time_since_previous_user_transaction": u.time_since_previous(ts),
            # Device (past-only)
            "device_transaction_count_before": d.count,
            "device_unique_users_before": float(len(d.users)),
            "time_since_previous_device_transaction": d.time_since_previous(ts),
            # IP (past-only)
            "ip_transaction_count_before": i.count,
            "ip_unique_users_before": float(len(i.users)),
            "time_since_previous_ip_transaction": i.time_since_previous(ts),
            # Payment instrument (past-only)
            "payment_instrument_transaction_count_before": p.count,
            "payment_instrument_unique_users_before": float(len(p.users)),
            "time_since_previous_payment_transaction": p.time_since_previous(ts),
            # Velocity (past windows, current excluded)
            "user_transactions_last_5m": u.count_in_window(ts, VELOCITY_WINDOW_5M),
            "user_transactions_last_1h": u.count_in_window(ts, VELOCITY_WINDOW_1H),
            "user_transactions_last_24h": u.count_in_window(ts, VELOCITY_WINDOW_24H),
            "device_transactions_last_5m": d.count_in_window(ts, VELOCITY_WINDOW_5M),
            "device_transactions_last_1h": d.count_in_window(ts, VELOCITY_WINDOW_1H),
            "ip_transactions_last_5m": i.count_in_window(ts, VELOCITY_WINDOW_5M),
            "ip_transactions_last_1h": i.count_in_window(ts, VELOCITY_WINDOW_1H),
            "payment_transactions_last_5m": p.count_in_window(ts, VELOCITY_WINDOW_5M),
            "payment_transactions_last_1h": p.count_in_window(ts, VELOCITY_WINDOW_1H),
        }

        rows.append(features)

        # Commit this transaction only AFTER reading its features.
        u.observe(ts, status, None)
        d.observe(ts, status, uid)
        i.observe(ts, status, uid)
        p.observe(ts, status, uid)

    feature_df = pd.DataFrame(rows)
    return feature_df.reindex(columns=ALL_FEATURES)[ALL_FEATURES]


def load_dataset(path_or_df: str | pd.DataFrame) -> pd.DataFrame:
    """Load transactions and sort chronologically by (timestamp, transaction_id)."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df, parse_dates=["timestamp"])
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    return df


def build_features(
    source: str | pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Produce (X, y, metadata) from a CSV path or DataFrame.

    X      -> exactly ALL_FEATURES columns (sorted chronologically), no ground truth
    y      -> is_fraud label series aligned to the sorted rows
    meta   -> evaluation/debugging metadata (scenario, campaign id, timestamp)
    """
    sorted_df = load_dataset(source)
    x = compute_past_features(sorted_df)
    y = sorted_df["is_fraud"].astype(int).reset_index(drop=True)
    meta = sorted_df[METADATA_COLUMNS].reset_index(drop=True)
    return x, y, meta


def ensure_no_ground_truth(x: pd.DataFrame) -> None:
    """Hard guard: raise ValueError if any ground-truth column leaked into X."""
    found = [col for col in GROUND_TRUTH_COLUMNS if col in x.columns]
    if found:
        raise ValueError(
            f"Ground-truth columns leaked into the feature matrix: {found}"
        )


def temporal_split(
    x: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    dict,
]:
    """Chronological 80/20 split (earliest -> train, latest -> held-out test).

    The held-out test set is isolated: it is never used for feature
    fitting, model fitting, threshold selection, or cost calibration.
    """
    n = len(x)
    split_idx = int(n * train_ratio)

    x_train = x.iloc[:split_idx].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    meta_train = meta.iloc[:split_idx].reset_index(drop=True)

    x_test = x.iloc[split_idx:].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)
    meta_test = meta.iloc[split_idx:].reset_index(drop=True)

    split_metadata = {
        "method": "time_based_chronological",
        "train_ratio": train_ratio,
        "train_start": meta_train["timestamp"].min().isoformat(),
        "train_end": meta_train["timestamp"].max().isoformat(),
        "test_start": meta_test["timestamp"].min().isoformat(),
        "test_end": meta_test["timestamp"].max().isoformat(),
        "train_count": int(len(x_train)),
        "test_count": int(len(x_test)),
    }
    return x_train, y_train, meta_train, x_test, y_test, meta_test, split_metadata


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/transactions.csv"
    X, y, meta = build_features(csv_path)
    ensure_no_ground_truth(X)
    xt, yt, mt, xv, yv, mv, smeta = temporal_split(X, y, meta)
    print(f"Dataset rows: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Feature columns: {list(X.columns)}")
    print(f"Fraud positives overall: {int(y.sum())} ({y.mean():.1%})")
    print(f"Train: {len(xt)} (fraud {int(yt.sum())}) | Test: {len(xv)} (fraud {int(yv.sum())})")