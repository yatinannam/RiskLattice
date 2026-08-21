"""Feature definitions for the RiskLattice transaction-level pipeline.

This module is the single source of truth for:

  * the definitive, deterministic ordering of model feature columns
  * the ground-truth columns that must never enter the model feature matrix
  * window configuration for velocity features
  * the documented null/default strategy for first-seen entities

All features here are *past-only*: they are computed from information available
strictly before the current transaction's timestamp. They must never include
future information (which would leak the outcome into training).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Ground truth. Used ONLY for evaluation, labeling, and debugging. These
# columns must NEVER be part of the model feature matrix X.
# ---------------------------------------------------------------------------
GROUND_TRUTH_COLUMNS: list[str] = [
    "is_fraud",
    "fraud_campaign_id",
    "scenario",
]

# Metadata columns carried alongside X for evaluation/debugging.
# They are NOT passed to the model.
METADATA_COLUMNS: list[str] = [
    "transaction_id",
    "timestamp",
    "scenario",
    "fraud_campaign_id",
]

# ---------------------------------------------------------------------------
# Transaction-level features (derived directly from the row; no temporal
# aggregates). Categorical fields are one-hot encoded into 0/1 numeric columns
# because Logistic Regression cannot consume raw categorical strings.
# ---------------------------------------------------------------------------
TRANSACTION_FEATURES: list[str] = [
    "amount",
    "log_amount",
    "hour",
    "day_of_week",
    "is_weekend",
    "payment_method_upi",
    "payment_method_card",
    "payment_method_netbanking",
    "payment_method_wallet",
    "status_success",
    "status_failed",
    "status_refunded",
]

# ---------------------------------------------------------------------------
# Past-only user-level aggregate features (exclude the current transaction).
# ---------------------------------------------------------------------------
USER_FEATURES: list[str] = [
    "user_transaction_count_before",
    "user_success_count_before",
    "user_failed_count_before",
    "user_refund_count_before",
    "user_success_rate_before",
    "user_failure_rate_before",
    "user_refund_rate_before",
    "time_since_previous_user_transaction",
]

# ---------------------------------------------------------------------------
# Past-only device-level aggregate features.
# ---------------------------------------------------------------------------
DEVICE_FEATURES: list[str] = [
    "device_transaction_count_before",
    "device_unique_users_before",
    "time_since_previous_device_transaction",
]

# ---------------------------------------------------------------------------
# Past-only IP-level aggregate features.
# A shared IP is evidence, never automatic proof of fraud.
# ---------------------------------------------------------------------------
IP_FEATURES: list[str] = [
    "ip_transaction_count_before",
    "ip_unique_users_before",
    "time_since_previous_ip_transaction",
]

# ---------------------------------------------------------------------------
# Past-only payment-instrument aggregate features.
# ---------------------------------------------------------------------------
PAYMENT_INSTRUMENT_FEATURES: list[str] = [
    "payment_instrument_transaction_count_before",
    "payment_instrument_unique_users_before",
    "time_since_previous_payment_transaction",
]

# ---------------------------------------------------------------------------
# Past-window velocity features (seconds-before-current, current excluded).
# ---------------------------------------------------------------------------
VELOCITY_FEATURES: list[str] = [
    "user_transactions_last_5m",
    "user_transactions_last_1h",
    "user_transactions_last_24h",
    "device_transactions_last_5m",
    "device_transactions_last_1h",
    "ip_transactions_last_5m",
    "ip_transactions_last_1h",
    "payment_transactions_last_5m",
    "payment_transactions_last_1h",
]

# ---------------------------------------------------------------------------
# The definitive ordered feature list. Any model X matrix must use exactly
# this column ordering (deterministic), and nothing else.
# ---------------------------------------------------------------------------
ALL_FEATURES: list[str] = (
    TRANSACTION_FEATURES
    + USER_FEATURES
    + DEVICE_FEATURES
    + IP_FEATURES
    + PAYMENT_INSTRUMENT_FEATURES
    + VELOCITY_FEATURES
)

# ---------------------------------------------------------------------------
# Velocity windows (seconds). Used to derive the *_last_*m / *_last_*h columns.
# ---------------------------------------------------------------------------
VELOCITY_WINDOW_5M = 5 * 60
VELOCITY_WINDOW_1H = 60 * 60
VELOCITY_WINDOW_24H = 24 * 60 * 60
LONGEST_VELOCITY_WINDOW = VELOCITY_WINDOW_24H

# ---------------------------------------------------------------------------
# Default value for time-since-previous features when an entity has no prior
# activity ("first time seen"). A documented 1-year sentinel is used instead of
# NaN so the final model matrix contains no missing values. This is a
# deterministic, documented rule, not a data transform trained on labels.
# ---------------------------------------------------------------------------
NO_PRIOR_TIME_SENTINEL: float = 365 * 24 * 60 * 60.0  # ~31,536,000 seconds