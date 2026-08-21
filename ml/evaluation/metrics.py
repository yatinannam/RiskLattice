"""Evaluation metrics for the RiskLattice transaction-level baseline.

Metrics are computed on a held-out test set produced by a chronological split.
``expected_cost`` is a DEMO cost model — it does not represent Razorpay's real
economic figures. Assumptions are documented below and configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class CostConfig:
    """Demo cost assumptions (NOT Razorpay economics).

    - false_positive_cost:  monetary cost of blocking a legitimate transaction
                            (INR). Includes potential refund handling, support,
                            and lost goodwill per affected transaction.
    - false_negative_cost:  monetary cost of one fraudulent transaction passing
                            (INR) — the amount siphoned in the demo campaign.

    Both are *demo assumptions* used to compare strategies on an equal basis.
    """

    false_positive_cost: float = 500.0
    false_negative_cost: float = 2500.0


DEFAULT_COST_CONFIG = CostConfig()


def evaluate_binary(
    y_true,
    proba,
    threshold: float = 0.50,
    cost_config: CostConfig = DEFAULT_COST_CONFIG,
) -> dict[str, Any]:
    """Compute the full evaluation result dictionary for a threshold.

    ``proba`` is the predicted probability of the positive (fraud) class.
    ``y_true`` is the true 0/1 label series. Returns the documented metrics
    plus the demo expected cost.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    # PR-AUC uses all thresholds so it is threshold-independent.
    pr_auc = average_precision_score(y_true, proba)
    try:
        roc_auc = roc_auc_score(y_true, proba)
    except ValueError as exc:  # only one class present in test
        roc_auc = float("nan")
        _ = exc

    # Standard false-positive rate: FP / (FP + TN).
    denominator_fpr = float(fp + tn)
    false_positive_rate = fp / denominator_fpr if denominator_fpr > 0 else 0.0
    fraud_detection_rate = recall  # share of fraud caught

    fp_cost = float(fp) * cost_config.false_positive_cost
    fn_cost = float(fn) * cost_config.false_negative_cost
    total_expected_cost = fp_cost + fn_cost
    total_expected_label = fp + fn

    return {
        "threshold": float(threshold),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "false_positive_rate": round(float(false_positive_rate), 4),
        "fraud_detection_rate": round(float(fraud_detection_rate), 4),
        # Demo cost section (clearly labeled, configurable).
        "false_positive_cost": round(fp_cost, 2),
        "false_negative_cost": round(fn_cost, 2),
        "total_expected_cost": round(total_expected_cost, 2),
        "expected_wrong_total": int(total_expected_label),
        "cost_model": {
            "demo_assumptions": True,
            "per_false_positive_inr": cost_config.false_positive_cost,
            "per_false_negative_inr": cost_config.false_negative_cost,
            "note": "Demo cost assumptions, not Razorpay internal economics.",
        },
    }