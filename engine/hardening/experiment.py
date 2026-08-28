"""Phase 5.5 hardening experiment — the primary adversarial evaluation.

Methodology (documented):
  1. Load the hardened dataset (``transactions_hardened.csv``, seed 2026).
  2. Build past-only features with the existing Phase-2 builder (unchanged) and
     a temporal 80/20 split.
  3. Train a fresh Logistic Regression on the hardened TRAINING split only
     (never the held-out test).
  4. Evaluate the baseline on the hardened held-out test.
  5. Run the lattice over the FULL hardened dataset (graph + candidates + risk
     scoring + containment) using the trained model's risk probabilities —
     ground truth never feeds detection.
  6. False-negative recovery: baseline test-window FNs that sit inside a
     high-risk RiskLattice campaign -> recovery %.
  7. Containment over high-risk test-window campaigns.
  8. Per-scenario table (ground-truth labels used only here).

Output: ml/artifacts/hardening_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

HARDENED_CSV = _PROJECT_ROOT / "data" / "samples" / "transactions_hardened.csv"
OUT = _PROJECT_ROOT / "ml" / "artifacts" / "hardening_report.json"
HIGH_RISK_LEVEL = "HIGH"


def load_hardened() -> pd.DataFrame:
    if not HARDENED_CSV.exists():
        raise FileNotFoundError(
            "hardened dataset missing; run: python data/generators/generate_hardened.py"
        )
    return pd.read_csv(HARDENED_CSV, parse_dates=["timestamp"])


def _full_risk_probabilities(df, pipeline):
    """Infer fraud probabilities over the whole hardened dataset (past-only)."""
    from ml.features.build_features import build_features

    x, _y, meta = build_features(df)
    return {tx: float(p) for tx, p in zip(meta["transaction_id"],
                                          pipeline.predict_proba(x)[:, 1])}


def train_hardened_baseline(df) -> dict:
    """Train Phase-2 LR on the hardened temporal-train split; return artifacts."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from ml.evaluation.metrics import evaluate_binary
    from ml.features.build_features import build_features, temporal_split
    from ml.training.model import (
        build_pipeline,
        logistic_regression_config,
        select_threshold_from_oof,
    )

    x, y, meta = build_features(df)
    xt, yt, mt, xv, yv, mv, split_meta = temporal_split(x, y, meta)

    pipeline = build_pipeline(logistic_regression_config())
    proba_train = cross_val_predict(
        pipeline, xt, yt,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    thr_info = select_threshold_from_oof(proba_train, yt, default_threshold=0.50)
    threshold = thr_info["threshold"]
    pipeline.fit(xt, yt)

    proba_test = pipeline.predict_proba(xv)[:, 1]
    metrics = evaluate_binary(yv, proba_test, threshold=threshold)

    test_ids = list(mv["transaction_id"])
    full_risk = _full_risk_probabilities(df, pipeline)

    return {
        "target": "hardened_baseline",
        "trained_on": "hardened_temporal_train_only",
        "threshold": threshold,
        "classification_metrics": {
            k: metrics[k] for k in (
                "precision", "recall", "f1", "roc_auc", "pr_auc",
                "true_positive", "true_negative", "false_positive",
                "false_negative", "false_positive_rate", "fraud_detection_rate",
            )
        },
        "split": split_meta,
        "test_transaction_ids": test_ids,
        "test_y": {tx: int(y) for tx, y in zip(test_ids, yv)},
        "test_probabilities": {tx: float(p) for tx, p in zip(test_ids, proba_test)},
        "test_predictions": {
            tx: int(p >= threshold) for tx, p in zip(test_ids, proba_test)
        },
        "full_risk_probabilities": full_risk,
    }


def run_lattice(df, baseline) -> dict:
    """Build graph + campaigns + risk + containment over the hardened dataset.

    Uses only model risk probabilities (``baseline['full_risk_probabilities']``)
    for suspicion; ground truth is never read by detection. Ground truth is used
    here ONLY to compute evaluation metrics inside this function.
    """
    from engine.containment.optimizer import ContainmentOptimizer
    from engine.containment.simulation import ContainmentIndex
    from engine.graph.campaign_detector import find_campaign_candidates
    from engine.graph.graph_builder import build_graph
    from engine.risk.risk_engine import assess_all, rank_campaigns

    risk = baseline["full_risk_probabilities"]
    graph = build_graph(df)
    candidates = find_campaign_candidates(graph, risk_scores=risk)

    from engine.risk.risk_engine import TxIndex

    assessments = assess_all(candidates, graph, TxIndex(df), risk)
    ranked = rank_campaigns(assessments)

    # Ground-truth labels (evaluation only).
    gt = df.set_index("transaction_id")["is_fraud"].astype(int).to_dict()
    test_ids = set(baseline["test_transaction_ids"])

    def _risk_level_met(assessment, level):
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        return order[assessment.risk_level] >= order[level]

    high_risk = [a for a in ranked if _risk_level_met(a, HIGH_RISK_LEVEL)]
    test_high = [a for a in high_risk if any(t in test_ids for t in a.transaction_ids)]

    # Fraud coverage & legitimate impact over the TEST window only.
    # (Campaigns span train+test; restrict to test-window tx for the held-out
    # evaluation to avoid counting train fraud.)
    all_tx = {t for a in high_risk for t in a.transaction_ids if t in test_ids}
    fraud_in = sum(1 for t in all_tx if gt.get(t, 0) == 1)
    legit_in = sum(1 for t in all_tx if gt.get(t, 0) == 0)
    total_fraud_test = sum(1 for t in test_ids if gt.get(t, 0) == 1)
    total_legit_test = sum(1 for t in test_ids if gt.get(t, 0) == 0)

    # Baseline false negatives (test window) recovered by high-risk campaigns.
    base_fn_ids = [t for t, p in baseline["test_predictions"].items()
                   if p == 0 and baseline["test_y"].get(t, 0) == 1]
    test_high_tx = {t for a in high_risk for t in a.transaction_ids if t in test_ids}
    recovered = [t for t in base_fn_ids if t in test_high_tx]

    # Containment over high-risk test-window campaigns.
    cindex = ContainmentIndex(df, risk, gt_is_fraud={k: bool(v) for k, v in gt.items()})
    optimizer = ContainmentOptimizer(df, cindex, risk)
    containment_rows = []
    no_safe = 0
    for assessment in test_high:
        rec = optimizer.recommend(assessment)
        if rec["recommendation"] == "NO_SAFE_ACTION":
            no_safe += 1
        else:
            containment_rows.append(rec["recommended_strategy"])

    return {
        "candidate_count": len(candidates),
        "assessed_count": len(assessments),
        "high_risk_count": len(high_risk),
        "test_window_high_risk_campaigns": len(test_high),
        "fraud_transaction_coverage": round(fraud_in / total_fraud_test, 4) if total_fraud_test else 0.0,
        "legitimate_impact": round(legit_in / total_legit_test, 4) if total_legit_test else 0.0,
        "baseline_false_negatives_test": len(base_fn_ids),
        "false_negatives_recovered": len(recovered),
        "recovery_percentage": round(len(recovered) / max(len(base_fn_ids), 1) * 100, 2),
        "no_safe_action_count": no_safe,
        "containment_actions": len(containment_rows),
        "average_containment": float(sum(r["fraud_containment_rate"] for r in containment_rows) / len(containment_rows)) if containment_rows else 0.0,
    }


def main() -> dict:
    df = load_hardened()
    baseline = train_hardened_baseline(df)
    lattice = run_lattice(df, baseline)
    report = {"dataset": "hardened", "seed": 2026, "baseline": baseline, "lattice": lattice}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "baseline"}, indent=2, default=str))
    return report


if __name__ == "__main__":
    main()