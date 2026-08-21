"""Held-out evaluation for a trained RiskLattice baseline model.

Usage:

    python ml/evaluation/evaluate.py --model logistic_regression [--csv path]

The evaluation rebuilds past-only features, performs the same chronological
80/20 split, transforms the held-out test via the *fitted* training pipeline
(no refit), and computes all documented metrics plus the demo expected cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly: python ml/evaluation/evaluate.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from joblib import dump

from ml.evaluation.metrics import evaluate_binary
from ml.features.build_features import build_features, temporal_split
from ml.inference.predict import load_model

DEFAULT_CSV = "data/samples/transactions.csv"
ARTIFACT_DIR = _PROJECT_ROOT / "ml" / "artifacts"


def evaluate_artifact(model_name: str = "logistic_regression", csv_path: str = DEFAULT_CSV) -> dict:
    """Load a persisted model and evaluate it on the held-out test split."""
    X, y, meta = build_features(csv_path)
    xt, yt, mt, xv, yv, mv, split_meta = temporal_split(X, y, meta)

    model_path = ARTIFACT_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No artifact at {model_path}. Run train_baseline.py first "
            f"(python ml/training/train_baseline.py --model {model_name})."
        )
    pipeline = load_model(model_path)

    threshold_meta_path = ARTIFACT_DIR / f"{model_name}_threshold.json"
    threshold = 0.50
    if threshold_meta_path.exists():
        threshold = json.loads(threshold_meta_path.read_text(encoding="utf-8"))["threshold"]

    proba = pipeline.predict_proba(xv)[:, 1]
    result = evaluate_binary(yv, proba, threshold=threshold)
    result["model"] = model_name
    result["split"] = split_meta

    out_path = ARTIFACT_DIR / f"{model_name}_metrics.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    _print_metrics(result, split_meta)
    return result


def _print_metrics(result: dict, split_meta: dict) -> None:
    print("============================================================")
    print("RiskLattice Baseline - Held-out Evaluation")
    print("============================================================")
    print(f"Model:      {result['model']}")
    print(f"Threshold:  {result['threshold']}")
    print(f"Test rows:  {split_meta['test_count']}")
    print(f"Precision:  {result['precision']}")
    print(f"Recall:     {result['recall']}")
    print(f"F1:         {result['f1']}")
    print(f"ROC-AUC:    {result['roc_auc']}")
    print(f"PR-AUC:     {result['pr_auc']}")
    print(f"TP/FP:      {result['true_positive']} / {result['false_positive']}")
    print(f"TN/FN:      {result['true_negative']} / {result['false_negative']}")
    print("--- Demo cost section (assumptions, not Razorpay economics) ---")
    print(f"False-positive cost: INR {result['false_positive_cost']}")
    print(f"False-negative cost: INR {result['false_negative_cost']}")
    print(f"Total expected cost: INR {result['total_expected_cost']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a RiskLattice baseline")
    parser.add_argument("--model", default="logistic_regression",
                        choices=["logistic_regression", "random_forest"])
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()
    evaluate_artifact(model_name=args.model, csv_path=args.csv)