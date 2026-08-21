"""Train and persist a RiskLattice transaction-level baseline model.

Usage:

    python ml/training/train_baseline.py [--model logistic_regression|random_forest] [--csv path]

Flow:
    1. Load the Phase-1 synthetic transaction CSV.
    2. Build past-only features (leakage-free).
    3. Chronological 80/20 train/test split (test never touches training).
    4. Optionally pick a probability threshold using out-of-fold TRAINING
       predictions; otherwise use the 0.50 default.
    5. Fit preprocessing + estimator on training data only.
    6. Persist the fitted pipeline, split metadata, and run report to
       ml/artifacts/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from joblib import dump
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Allow running directly: python ml/training/train_baseline.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.features.build_features import (
    build_features,
    ensure_no_ground_truth,
    temporal_split,
)
from ml.training.model import (
    build_pipeline,
    logistic_regression_config,
    random_forest_config,
    select_threshold_from_oof,
)

DEFAULT_CSV = "data/samples/transactions.csv"
ARTIFACT_DIR = _PROJECT_ROOT / "ml" / "artifacts"


def _resolve_config(model_name: str):
    if model_name == "logistic_regression":
        return logistic_regression_config()
    if model_name == "random_forest":
        return random_forest_config()
    raise ValueError(f"Unknown model '{model_name}'")


def train_baseline(
    csv_path: str = DEFAULT_CSV,
    model_name: str = "logistic_regression",
    auto_threshold: bool = True,
    default_threshold: float = 0.50,
) -> dict:
    """Run the full baseline training and return a run report dict."""
    X, y, meta = build_features(csv_path)
    ensure_no_ground_truth(X)
    xt, yt, mt, xv, yv, mv, split_meta = temporal_split(X, y, meta)

    config = _resolve_config(model_name)
    pipeline = build_pipeline(config)

    threshold_info = {"threshold": default_threshold}
    if auto_threshold:
        proba = cross_val_predict(
            pipeline,
            xt,
            yt,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            method="predict_proba",
        )[:, 1]
        threshold_info = select_threshold_from_oof(
            proba, yt, default_threshold=default_threshold
        )

    pipeline.fit(xt, yt)

    artifact_dir = ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{model_name}.joblib"

    report = {
        "model": config.name,
        "config": {
            "estimator": str(pipeline.named_steps["clf"]),
            "scale": config.scale,
            "class_weight": "balanced (see model.py rationale)",
            "random_state": 42,
        },
        "threshold": threshold_info,
        "split": split_meta,
        "features_count": int(X.shape[1]),
        "fraud_train": int(yt.sum()),
        "fraud_test": int(yv.sum()),
        "artifacts": {
            "model": str(model_path),
            "split_metadata": str(artifact_dir / f"{model_name}_split.json"),
            "threshold_metadata": str(artifact_dir / f"{model_name}_threshold.json"),
            "report": str(artifact_dir / f"{model_name}_report.json"),
        },
    }

    dump(pipeline, model_path)
    dump(split_meta, artifact_dir / f"{model_name}_split.joblib")
    (artifact_dir / f"{model_name}_split.json").write_text(
        json.dumps(split_meta, indent=2), encoding="utf-8"
    )
    (artifact_dir / f"{model_name}_threshold.json").write_text(
        json.dumps(threshold_info, indent=2), encoding="utf-8"
    )
    (artifact_dir / f"{model_name}_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    split = report["split"]
    print("============================================================")
    print("RiskLattice Transaction-Level Baseline - Training Report")
    print("============================================================")
    print(f"Dataset: {split['train_count'] + split['test_count']:,} transactions")
    print(f"Train: {split['train_count']:,}")
    print(f"Test:  {split['test_count']:,}")
    print(f"Features: {report['features_count']}")
    print(f"Fraud train: {report['fraud_train']}")
    print(f"Fraud test:  {report['fraud_test']}")
    print(f"Model: {report['model']}")
    thr = report["threshold"]
    print(f"Threshold: {thr['threshold']} (selection: {thr['threshold_selection']})")
    print("Adapted to: ", report["config"]["estimator"])
    print("Artifacts:")
    for key, path in report["artifacts"].items():
        print(f"  {key}: {path}")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RiskLattice baseline model")
    parser.add_argument("--model", default="logistic_regression",
                        choices=["logistic_regression", "random_forest"])
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--no-auto-threshold", action="store_true")
    args = parser.parse_args()
    train_baseline(
        csv_path=args.csv,
        model_name=args.model,
        auto_threshold=not args.no_auto_threshold,
    )