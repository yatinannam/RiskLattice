"""Tests for the baseline model training, preprocessing, and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.evaluation.metrics import CostConfig, evaluate_binary
from ml.features import build_features as fb
from ml.training.model import (
    build_pipeline,
    logistic_regression_config,
    random_forest_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "samples" / "transactions.csv"


@pytest.fixture(scope="module")
def split():
    x, y, meta = fb.build_features(str(DEFAULT_CSV))
    return fb.temporal_split(x, y, meta)


def test_model_trains_successfully(split):
    xt, yt, *_ = split
    pipeline = build_pipeline(logistic_regression_config())
    pipeline.fit(xt, yt)
    assert hasattr(pipeline.named_steps["clf"], "classes_")
    # Deterministic with fixed random_state.
    pipeline2 = build_pipeline(logistic_regression_config()).fit(xt, yt)
    assert (pipeline.predict_proba(xt) == pipeline2.predict_proba(xt)).all()


def test_preprocessing_fit_only_on_training(split):
    """Imputer/scaler statistics must come from training data alone."""
    xt, yt, mt, xv, yv, mv, _ = split
    pipeline = build_pipeline(logistic_regression_config()).fit(xt, yt)

    # Median imputer statistics equal training medians.
    np.testing.assert_allclose(
        pipeline.named_steps["imputer"].statistics_,
        xt.median().to_numpy(),
    )
    # StandardScaler means equal training means (not combined train+test mean).
    np.testing.assert_allclose(
        pipeline.named_steps["scaler"].mean_,
        xt.mean().to_numpy(),
        rtol=1e-6,
    )
    # The training mean should differ from the combined mean -> confirms we did
    # not silently fit on the whole dataset.
    combined_mean = np.concatenate([xt.to_numpy(), xv.to_numpy()]).mean(axis=0)
    assert not np.allclose(pipeline.named_steps["scaler"].mean_, combined_mean,
                           atol=1e-6)


def test_random_forest_trains(split):
    xt, yt, *_ = split
    pipeline = build_pipeline(random_forest_config()).fit(xt, yt)
    assert hasattr(pipeline.named_steps["clf"], "classes_")


def test_evaluation_returns_required_metrics():
    y_true = np.array([1, 0, 1, 0, 1, 1, 0])
    proba = np.array([0.9, 0.1, 0.6, 0.2, 0.7, 0.8, 0.3])
    result = evaluate_binary(y_true, proba, threshold=0.5)

    required = {
        "precision", "recall", "f1", "roc_auc", "pr_auc",
        "true_positive", "true_negative", "false_positive", "false_negative",
        "false_positive_rate", "fraud_detection_rate",
        "false_positive_cost", "false_negative_cost", "total_expected_cost",
    }
    assert required <= set(result)
    assert result["threshold"] == 0.5
    assert result["true_positive"] + result["false_negative"] == int(y_true.sum())


def test_evaluation_respects_cost_model():
    y_true = np.array([1, 1, 1, 0])
    proba = np.array([0.1, 0.2, 0.3, 0.05])
    costs = CostConfig(false_positive_cost=100.0, false_negative_cost=50.0)
    result = evaluate_binary(y_true, proba, threshold=0.5, cost_config=costs)
    assert result["false_positive"] == 0
    assert result["false_negative"] == 3
    assert result["true_negative"] == 1
    assert result["false_negative_cost"] == 150.0
    assert result["false_positive_cost"] == 0.0


def test_threshold_metadata_defined_for_both_models(split):
    from ml.training.model import select_threshold_from_oof

    xt, yt, *_ = split
    rng = np.random.RandomState(0)
    oof = rng.uniform(0, 1, size=len(yt))
    info = select_threshold_from_oof(oof, yt, default_threshold=0.5)
    assert "threshold" in info
    assert 0.0 <= info["threshold"] <= 1.0