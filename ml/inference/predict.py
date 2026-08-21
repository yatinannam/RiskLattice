"""Inference helpers for a trained RiskLattice baseline pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from joblib import load


def load_model(model_path: str | Path):
    """Load a persisted sklearn/Jupyter pipeline from ``model_path``."""
    return load(str(model_path))


def predict_proba_pipeline(pipeline, x_features) -> list[float]:
    """Return fraud-class probabilities for a feature matrix."""
    return pipeline.predict_proba(x_features)[:, 1].tolist()


def predict_labels(pipeline, x_features, threshold: float = 0.5) -> list[int]:
    """Return 0/1 labels at a given probability threshold."""
    proba = predict_proba_pipeline(pipeline, x_features)
    return [1 if p >= threshold else 0 for p in proba]


def prediction_input_row(features: dict[str, Any], feature_order: list[str]):
    """Build a single feature row aligned to the trained column ordering."""
    return [[features[col] for col in feature_order]]