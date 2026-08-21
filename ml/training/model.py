"""Model configurations and pipeline construction for the baseline.

RiskLattice's baseline is intentionally simple: the purpose is an honest,
measurable transaction-level comparison point for later graph/campaign phases,
not a leaderboard. Configurations below are defensible defaults, documented.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelConfig:
    """Descriptive configuration for one baseline model family."""

    name: str
    estimator_factory: callable
    scale: bool = True


def logistic_regression_config() -> ModelConfig:
    """Linear baseline.

    ``class_weight="balanced"`` is used deliberately: the fraud class is the
    minority (~15%) and we are building a *fraud detection* baseline, so
    down-weighting majority samples improves minority recall. The cost of this
    choice (higher false positives / lower precision) is reported honestly.
    """
    return ModelConfig(
        name="logistic_regression",
        estimator_factory=lambda: LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        scale=True,
    )


def random_forest_config() -> ModelConfig:
    """Tree ensemble baseline (comparison only, no heavy tuning)."""
    return ModelConfig(
        name="random_forest",
        estimator_factory=lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scale=False,
    )


def build_pipeline(config: ModelConfig) -> Pipeline:
    """Build a fit-on-train-only preprocessing + estimator pipeline.

    All columns are numeric (categoricals one-hot encoded upstream), so the
    preprocessing is an imputer + optional StandardScaler. Preprocessing is
    fit exclusively on training data inside ``fit``; the test set is only ever
    transformed with the fitted training parameters.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if config.scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", config.estimator_factory()))
    return Pipeline(steps=steps)


def select_threshold_from_oof(
    oof_probabilities,
    y_train,
    candidates=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
               0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    default_threshold: float = 0.50,
) -> dict:
    """Pick a probability threshold using out-of-fold TRAINING predictions.

    A threshold may be chosen by scanning a candidate grid and maximizing F1
    on out-of-fold predictions produced from the training set alone. The
    held-out test set is never used for threshold selection.

    Returns metadata describing the choice plus the chosen threshold.
    """
    best_threshold = default_threshold
    best_f1 = -1.0
    for t in candidates:
        preds = (oof_probabilities >= t).astype(int)
        score = f1_score(y_train, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)

    return {
        "threshold": best_threshold,
        "threshold_selection": "train_oof_f1_max",
        "oof_f1_at_threshold": round(float(best_f1), 4),
        "default_threshold": default_threshold,
    }