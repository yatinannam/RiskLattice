"""Prepared application state for the RiskLattice API.

Loads the dataset, builds model predictions, the campaign graph, assessments,
containment recommendations and investigator evidence ONCE at startup and
caches them in memory. API routes read from this object only.

Ground-truth labels are never loaded into the prepared state.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Project root on path (mirrors engine module convention).
# app_state.py lives at apps/api/app/services/ -> 4 levels below project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("risklattice.app_state")


class AppState:
    """Holds the full prepared RiskLattice pipeline for the API."""

    def __init__(self, dataset_path: Path, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
        self.risk: dict[str, float] = {}
        self.graph = None
        self.assessments = []          # ranked CampaignAssessment list
        self.containments: dict[str, dict] = {}
        self.investigations: dict[str, dict] = {}
        self.index = None              # ContainmentIndex
        self.optimizer = None
        self.startup_seconds = 0.0
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        t0 = time.time()

        from engine.containment.optimizer import ContainmentOptimizer
        from engine.containment.simulation import ContainmentIndex
        from engine.graph.campaign_detector import find_campaign_candidates
        from engine.graph.graph_builder import build_graph
        from engine.risk.risk_engine import TxIndex, assess_all, rank_campaigns

        df = self.df

        # 1. Model predictions (risk probabilities) for every transaction.
        self.risk = self._risk_predictions(df)

        # 2. Relationship graph.
        self.graph = build_graph(df)

        # 3. Candidate campaigns + assessments.
        candidates = find_campaign_candidates(self.graph, risk_scores=self.risk)
        assessments = assess_all(
            candidates, self.graph, TxIndex(df), self.risk
        )
        self.assessments = rank_campaigns(assessments)

        # 4. Containment prepared eagerly for all campaigns; investigator
        #    reports built LAZILY on first request (and cached).
        self.index = ContainmentIndex(df, self.risk, gt_is_fraud={})
        self.optimizer = ContainmentOptimizer(df, self.index, self.risk)
        for assessment in self.assessments:
            cid = assessment.campaign_id
            self.containments[cid] = self.optimizer.recommend(assessment)

        self.startup_seconds = round(time.time() - t0, 2)
        logger.info(
            "AppState ready: dataset=%s campaigns=%d containment=%d "
            "startup=%.2fs",
            self.dataset_name, len(self.assessments), len(self.containments),
            self.startup_seconds,
        )

    # ------------------------------------------------------------------
    def investigation_for(self, campaign_id: str) -> dict:
        """Return a campaign's investigation report, building + caching it on
        first request. Falling back to an error dict keeps the API resilient.
        """
        cached = self.investigations.get(campaign_id)
        if cached is not None:
            return cached

        assessment = self.find_assessment(campaign_id)
        if assessment is None:
            return {"error": "campaign not found", "detail": "unknown campaign"}

        try:
            from engine.investigator.investigator import investigate_campaign

            result = investigate_campaign(
                assessment, self.containment_for(campaign_id))
        except Exception as exc:
            logger.exception(
                "investigation failed for %s: %s", campaign_id, exc)
            result = {"error": "investigation unavailable", "detail": str(exc)}

        self.investigations[campaign_id] = result
        return result

    # ------------------------------------------------------------------
    def _risk_predictions(self, df) -> dict[str, float]:
        """Return {transaction_id: fraud probability} for the whole dataset.

        For the baseline dataset we reuse the Phase-2 artifact. For the
        hardened dataset we train a fresh logistic regression on a temporal
        train split (deterministic), selecting the threshold on training OOF
        only.
        """
        from ml.features.build_features import build_features, temporal_split
        from ml.training.model import (
            build_pipeline,
            logistic_regression_config,
            select_threshold_from_oof,
        )

        x, _y, meta = build_features(df)
        pipeline = build_pipeline(logistic_regression_config())

        if self.dataset_name == "baseline":
            from joblib import load

            artifact = (_PROJECT_ROOT / "ml" / "artifacts"
                        / "logistic_regression.joblib")
            if artifact.exists():
                pipeline = load(artifact)
            proba_all = pipeline.predict_proba(x)[:, 1]
        else:
            from sklearn.model_selection import (
                StratifiedKFold,
                cross_val_predict,
            )

            xt, yt, mt, xv, yv, mv, _split = temporal_split(x, _y, meta)
            proba_train = cross_val_predict(
                pipeline, xt, yt,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                method="predict_proba",
            )[:, 1]
            _thr_info = select_threshold_from_oof(proba_train, yt,
                                                  default_threshold=0.50)
            pipeline.fit(xt, yt)
            proba_all = pipeline.predict_proba(x)[:, 1]

        return {tx: float(p) for tx, p in zip(meta["transaction_id"], proba_all)}

    # ------------------------------------------------------------------
    # Lookups used by routes
    # ------------------------------------------------------------------
    def find_assessment(self, campaign_id: str):
        for assessment in self.assessments:
            if assessment.campaign_id == campaign_id:
                return assessment
        return None

    def containment_for(self, campaign_id: str) -> dict:
        return self.containments.get(campaign_id, {})


_app_state_singleton: AppState | None = None


def build_app_state() -> AppState:
    from app.config import settings

    return AppState(settings.dataset_path, settings.dataset_name)


def get_app_state() -> AppState:
    """Return the prepared singleton AppState (built once at startup)."""
    global _app_state_singleton
    if _app_state_singleton is None:
        _app_state_singleton = build_app_state()
    return _app_state_singleton