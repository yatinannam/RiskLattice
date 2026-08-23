"""Phase 4 experiment report — reproducible campaign intelligence.

Run:  python engine/risk/experiment.py

Produces candidate/assessed campaign counts, risk-level distribution, exposure,
fraud/legitimate coverage, Phase-2 false-negative recovery, and campaign
precision/recall. Ground-truth labels are used ONLY for evaluation, never in
the scoring pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly: python engine/risk/experiment.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from engine.graph.campaign_detector import (
    find_campaign_candidates,
    load_phase2_risk_scores,
)
from engine.graph.graph_builder import build_graph
from engine.risk.risk_engine import (
    TxIndex,
    analyze_false_negatives,
    assess_all,
    deduplicate_candidates,
    evaluate_against_ground_truth,
    rank_campaigns,
)

CSV = _PROJECT_ROOT / "data" / "samples" / "transactions.csv"
MIN_RISK_LEVEL = "HIGH"


def _load_phase2_predictions(df: pd.DataFrame) -> dict[str, int]:
    """Load Phase-2 test predictions (binary) at the model threshold."""
    import json
    import joblib
    from ml.features.build_features import build_features

    thr_path = _PROJECT_ROOT / "ml" / "artifacts" / "logistic_regression_threshold.json"
    threshold = json.loads(thr_path.read_text(encoding="utf-8"))["threshold"]
    pipeline = joblib.load(_PROJECT_ROOT / "ml" / "artifacts" / "logistic_regression.joblib")

    x, _y, meta = build_features(str(CSV))
    proba = pipeline.predict_proba(x)[:, 1]
    labels = (proba >= threshold).astype(int)
    return {tx: int(p) for tx, p in zip(meta["transaction_id"], labels)}


def main() -> None:
    df = pd.read_csv(CSV, parse_dates=["timestamp"])
    graph = build_graph(df)
    tx_index = TxIndex(df)
    risk_scores = load_phase2_risk_scores(str(CSV))

    print("Phase 4 - Campaign Intelligence Experiment")
    print("=" * 70)

    candidates = find_campaign_candidates(graph, risk_scores=risk_scores)
    print(f"Candidate campaigns (Phase 3): {len(candidates)}")
    print(f"Assessed after dedup: {len(deduplicate_candidates(candidates))}")

    deduped = deduplicate_candidates(candidates)
    assessments = assess_all(deduped, graph, tx_index, risk_scores)
    print(f"Assessed campaigns: {len(assessments)}")

    # Risk-level distribution & summary
    from collections import Counter
    dist = Counter(a.risk_level for a in assessments)
    print("\nRisk-level distribution:")
    for level in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        print(f"  {level}: {dist.get(level, 0)}")
    avg = sum(a.risk_score for a in assessments) / max(len(assessments), 1)
    critical_high = dist.get("CRITICAL", 0) + dist.get("HIGH", 0)
    total_exposure = sum(a.estimated_exposure for a in assessments)
    print(f"average campaign risk: {avg:.2f}")
    print(f"critical/high campaign count: {critical_high}")
    print(f"estimated transaction exposure (sum of all assessed campaigns): "
          f"INR {total_exposure:,.2f}")

    # Ground-truth evaluation (HIGH and above)
    metrics = evaluate_against_ground_truth(assessments, df, min_risk_level=MIN_RISK_LEVEL)
    print("\nGround-truth evaluation (min level = HIGH):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Phase-2 false negatives recovered by high-risk campaigns
    preds = _load_phase2_predictions(df)
    fn = analyze_false_negatives(assessments, df, preds, risk_scores=risk_scores,
                                 min_risk_level=MIN_RISK_LEVEL)
    recovered = [row for row in fn if row["campaigns"]]
    print(f"\nPhase-2 false negatives (full dataset): {len(fn)}")
    print(f"False negatives inside a high-risk campaign: {len(recovered)}")
    for row in recovered[:8]:
        print(f"  {row['false_negative_transaction_id']} "
              f"risk={row['transaction_risk']:.3f} campaigned="
              f"{[c['campaign_id'] for c in row['campaigns']]}")

    # Legitimate collateral analysis for high-risk campaigns
    legit_in = metrics["legit_tx_in_high_risk"]
    print(f"\nLegitimate collateral in high-risk campaigns: {legit_in} transactions")

    return {
        "candidate_count": len(candidates),
        "assessed_count": len(assessments),
        "risk_distribution": dict(dist),
        "average_risk": round(avg, 2),
        "critical_high_count": critical_high,
        "total_exposure_inr": round(total_exposure, 2),
        "metrics": metrics,
        "false_negatives_recovered": len(recovered),
    }


if __name__ == "__main__":
    main()