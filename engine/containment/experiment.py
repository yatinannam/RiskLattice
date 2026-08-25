"""Phase 5 containment experiment — reproducible report.

Run:  python engine/containment/experiment.py

Produces per-campaign recommendations, strategy distribution, average
containment/collateral, NO_SAFE_ACTION count, the "block everything vs
RiskLattice heuristic" comparison, and a ground-truth evaluation of the
recommended strategies.

Ground-truth labels are used only in the dedicated evaluation section, never
to choose actions.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Allow running directly: python engine/containment/experiment.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json

import pandas as pd

from engine.containment.optimizer import ContainmentOptimizer
from engine.containment.simulation import (
    ContainmentIndex,
    evaluate_strategy_ground_truth,
)
from engine.graph.campaign_detector import (
    find_campaign_candidates,
    load_phase2_risk_scores,
)
from engine.graph.graph_builder import build_graph
from engine.risk.risk_engine import TxIndex, assess_all, deduplicate_candidates

CSV = _PROJECT_ROOT / "data" / "samples" / "transactions.csv"
OUT = _PROJECT_ROOT / "ml" / "artifacts" / "containment_experiment.json"


def _block_everything_strategy(assessment, index, campaign_tx):
    """Baseline: block every user in the campaign."""
    affected: set[str] = set()
    for uid in assessment.user_ids:
        affected |= set(index.lookup("USER", uid))
    suspicious = {t for t in affected if index.is_suspicious(t)}
    legit_tx = affected - suspicious
    contain = len(suspicious & campaign_tx) / max(
        len({t for t in campaign_tx if index.is_suspicious(t)}), 1)
    legit_users = set()
    for t in legit_tx:
        u = index.transaction_entities.get(t, {}).get("USER")
        if u:
            legit_users.add(u)
    return {
        "strategy": "BLOCK_ALL_USERS",
        "fraud_containment_rate": round(contain, 4),
        "legitimate_users_affected": len(legit_users),
        "legitimate_transactions_affected": len(legit_tx),
        "action_count": len(assessment.user_ids),
        "collateral": ("HIGH" if len(legit_users) >= 5
                       else "MEDIUM" if len(legit_users) >= 2 else "LOW"),
    }


def main() -> dict:
    df = pd.read_csv(CSV, parse_dates=["timestamp"])
    graph = build_graph(df)
    risk_scores = load_phase2_risk_scores(str(CSV))
    candidates = find_campaign_candidates(graph, risk_scores=risk_scores)
    assessments = assess_all(
        deduplicate_candidates(candidates), graph, TxIndex(df), risk_scores
    )

    gt = df.set_index("transaction_id")["is_fraud"].astype(int).to_dict()
    index = ContainmentIndex(df, risk_scores,
                             gt_is_fraud={k: bool(v) for k, v in gt.items()})
    optimizer = ContainmentOptimizer(df, index, risk_scores)

    recommendations = []
    no_safe = 0
    strategy_counter = Counter()
    contain_rates = []
    legit_impact = []
    collateral_scores = []

    block_all_contain: list[float] = []
    block_all_legit: list[float] = []
    heur_contain: list[float] = []
    heur_legit: list[float] = []

    for assessment in assessments:
        campaign_tx = set(assessment.transaction_ids)
        rec = optimizer.recommend(assessment)
        recommendations.append({
            "campaign_id": assessment.campaign_id,
            "recommendation": rec["recommendation"],
            "strategy": (rec["recommended_strategy"]
                         if rec.get("recommended_strategy") else None),
            "risk_score": assessment.risk_score,
        })

        if rec["recommendation"] == "NO_SAFE_ACTION":
            no_safe += 1
            continue

        s = rec["recommended_strategy"]
        first_type = s["action_types"][0] if s["action_types"] else "NO_ACTION"
        strategy_counter[first_type] += 1
        contain_rates.append(s["fraud_containment_rate"])
        legit_impact.append(s["legitimate_users_affected"])
        collateral_scores.append(s["collateral_risk"])

        be = _block_everything_strategy(assessment, index, campaign_tx)
        block_all_contain.append(be["fraud_containment_rate"])
        block_all_legit.append(be["legitimate_users_affected"])
        heur_contain.append(s["fraud_containment_rate"])
        heur_legit.append(s["legitimate_users_affected"])

        # Ground-truth evaluation of the recommended strategy (only here).
        affected: set[str] = set()
        for aid in s["action_ids"]:
            action = next(a for a in optimizer._generate_actions(assessment)
                          if a.action_id == aid)
            affected |= set(index.lookup(action.target_type.value,
                                         action.target_id))
        rec["ground_truth"] = evaluate_strategy_ground_truth(
            affected, campaign_tx, index,
        )

    avg = lambda xs, default=0.0: (sum(xs) / len(xs)) if xs else default
    comparison = {
        "block_everything": {
            "average_fraud_containment": round(avg(block_all_contain), 4),
            "average_legitimate_users_affected": round(avg(block_all_legit), 2),
        },
        "risklattice_heuristic": {
            "average_fraud_containment": round(avg(heur_contain), 4),
            "average_legitimate_users_affected": round(avg(heur_legit), 2),
        },
    }

    report = {
        "number_of_campaigns": len(assessments),
        "recommendations_generated": len(assessments) - no_safe,
        "no_safe_action_count": no_safe,
        "strategy_distribution": dict(strategy_counter),
        "average_fraud_containment": round(avg(contain_rates), 4),
        "average_legitimate_impact": round(avg(legit_impact), 2),
        "average_collateral_risk": round(avg(collateral_scores), 4),
        "comparison": comparison,
        "campaign_details": recommendations[:10],
    }
    print(json.dumps(report, indent=2, default=str))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()