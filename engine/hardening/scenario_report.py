"""Per-scenario report for the Phase 5.5 hardening run.

Produces a table like:
  Scenario | Count | Fraud Tx | Baseline Recall | Campaign-flagged | Legit Impact

Ground-truth labels (is_fraud / scenario) are used ONLY here to summarize how
each synthetic scenario behaves — never in detection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def high_risk_campaign_tx(df, baseline) -> set[str]:
    """Evaluate the lattice and return transaction IDs inside HIGH/CRITICAL
    campaigns. Deterministic; ground truth never read."""
    from engine.graph.campaign_detector import find_campaign_candidates
    from engine.graph.graph_builder import build_graph
    from engine.risk.risk_engine import TxIndex, assess_all

    risk = baseline["full_risk_probabilities"]
    graph = build_graph(df)
    candidates = find_campaign_candidates(graph, risk_scores=risk)
    assessments = assess_all(candidates, graph, TxIndex(df), risk)
    return {
        t for a in assessments if a.risk_level in ("HIGH", "CRITICAL")
        for t in a.transaction_ids
    }


def build_scenario_table(df, baseline, hi_tx: set[str]) -> list[dict]:
    """Per-scenario metrics: count, baseline recall, campaign coverage."""
    test_y = baseline["test_y"]
    preds = baseline["test_predictions"]
    is_fraud = df.set_index("transaction_id")["is_fraud"].astype(int).to_dict()

    rows = []
    for scenario in sorted(df["scenario"].unique()):
        scn_ids = list(df[df["scenario"] == scenario]["transaction_id"])
        fraud_test = [t for t in scn_ids if t in test_y and is_fraud.get(t, 0) == 1]
        recalled = [t for t in fraud_test if preds.get(t, 0) == 1]
        in_hi_camp = [t for t in fraud_test if t in hi_tx]

        base_recall = len(recalled) / len(fraud_test) if fraud_test else None
        campaign_cov = len(in_hi_camp) / len(fraud_test) if fraud_test else None

        rows.append({
            "scenario": scenario,
            "count": len(scn_ids),
            "fraud_in_test": len(fraud_test),
            "baseline_recall": round(base_recall, 4) if base_recall is not None else None,
            "campaign_detection_rate": round(campaign_cov, 4) if campaign_cov is not None else None,
            "baseline_false_negatives": len(fraud_test) - len(recalled),
            "false_negatives_in_high_risk_campaign": len([t for t in fraud_test if t in hi_tx and preds.get(t, 0) == 0]),
        })
    return rows


def main() -> None:
    from engine.hardening.experiment import load_hardened, train_hardened_baseline

    df = load_hardened()
    baseline = train_hardened_baseline(df)
    hi_tx = high_risk_campaign_tx(df, baseline)
    rows = build_scenario_table(df, baseline, hi_tx)
    print(json.dumps(rows, indent=2))
    out = _PROJECT_ROOT / "ml" / "artifacts" / "hardening_scenario_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()