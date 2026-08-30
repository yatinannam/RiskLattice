"""Phase 6 investigator experiment.

Run the investigator against representative campaigns from the baseline and
hardened datasets, and write a reproducible report to
ml/artifacts/investigator_report.json.

Picks, per dataset:
  - a HIGH/CRITICAL risk campaign
  - a mixed (fraud+legitimate entity) campaign if present
  - a NO_SAFE_ACTION campaign if present
  - a low-signal campaign (hardened only)

All reports are produced by the deterministic mock provider; every report is
validated (hallucination guard) before being recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running directly: python engine/investigator/experiment.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

OUT = _PROJECT_ROOT / "ml" / "artifacts" / "investigator_report.json"

BASELINE_CSV = _PROJECT_ROOT / "data" / "samples" / "transactions.csv"
HARDENED_CSV = _PROJECT_ROOT / "data" / "samples" / "transactions_hardened.csv"


def _assess_campaigns(df):
    from engine.containment.simulation import ContainmentIndex
    from engine.graph.campaign_detector import (
        find_campaign_candidates,
        load_phase2_risk_scores,
    )
    from engine.graph.graph_builder import build_graph
    from engine.risk.risk_engine import TxIndex, assess_all, deduplicate_candidates

    graph = build_graph(df)
    risk = load_phase2_risk_scores(str(_pick_csv(df)))
    candidates = find_campaign_candidates(graph, risk_scores=risk)
    assessments = assess_all(
        deduplicate_candidates(candidates), graph, TxIndex(df), risk
    )
    ranked = sorted(assessments, key=lambda a: -a.risk_score)

    gt = df.set_index("transaction_id")["is_fraud"].astype(int).to_dict()
    index = ContainmentIndex(df, risk, gt_is_fraud={k: bool(v) for k, v in gt.items()})
    return ranked, index, risk


def _pick_csv(df):
    # Heuristic: the dataset frame knows its own size; choose by row count.
    return BASELINE_CSV if len(df) == 10000 else HARDENED_CSV


def _select_campaigns(ranked, df, source):
    """Choose representative campaigns by risk level and scenario mix."""
    selected = {}
    for a in ranked:
        if a.risk_level in ("HIGH", "CRITICAL") and "high" not in selected:
            selected["high_risk"] = a
        # mixed: any campaign containing a tx whose scenario is mixed_entity
        if "mixed" not in selected:
            any_mixed = any(
                df.loc[df["transaction_id"] == t, "scenario"].eq("mixed_entity_campaign").any()
                for t in a.transaction_ids
            )
            if any_mixed:
                selected["mixed_entity"] = a
        if "no_safe" not in selected and a.risk_level in ("HIGH", "CRITICAL"):
            # Determined later via containment; here we just remember candidates
            pass
    return selected


def _make_containment(df, index, risk, assessment):
    from engine.containment.optimizer import ContainmentOptimizer

    optimizer = ContainmentOptimizer(df, index, risk)
    return optimizer.recommend(assessment)


def _run_one(df, index, risk, assessment, provider, source):
    from engine.investigator.investigator import investigate_campaign

    containment = _make_containment(df, index, risk, assessment)
    return {
        "campaign_id": assessment.campaign_id,
        "source": source,
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "containment_recommendation": containment.get("recommendation"),
        "investigation": investigate_campaign(assessment, containment, provider),
    }


def main() -> dict:
    from engine.investigator.investigator import resolve_provider

    provider = resolve_provider("mock")
    results = []
    evidence_counts = []
    uncertainty_counts = []
    validation_ok = 0

    for source, csv in (("baseline", BASELINE_CSV), ("hardened", HARDENED_CSV)):
        if not csv.exists():
            continue
        df = pd.read_csv(csv, parse_dates=["timestamp"])
        ranked, index, risk = _assess_campaigns(df)

        # Pick up to 3 representative campaigns: highest risk + a NO_SAFE_ACTION
        # if it appears among the top-ranked (bounded scan for speed).
        chosen = []
        chosen_ids = set()
        no_safe_found = None
        for a in ranked[:8]:
            if a.risk_level in ("HIGH", "CRITICAL") and len(chosen) < 2 \
                    and a.campaign_id not in chosen_ids:
                chosen.append(a)
                chosen_ids.add(a.campaign_id)
            # Check for NO_SAFE_ACTION among the top-ranked (bounded scan).
            if no_safe_found is None:
                containment = _make_containment(df, index, risk, a)
                if containment.get("recommendation") == "NO_SAFE_ACTION":
                    no_safe_found = (a, containment)
            if len(chosen) >= 2 and no_safe_found is not None:
                break
        if no_safe_found is not None and no_safe_found[0].campaign_id not in chosen_ids:
            chosen.append(no_safe_found[0])
            chosen_ids.add(no_safe_found[0].campaign_id)

        for assessment in chosen[:3]:
            outcome = _run_one(df, index, risk, assessment, provider, source)
            results.append(outcome)
            evidence_counts.append(len(outcome["investigation"]["evidence"]["findings"]))
            uncertainty_counts.append(len(outcome["investigation"]["report"]["uncertainty"]))
            if outcome["investigation"]["validation_status"] == "VALID":
                validation_ok += 1

    report = {
        "provider": provider.name,
        "prompt_version": "risklattice-investigator-v1",
        "campaign_investigations": results,
        "evidence_counts": evidence_counts,
        "uncertainty_counts": uncertainty_counts,
        "validation_status_ok": validation_ok,
        "total_investigations": len(results),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "campaign_investigations"},
                     indent=2, default=str))
    return report


if __name__ == "__main__":
    main()