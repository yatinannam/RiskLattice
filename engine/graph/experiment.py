"""Phase 3 graph experiment — statistics, high-connectivity entities, and
evidence examples (legitimate shared infrastructure + ground-truth validation).

Run:  python engine/graph/experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly:  python engine/graph/experiment.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from engine.graph import graph_builder as gb
from engine.graph import graph_features as gf

CSV = _PROJECT_ROOT / "data" / "samples" / "transactions.csv"


def top_entities(mapping: dict[str, int], k: int = 8) -> list[tuple[str, int]]:
    return sorted(mapping.items(), key=lambda kv: -kv[1])[:k]


def pick_legit_shared_example(raw: pd.DataFrame, g) -> dict:
    """Pick a LEGITIMATE transaction on a high-sharing IP (shared infra)."""
    users_per_ip = gf.users_per_ip(g)
    # Find a shared IP with several users that appears only among legit rows.
    legit_df = raw[raw["is_fraud"] == 0]
    ip_counts = legit_df.groupby("ip_id")["user_id"].nunique()
    candidates = [(ip, n) for ip, n in ip_counts.items() if n >= 3]
    if not candidates:
        return {"note": "no legit shared-IP example found"}
    ip, _ = candidates[0]
    tx_id = legit_df[legit_df["ip_id"] == ip]["transaction_id"].iloc[0]
    return gf.extract_graph_evidence(g, tx_id)


def pick_fraud_campaign_example(raw: pd.DataFrame, g) -> dict:
    """Pick a ground-truth FARM transaction (validation only)."""
    fraud = raw[raw["is_fraud"] == 1]
    farm = fraud[fraud["scenario"] == "account_farm"]
    tx_id = farm["transaction_id"].iloc[0]
    evidence = gf.extract_graph_evidence(g, tx_id)
    evidence["ground_truth_validation"] = {
        "is_fraud": True,
        "scenario": farm["scenario"].iloc[0],
        "fraud_campaign_id": farm["fraud_campaign_id"].iloc[0],
    }
    return evidence


def main() -> None:
    raw = pd.read_csv(CSV, parse_dates=["timestamp"])
    print("Building graph over", len(raw), "transactions...")
    g = gb.build_graph(raw)

    summary = gb.graph_summary(g)
    print("=" * 70)
    print("GRAPH SUMMARY")
    print("=" * 70)
    print(f"nodes: {summary['node_count']}")
    print(f"edges: {summary['edge_count']}")
    print(f"nodes_by_type: {summary['nodes_by_type']}")
    print(f"edges_by_type: {summary['edges_by_type']}")
    print(f"connected_components: {summary['connected_components']}")
    print(f"largest_component_size: {summary['largest_component_size']}")
    print(f"average_degree: {summary['average_degree']}")

    print("\n" + "=" * 70)
    print("HIGH-CONNECTIVITY ENTITIES (evidence, NOT fraud verdicts)")
    print("=" * 70)
    print("devices:", top_entities(gf.device_degree(g)))
    print("ips:", top_entities(gf.ip_degree(g)))
    print("payment instruments:", top_entities(gf.payment_instrument_degree(g)))
    print("users per device:", top_entities(gf.users_per_device(g)))
    print("users per ip:", top_entities(gf.users_per_ip(g)))
    print("users per payment instrument:", top_entities(gf.users_per_payment_instrument(g)))

    print("\n" + "=" * 70)
    print("EXAMPLE - LEGITIMATE SHARED-INFRASTRUCTURE EVIDENCE")
    print("=" * 70)
    legit_ev = pick_legit_shared_example(raw, g)
    print(legit_ev)

    print("\n" + "=" * 70)
    print("EXAMPLE - GROUND-TRUTH FRAUD CAMPAIGN (VALIDATION ONLY)")
    print("=" * 70)
    fraud_ev = pick_fraud_campaign_example(raw, g)
    print(fraud_ev)


if __name__ == "__main__":
    main()