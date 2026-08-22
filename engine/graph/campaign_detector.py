"""Campaign candidate detection (Phase 3 prelude — NOT the final verdict).

A candidate campaign is a group of *suspicious* transactions connected through
shared entities and temporal proximity. This module may use the Phase-2
transaction-level risk score as an input signal. It NEVER uses the ground-truth
labels (is_fraud / fraud_campaign_id / scenario) to find candidates; those
remain evaluation-only.

The grouping is a documented heuristic (not claimed optimal):
  1. Select suspicious transactions by risk score (or structural fallback).
  2. Connect suspicious transactions that share an entity AND occur within a
     temporal window of each other.
  3. Each connected component becomes one candidate campaign.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx

# Allow running directly:  python engine/graph/campaign_detector.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine.graph.graph_builder import build_graph
from engine.graph.graph_features import (
    transaction_entities,
    transaction_timestamps,
    relationship_density,
)

DEFAULT_TEMPORAL_WINDOW = 60 * 60  # 1 hour
DEFAULT_RISK_THRESHOLD = 0.5


def load_phase2_risk_scores(csv_path: str | Path = "data/samples/transactions.csv",
                            model_name: str = "logistic_regression") -> dict[str, float]:
    """Return {transaction_id: fraud probability} from the Phase-2 artifact.

    Uses only the trained Phase-2 model (no retraining) and the past-only
    feature pipeline. If the artifact is missing, returns {} and callers fall
    back to a structural heuristic.
    """
    from joblib import load

    model_path = _PROJECT_ROOT / "ml" / "artifacts" / f"{model_name}.joblib"
    if not model_path.exists():
        return {}

    from ml.features.build_features import build_features

    x, _y, meta = build_features(str(csv_path))
    pipeline = load(model_path)
    proba = pipeline.predict_proba(x)[:, 1]
    return {tx_id: float(p) for tx_id, p in zip(meta["transaction_id"], proba)}


def _structural_risk(graph: nx.Graph, risk_scores: dict[str, float],
                     fallback_threshold: int = 8) -> dict[str, float]:
    """Structural fallback risk when no model scores exist.

    A transaction is considered structurally suspicious if one of its shared
    entities (device/IP/payment instrument) is used by many *distinct users*.
    This is a heuristic and, by itself, never a fraud verdict.
    """
    if risk_scores:
        return risk_scores
    from engine.graph.graph_features import users_per_device, users_per_ip, users_per_payment_instrument

    high_devs = {d for d, n in users_per_device(graph).items() if n >= fallback_threshold}
    high_ips = {ip for ip, n in users_per_ip(graph).items() if n >= fallback_threshold}
    high_pis = {pi for pi, n in users_per_payment_instrument(graph).items() if n >= fallback_threshold}

    scores: dict[str, float] = {}
    for tx_id in transaction_timestamps(graph):
        ents = transaction_entities(graph, tx_id)
        suspicious = (
            ents["device_id"] in high_devs
            or ents["ip_id"] in high_ips
            or ents["payment_instrument_id"] in high_pis
        )
        scores[tx_id] = 1.0 if suspicious else 0.0
    return scores


def _suspicious_transactions(graph, risk_scores, risk_threshold):
    timestamps = transaction_timestamps(graph)  # computed once (fast lookup)
    return sorted(
        (tx for tx, r in risk_scores.items() if r >= risk_threshold),
        key=lambda t: (timestamps.get(t), t),
    )


def _build_candidate_graph(graph, suspicious, temporal_window, entity_keys) -> nx.Graph:
    """Compose candidate groups: suspicious txs sharing an entity within window."""
    timestamps = transaction_timestamps(graph)
    cand = nx.Graph()
    for tx in suspicious:
        cand.add_node(tx)

    bucket: dict[tuple, list[tuple[str, datetime]]] = defaultdict(list)
    for tx in suspicious:
        ents = transaction_entities(graph, tx)
        ts = timestamps[tx]
        for key in entity_keys:
            eid = ents.get(key)
            if eid:
                bucket[(key, eid)].append((tx, ts))

    added: set[tuple] = set()
    for group in bucket.values():
        group.sort(key=lambda kv: kv[1])
        for i in range(len(group)):
            tx_i, ts_i = group[i]
            for j in range(i + 1, len(group)):
                tx_j, ts_j = group[j]
                if (ts_j - ts_i).total_seconds() > temporal_window:
                    break
                edge = tuple(sorted((tx_i, tx_j)))
                if edge not in added:
                    cand.add_edge(tx_i, tx_j)
                    added.add(edge)
    return cand


def _build_candidate(cand_graph_comp, graph, risk_scores, temporal_window) -> dict:
    """Build a candidate dict from an induced subgraph of suspicious txs."""
    timestamps = transaction_timestamps(graph)
    txs = sorted(cand_graph_comp.nodes())
    member_edges = [(u, v) for u, v in cand_graph_comp.edges()]
    ts_list = [timestamps[t] for t in txs]
    start = min(ts_list)
    end = max(ts_list)

    user_ids: set[str] = set()
    device_ids: set[str] = set()
    ip_ids: set[str] = set()
    pi_ids: set[str] = set()
    for tx in txs:
        ents = transaction_entities(graph, tx)
        if ents["user_id"]:
            user_ids.add(ents["user_id"])
        if ents["device_id"]:
            device_ids.add(ents["device_id"])
        if ents["ip_id"]:
            ip_ids.add(ents["ip_id"])
        if ents["payment_instrument_id"]:
            pi_ids.add(ents["payment_instrument_id"])

    member_edges = [(u, v) for u, v in cand_graph_comp.edges() if u in txs and v in txs]
    risk_vals = [risk_scores.get(t, 0.0) for t in txs]

    return {
        "transaction_ids": txs,
        "user_ids": sorted(user_ids),
        "device_ids": sorted(device_ids),
        "ip_ids": sorted(ip_ids),
        "payment_instrument_ids": sorted(pi_ids),
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_seconds": int((end - start).total_seconds()),
        "transaction_count": len(txs),
        "entity_count": len(user_ids) + len(device_ids) + len(ip_ids) + len(pi_ids),
        "relationship_density": round(relationship_density(txs, member_edges), 4),
        "edge_count": len(member_edges),
        "risk_score_summary": {
            "mean": round(float(sum(risk_vals) / len(risk_vals)), 4) if risk_vals else 0.0,
            "max": round(float(max(risk_vals)), 4) if risk_vals else 0.0,
        },
        "evidence": {
            "shared_device_count": len(device_ids),
            "shared_ip_count": len(ip_ids),
            "shared_payment_instrument_count": len(pi_ids),
            "temporal_window_seconds": temporal_window,
        },
    }


def find_campaign_candidates(
    graph: nx.Graph,
    risk_scores: dict[str, float] | None = None,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    temporal_window: int = DEFAULT_TEMPORAL_WINDOW,
    min_transactions: int = 3,
    entity_keys=("device_id", "ip_id", "payment_instrument_id"),
) -> list[dict]:
    """Return candidate campaign structures.

    ``risk_scores`` maps transaction_id -> fraud probability. Ground-truth
    fields are never used here.
    """
    if risk_scores is None:
        risk_scores = {}

    effective_scores = _structural_risk(graph, risk_scores)
    suspicious = _suspicious_transactions(graph, effective_scores, risk_threshold)

    cand_graph = _build_candidate_graph(graph, suspicious, temporal_window, entity_keys)
    candidates: list[dict] = []
    for comp in nx.connected_components(cand_graph):
        if len(comp) < min_transactions:
            continue
        comp_sub = cand_graph.subgraph(comp)
        candidate = _build_candidate(comp_sub, graph, effective_scores, temporal_window)
        candidate["campaign_candidate_id"] = f"CAMP_{len(candidates) + 1:04d}"
        candidate["evidence"]["member_edges"] = len(list(comp_sub.edges()))
        candidates.append(candidate)

    candidates.sort(key=lambda c: -c["risk_score_summary"]["max"])
    for idx, candidate in enumerate(candidates, start=1):
        candidate["campaign_candidate_id"] = f"CAMP_{idx:04d}"
    return candidates


if __name__ == "__main__":
    import json
    import pandas as pd

    df = pd.read_csv(
        _PROJECT_ROOT / "data" / "samples" / "transactions.csv",
        parse_dates=["timestamp"],
        nrows=10000,
    )
    g = build_graph(df)
    scores = load_phase2_risk_scores()
    if not scores:
        print("No Phase-2 artifact found; using structural fallback.")
    candidates = find_campaign_candidates(g, risk_scores=scores)
    print(f"Candidate campaigns: {len(candidates)}")
    for c in candidates[:3]:
        print(json.dumps(c, indent=2, default=str)[:1200])