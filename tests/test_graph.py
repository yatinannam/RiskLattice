"""Tests for the RiskLattice graph engine (Phase 3).

Covers graph construction, typed nodes/edges, temporal aggregation, the
"shared IP / shared device != fraud" rule, subgraph extraction, evidence, the
no-future temporal rule, deterministic construction, and the ground-truth-free
campaign-candidate rule.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from engine.graph import graph_builder as gb
from engine.graph import graph_features as gf
from engine.graph.campaign_detector import (
    find_campaign_candidates,
    load_phase2_risk_scores,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "samples" / "transactions.csv"


@pytest.fixture(scope="module")
def transactions():
    return pd.read_csv(DEFAULT_CSV, parse_dates=["timestamp"])


@pytest.fixture(scope="module")
def graph(transactions):
    return gb.build_graph(transactions)


def _row(txid, ts, user, device, ip, pi, amount=100.0, status="success"):
    return {
        "transaction_id": txid,
        "timestamp": ts,
        "merchant_id": "MERCH_001",
        "user_id": user,
        "device_id": device,
        "ip_id": ip,
        "payment_instrument_id": pi,
        "amount": amount,
        "status": status,
    }


def _any_transaction_id(graph):
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == "TRANSACTION":
            return node
    raise AssertionError("no transaction node")


# ---------------------------------------------------------------------------
# Construction & structure
# ---------------------------------------------------------------------------

def test_graph_construction_succeeds(graph):
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0


def test_expected_node_types_exist(graph):
    node_types = {attrs.get("node_type") for _, attrs in graph.nodes(data=True)}
    assert {"USER", "DEVICE", "IP", "PAYMENT_INSTRUMENT", "TRANSACTION", "MERCHANT"} <= node_types


def test_expected_edge_types_exist(graph):
    edge_types = {attrs.get("relationship_type") for _, _, attrs in graph.edges(data=True)}
    expected = {
        "USES_DEVICE",
        "CONNECTS_FROM_IP",
        "USES_PAYMENT_INSTRUMENT",
        "PERFORMED",
        "AT_MERCHANT",
    }
    assert expected <= edge_types


def test_repeated_relationships_aggregate():
    """Two separate user-device relationships stay distinct edges, while two
    transactions from the same user to the same device collapse into one edge
    with transaction_count==2."""
    rows = [
        _row("1", datetime(2026, 8, 1, 10, 0), "U1", "D1", "IP1", "P1"),
        _row("2", datetime(2026, 8, 1, 11, 0), "U1", "D1", "IP1", "P1"),
    ]
    g = gb.build_graph(pd.DataFrame(rows))
    rel = [a for u, v, a in g.edges(data=True) if a["relationship_type"] == "USES_DEVICE"]
    assert len(rel) == 1
    assert rel[0]["transaction_count"] == 2


def test_temporal_metadata_is_correct():
    rows = [
        _row("1", datetime(2026, 8, 1, 10, 0), "U1", "D1", "IP1", "P1"),
        _row("2", datetime(2026, 8, 3, 14, 30), "U1", "D1", "IP2", "P2"),
    ]
    g = gb.build_graph(pd.DataFrame(rows))
    rel = [a for u, v, a in g.edges(data=True) if a["relationship_type"] == "USES_DEVICE"]
    assert rel[0]["first_seen"] == datetime(2026, 8, 1, 10, 0)
    assert rel[0]["last_seen"] == datetime(2026, 8, 3, 14, 30)


# ---------------------------------------------------------------------------
# Shared infrastructure == evidence, NOT fraud
# ---------------------------------------------------------------------------

def test_shared_ip_creates_relationships_but_not_fraud_label():
    rows = [
        _row(f"t{i}", datetime(2026, 8, 1, 8, 0) + pd.to_timedelta(i, unit="m"),
             f"U{i}", f"D{i}", "IP_SHARED", f"P{i}")
        for i in range(1, 101)
    ]
    g = gb.build_graph(pd.DataFrame(rows))
    assert gf.users_per_ip(g).get("IP_SHARED", 0) >= 100
    for _, attrs in g.nodes(data=True):
        assert "is_fraud" not in attrs


def test_shared_device_creates_relationships_but_not_fraud_label():
    rows = [
        _row(f"t{i}", datetime(2026, 8, 1, 8, 0) + pd.to_timedelta(i, unit="m"),
             f"U{i}", "DEV_SHARED", f"IP{i}", f"P{i}")
        for i in range(1, 101)
    ]
    g = gb.build_graph(pd.DataFrame(rows))
    assert gf.users_per_device(g).get("DEV_SHARED", 0) >= 100
    for _, attrs in g.nodes(data=True):
        assert "is_fraud" not in attrs


# ---------------------------------------------------------------------------
# Temporal no-future rule
# ---------------------------------------------------------------------------

def test_future_transactions_not_included_in_temporal_evidence():
    rows = [
        _row("A", datetime(2026, 8, 1, 9, 0), "U1", "D1", "IP1", "P1"),
        _row("B", datetime(2026, 8, 1, 10, 30), "U1", "D1", "IP2", "P2"),
    ]
    g = gb.build_graph(pd.DataFrame(rows))
    # A's related transactions within 1h ending at 09:30 (lookback only) must
    # NOT include B (which is at 10:30, i.e. in the future relative to 09:30).
    window = gf.transactions_in_window(g, "A", datetime(2026, 8, 1, 9, 30), 3600)
    assert "B" not in window


# ---------------------------------------------------------------------------
# Subgraph & evidence
# ---------------------------------------------------------------------------

def test_get_transaction_subgraph_works(graph):
    tx_id = _any_transaction_id(graph)
    sub = gf.get_transaction_subgraph(graph, tx_id, max_hops=2)
    assert isinstance(sub, nx.Graph)
    assert tx_id in sub
    assert any(attrs.get("node_type") == "USER" for _, attrs in sub.nodes(data=True))


def test_graph_evidence_contains_expected_relationships(graph):
    tx_id = _any_transaction_id(graph)
    evidence = gf.extract_graph_evidence(graph, tx_id)
    assert "shared_device_users" in evidence
    assert "shared_ip_users" in evidence
    assert "shared_payment_users" in evidence
    assert "temporal_density" in evidence
    assert evidence["transaction_id"] == tx_id


# ---------------------------------------------------------------------------
# Determinism & ground-truth separation
# ---------------------------------------------------------------------------

def test_graph_construction_is_deterministic(transactions):
    g1 = gb.build_graph(transactions)
    g2 = gb.build_graph(transactions.sample(frac=1.0, random_state=7))  # shuffled input
    assert sorted(g1.nodes) == sorted(g2.nodes)
    assert sorted(sorted(e) for e in g1.edges) == sorted(sorted(e) for e in g2.edges)


def test_ground_truth_never_used_for_campaign_detection():
    """Campaign candidates must be findable without reading fraud labels."""
    small = pd.DataFrame([
        _row("1", datetime(2026, 8, 1, 10, 0), "U1", "D1", "IP1", "P1"),
        _row("2", datetime(2026, 8, 1, 10, 10), "U2", "D1", "IP1", "P1"),
        _row("3", datetime(2026, 8, 1, 10, 20), "U3", "D1", "IP1", "P1"),
    ])
    g = gb.build_graph(small)
    candidates = find_campaign_candidates(g, risk_scores={})
    assert isinstance(candidates, list)
    for cand in candidates:
        for forbidden in ("is_fraud", "fraud_campaign_id", "scenario"):
            assert forbidden not in cand


def test_full_graph_campaign_candidates_with_phase2_scores():
    """Sanity: Phase-2 risk-driven candidate detection runs on the full graph."""
    scores = load_phase2_risk_scores(str(DEFAULT_CSV))
    assert len(scores) == 10000
    graph = gb.build_graph(pd.read_csv(DEFAULT_CSV, parse_dates=["timestamp"]))
    candidates = find_campaign_candidates(graph, risk_scores=scores)
    assert isinstance(candidates, list)
    assert len(candidates) > 0