"""Deterministic, typed relationship graph construction.

The graph models entities and the relationships observed in the transaction
stream. Node identifiers are synthetic (USER_*/DEV_*/IP_*/PI_*/TXN_*/MERCH_*).
Repeated relationships between the same pair are aggregated into a single edge
carrying temporal metadata (first_seen, last_seen, transaction_count).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path

import networkx as nx
import pandas as pd

# Allow running directly:  python engine/graph/graph_builder.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class NodeType(str, Enum):
    """Node kinds present in the RiskLattice relationship graph."""

    USER = "USER"
    DEVICE = "DEVICE"
    IP = "IP"
    PAYMENT_INSTRUMENT = "PAYMENT_INSTRUMENT"
    TRANSACTION = "TRANSACTION"
    MERCHANT = "MERCHANT"


class RelationshipType(str, Enum):
    """Typed edges. The left side is the semantic source."""

    USES_DEVICE = "USES_DEVICE"                                  # USER -> DEVICE
    CONNECTS_FROM_IP = "CONNECTS_FROM_IP"                        # USER -> IP
    USES_PAYMENT_INSTRUMENT = "USES_PAYMENT_INSTRUMENT"          # USER -> PAYMENT_INSTRUMENT
    PERFORMED = "PERFORMED"                                      # USER -> TRANSACTION
    AT_MERCHANT = "AT_MERCHANT"                                  # TRANSACTION -> MERCHANT


# Canonical (source_type, target_type, relationship_type) tuples — the single
# source of truth for which edges are legal to create.
EDGE_SCHEMAS = {
    (NodeType.USER, NodeType.DEVICE): RelationshipType.USES_DEVICE,
    (NodeType.USER, NodeType.IP): RelationshipType.CONNECTS_FROM_IP,
    (NodeType.USER, NodeType.PAYMENT_INSTRUMENT): RelationshipType.USES_PAYMENT_INSTRUMENT,
    (NodeType.USER, NodeType.TRANSACTION): RelationshipType.PERFORMED,
    (NodeType.TRANSACTION, NodeType.MERCHANT): RelationshipType.AT_MERCHANT,
}

# Attribute names attached to every edge.
EDGE_ATTRS = ("relationship_type", "first_seen", "last_seen", "transaction_count")


def _add_unique_node(graph, node_id: str, node_type: NodeType, **attrs) -> None:
    """Add a node once; fill extra attributes only on first creation."""
    if not graph.has_node(node_id):
        attrs["node_id"] = node_id
        attrs["node_type"] = node_type.value
        graph.add_node(node_id, **attrs)


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def add_user_node(graph, user_id: str) -> None:
    _add_unique_node(graph, user_id, NodeType.USER)


def add_device_node(graph, device_id: str) -> None:
    _add_unique_node(graph, device_id, NodeType.DEVICE)


def add_ip_node(graph, ip_id: str) -> None:
    _add_unique_node(graph, ip_id, NodeType.IP)


def add_payment_instrument_node(graph, payment_instrument_id: str) -> None:
    _add_unique_node(graph, payment_instrument_id, NodeType.PAYMENT_INSTRUMENT)


def add_merchant_node(graph, merchant_id: str) -> None:
    _add_unique_node(graph, merchant_id, NodeType.MERCHANT)


def add_transaction_node(graph, transaction_id: str, timestamp, amount, status) -> None:
    _add_unique_node(
        graph,
        transaction_id,
        NodeType.TRANSACTION,
        timestamp=timestamp,
        amount=float(amount),
        status=str(status),
    )


# ---------------------------------------------------------------------------
# Relationship helper
# ---------------------------------------------------------------------------

def add_relationship(graph, source, target, relationship: RelationshipType, timestamp) -> None:
    """Add or aggregate a typed edge with temporal metadata.

    Repeated relationships between the same (source, target, relationship)
    are aggregated: ``first_seen`` tracks the earliest timestamp, ``last_seen``
    the latest, and ``transaction_count`` increments.
    """
    attrs = {
        "relationship_type": relationship.value,
        "first_seen": timestamp,
        "last_seen": timestamp,
        "transaction_count": 1,
    }

    if graph.has_edge(source, target):
        existing = graph.edges[source, target]
        existing["transaction_count"] += 1
        existing["last_seen"] = max(existing["last_seen"], timestamp)
        existing["first_seen"] = min(existing["first_seen"], timestamp)
        # Normalize to datetime in case of string input on first creation.
        return graph

    graph.add_edge(source, target, **attrs)
    return graph


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(transactions: pd.DataFrame) -> nx.Graph:
    """Build the full relationship graph from a transaction DataFrame.

    ``transactions`` must have the Phase-1 transaction columns
    (transaction_id, timestamp, merchant_id, user_id, device_id, ip_id,
    payment_instrument_id, amount, status at minimum). The frame is sorted
    deterministically by (timestamp, transaction_id) before construction so
    the result is reproducible regardless of input order.
    """
    graph = nx.Graph()

    df = transactions.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)

    for row in df.itertuples(index=False):
        ts = row.timestamp
        uid = row.user_id
        device = row.device_id
        ip = row.ip_id
        pi = row.payment_instrument_id
        txid = row.transaction_id
        merchant = row.merchant_id
        amount = float(row.amount)
        status = str(row.status)

        add_user_node(graph, uid)
        add_device_node(graph, device)
        add_ip_node(graph, ip)
        add_payment_instrument_node(graph, pi)
        add_transaction_node(graph, txid, ts, amount, status)
        add_merchant_node(graph, merchant)

        add_relationship(graph, txid, uid, RelationshipType.PERFORMED, ts)
        add_relationship(graph, uid, device, RelationshipType.USES_DEVICE, ts)
        add_relationship(graph, uid, ip, RelationshipType.CONNECTS_FROM_IP, ts)
        add_relationship(graph, uid, pi, RelationshipType.USES_PAYMENT_INSTRUMENT, ts)
        add_relationship(graph, txid, merchant, RelationshipType.AT_MERCHANT, ts)

    return graph


# ---------------------------------------------------------------------------
# Graph summary & serialization
# ---------------------------------------------------------------------------

def nodes_by_type(graph: nx.Graph) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for _node, attrs in graph.nodes(data=True):
        counts[attrs.get("node_type", "UNKNOWN")] += 1
    return dict(sorted(counts.items()))


def edges_by_type(graph: nx.Graph) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for _u, _v, attrs in graph.edges(data=True):
        counts[attrs.get("relationship_type", "UNKNOWN")] += 1
    return dict(sorted(counts.items()))


def graph_summary(graph: nx.Graph) -> dict:
    """Return a structural summary of the graph.

    Deliberately excludes ground-truth fraud labels: the summary is about the
    relationship structure only.
    """
    components = list(nx.connected_components(graph))
    largest_size = max((len(c) for c in components), default=0)
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "nodes_by_type": nodes_by_type(graph),
        "edges_by_type": edges_by_type(graph),
        "connected_components": len(components),
        "largest_component_size": largest_size,
        "average_degree": round(
            (2.0 * graph.number_of_edges()) / graph.number_of_nodes()
            if graph.number_of_nodes() else 0.0,
            3,
        ),
    }


def serialize_graph(graph: nx.Graph) -> dict:
    """Return a JSON-friendly representation for later frontend consumption.

    This is backend graph data only — no UI is built in this phase.
    """
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        node = {"id": node_id, "type": attrs.get("node_type")}
        # Carry lightweight, non-sensitive attributes for transaction nodes.
        for key in ("timestamp", "amount", "status"):
            if key in attrs:
                node[key] = attrs[key].isoformat() if key == "timestamp" else attrs[key]
        nodes.append(node)

    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "type": attrs.get("relationship_type"),
            "weight": attrs.get("transaction_count", 1),
            "first_seen": attrs.get("first_seen").isoformat() if attrs.get("first_seen") else None,
            "last_seen": attrs.get("last_seen").isoformat() if attrs.get("last_seen") else None,
        })

    return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    import json

    df = pd.read_csv(
        _PROJECT_ROOT / "data" / "samples" / "transactions.csv",
        parse_dates=["timestamp"],
        nrows=10000,
    )
    g = build_graph(df)
    print("Graph:", g.number_of_nodes(), "nodes,", g.number_of_edges(), "edges")
    print(json.dumps(nodes_by_type(g), indent=2))
    print(json.dumps(edges_by_type(g), indent=2))