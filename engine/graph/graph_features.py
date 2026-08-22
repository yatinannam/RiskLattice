"""Graph-level features and temporal helpers.

All functions here are evidence-oriented: they quantify relationships and
density but never assign a fraud verdict. High degree or shared infrastructure
is treated as *high-connectivity evidence*, not a fraud label.

Temporal helpers respect the no-future rule: window queries only consider
transactions at or before a reference timestamp.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx

# Allow running directly:  python engine/graph/graph_features.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine.graph.graph_builder import NodeType, RelationshipType

WINDOW_5M = 5 * 60
WINDOW_1H = 60 * 60
WINDOW_24H = 24 * 60 * 60


# ---------------------------------------------------------------------------
# Transaction / entity lookup indexes
# ---------------------------------------------------------------------------

def transaction_timestamps(graph: nx.Graph) -> dict[str, datetime]:
    """Map transaction_id -> timestamp for every TRANSACTION node."""
    out: dict[str, datetime] = {}
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") == NodeType.TRANSACTION.value:
            out[node_id] = attrs["timestamp"]
    return out


def _user_transactions_index(graph: nx.Graph) -> dict[str, set[str]]:
    """Map user_id -> set of transaction_ids the user performed."""
    idx: dict[str, set[str]] = {}
    for tx_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != NodeType.TRANSACTION.value:
            continue
        for neighbor in graph.neighbors(tx_id):
            if graph.nodes[neighbor].get("node_type") == NodeType.USER.value:
                idx.setdefault(neighbor, set()).add(tx_id)
    return idx


def users_of(graph: nx.Graph, entity_id: str, entity_type) -> set[str]:
    """Distinct USER nodes related to ``entity_id`` via the typed edge."""
    if isinstance(entity_type, str):
        entity_type = NodeType(entity_type)
    relationship = {
        NodeType.DEVICE: RelationshipType.USES_DEVICE,
        NodeType.IP: RelationshipType.CONNECTS_FROM_IP,
        NodeType.PAYMENT_INSTRUMENT: RelationshipType.USES_PAYMENT_INSTRUMENT,
    }.get(entity_type)

    users: set[str] = set()
    for neighbor in graph.neighbors(entity_id):
        if graph.nodes[neighbor].get("node_type") != NodeType.USER.value:
            continue
        edge = graph.edges[entity_id, neighbor]
        if relationship is not None and edge.get("relationship_type") == relationship.value:
            users.add(neighbor)
    return users


def transactions_of(graph: nx.Graph, entity_id: str, user_tx_index=None) -> set[str]:
    """Distinct TRANSACTION ids related to an entity (through its users)."""
    node_type = graph.nodes[entity_id].get("node_type")

    if node_type == NodeType.TRANSACTION.value:
        return {entity_id}
    if node_type == NodeType.USER.value:
        if user_tx_index is not None:
            return set(user_tx_index.get(entity_id, set()))
        return _user_transactions_index(graph).get(entity_id, set())

    # Device / IP / PaymentInstrument -> union of the users' transactions.
    if user_tx_index is None:
        user_tx_index = _user_transactions_index(graph)
    users = users_of(graph, entity_id, node_type)
    out: set[str] = set()
    for user in users:
        out |= user_tx_index.get(user, set())
    return out


# ---------------------------------------------------------------------------
# Degree / sharing features (no fraud verdict)
# ---------------------------------------------------------------------------

def degree_by_type(graph: nx.Graph, node_type: NodeType) -> dict[str, int]:
    """Return {entity_id: degree} for nodes of ``node_type``."""
    return {
        node: int(graph.degree(node))
        for node, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == node_type.value
    }


def user_degree(graph: nx.Graph) -> dict[str, int]:
    return degree_by_type(graph, NodeType.USER)


def device_degree(graph: nx.Graph) -> dict[str, int]:
    return degree_by_type(graph, NodeType.DEVICE)


def ip_degree(graph: nx.Graph) -> dict[str, int]:
    return degree_by_type(graph, NodeType.IP)


def payment_instrument_degree(graph: nx.Graph) -> dict[str, int]:
    return degree_by_type(graph, NodeType.PAYMENT_INSTRUMENT)


def users_per_device(graph: nx.Graph) -> dict[str, int]:
    return {d: len(users_of(graph, d, NodeType.DEVICE)) for d in degree_by_type(graph, NodeType.DEVICE)}


def users_per_ip(graph: nx.Graph) -> dict[str, int]:
    return {ip: len(users_of(graph, ip, NodeType.IP)) for ip in degree_by_type(graph, NodeType.IP)}


def users_per_payment_instrument(graph: nx.Graph) -> dict[str, int]:
    return {pi: len(users_of(graph, pi, NodeType.PAYMENT_INSTRUMENT)) for pi in degree_by_type(graph, NodeType.PAYMENT_INSTRUMENT)}


def transactions_per_entity(graph: nx.Graph, node_type: NodeType, user_tx_index=None) -> dict[str, int]:
    """Count distinct transactions related to each entity of ``node_type``."""
    if user_tx_index is None:
        user_tx_index = _user_transactions_index(graph)
    return {
        eid: len(transactions_of(graph, eid, user_tx_index))
        for eid in degree_by_type(graph, node_type)
    }


def transactions_per_device(graph: nx.Graph) -> dict[str, int]:
    return transactions_per_entity(graph, NodeType.DEVICE)


def transactions_per_ip(graph: nx.Graph) -> dict[str, int]:
    return transactions_per_entity(graph, NodeType.IP)


def transactions_per_payment_instrument(graph: nx.Graph) -> dict[str, int]:
    return transactions_per_entity(graph, NodeType.PAYMENT_INSTRUMENT)


def component_size_map(graph: nx.Graph) -> dict[str, int]:
    """Map node_id -> size of its connected component."""
    out: dict[str, int] = {}
    for comp in nx.connected_components(graph):
        size = len(comp)
        for node in comp:
            out[node] = size
    return out


def connected_component_size(graph: nx.Graph, node_id: str) -> int:
    return component_size_map(graph).get(node_id, 0)


def relationship_density(nodes: list[str], edges: list[tuple[str, str]]) -> float:
    """Density among an induced subset: edges / possible undirected edges."""
    n = len(nodes)
    if n < 2:
        return 0.0
    possible = n * (n - 1) / 2.0
    return len(edges) / possible if possible > 0 else 0.0


# ---------------------------------------------------------------------------
# Temporal cluster helpers (no-future aware)
# ---------------------------------------------------------------------------

def transactions_in_window(
    graph: nx.Graph,
    entity_id: str,
    reference_timestamp: datetime,
    window_seconds: int,
    lookback_only: bool = True,
    user_tx_index=None,
) -> set[str]:
    """Transactions related to an entity within ``window_seconds`` ending at
    ``reference_timestamp``.

    Only transactions at or before the reference are considered (never future
    information), consistent with the no-future-leakage rule.
    """
    timestamps = transaction_timestamps(graph)
    related = transactions_of(graph, entity_id, user_tx_index)
    cutoff = reference_timestamp - timedelta(seconds=window_seconds)

    result: set[str] = set()
    for tx_id in related:
        ts = timestamps.get(tx_id)
        if ts is None:
            continue
        if lookback_only and ts > reference_timestamp:
            continue
        if ts >= cutoff:
            result.add(tx_id)
    return result


# ---------------------------------------------------------------------------
# Subgraph extraction & evidence
# ---------------------------------------------------------------------------

def get_transaction_subgraph(graph: nx.Graph, transaction_id: str, max_hops: int = 2) -> nx.Graph:
    """Induced subgraph of the k-hop neighborhood around a transaction.

    ``max_hops=0`` returns just the transaction node; ``max_hops=2`` includes
    the connected user, device, IP, payment instrument, merchant, and nearby
    related transactions.
    """
    if transaction_id not in graph:
        raise KeyError(f"Transaction {transaction_id} not in graph")

    if max_hops <= 0:
        return graph.subgraph([transaction_id]).copy()

    nodes = {transaction_id}
    frontier = {transaction_id}
    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for node in frontier:
            next_frontier.update(graph.neighbors(node))
        nodes.update(next_frontier)
        frontier = next_frontier
    return graph.subgraph(nodes).copy()


def _first_neighbor_of_type(graph: nx.Graph, node: str | None, node_type: NodeType) -> str | None:
    if node is None:
        return None
    for neighbor in graph.neighbors(node):
        if graph.nodes[neighbor].get("node_type") == node_type.value:
            return neighbor
    return None


def transaction_entities(graph: nx.Graph, transaction_id: str) -> dict[str, str | None]:
    """Return the related {user, device, ip, payment_instrument, merchant} for a
    transaction by walking its typed edges."""
    if transaction_id not in graph:
        raise KeyError(f"Transaction {transaction_id} not in graph")
    user = _first_neighbor_of_type(graph, transaction_id, NodeType.USER)
    return {
        "user_id": user,
        "device_id": _first_neighbor_of_type(graph, user, NodeType.DEVICE) if user else None,
        "ip_id": _first_neighbor_of_type(graph, user, NodeType.IP) if user else None,
        "payment_instrument_id": _first_neighbor_of_type(graph, user, NodeType.PAYMENT_INSTRUMENT) if user else None,
        "merchant_id": _first_neighbor_of_type(graph, transaction_id, NodeType.MERCHANT),
    }


def _relationship_counts(graph: nx.Graph, node: str) -> dict:
    counts: dict = {}
    for _u, _v, attrs in graph.edges(node, data=True):
        rel = attrs.get("relationship_type")
        counts[rel] = counts.get(rel, 0) + 1
    return counts


def _temporal_density(graph: nx.Graph, transaction_id: str, ts, user_tx_index) -> dict:
    """Count related transactions within 5m/1h/24h windows (lookback-only)."""
    return {
        "within_5m": len(transactions_in_window(graph, transaction_id, ts, WINDOW_5M, user_tx_index=user_tx_index)),
        "within_1h": len(transactions_in_window(graph, transaction_id, ts, WINDOW_1H, user_tx_index=user_tx_index)),
        "within_24h": len(transactions_in_window(graph, transaction_id, ts, WINDOW_24H, user_tx_index=user_tx_index)),
    }


def extract_graph_evidence(graph: nx.Graph, transaction_id: str) -> dict:
    """Return structured (non-AI) evidence for a transaction from the graph.

    Returns relationship counts and temporal density derived from *observed*
    graph relationships. This is structured data only — no natural-language
    explanation is generated here.
    """
    if transaction_id not in graph:
        raise KeyError(f"Transaction {transaction_id} not in graph")

    ts = graph.nodes[transaction_id].get("timestamp")
    user_tx = _user_transactions_index(graph)

    tx_attrs = graph.nodes[transaction_id]
    user = _first_neighbor_of_type(graph, transaction_id, NodeType.USER)
    device = _first_neighbor_of_type(graph, user, NodeType.DEVICE) if user else None
    ip = _first_neighbor_of_type(graph, user, NodeType.IP) if user else None
    pi = _first_neighbor_of_type(graph, user, NodeType.PAYMENT_INSTRUMENT) if user else None

    evidence = {
        "transaction_id": transaction_id,
        "transaction_timestamp": ts.isoformat() if ts else None,
        "amount": tx_attrs.get("amount"),
        "status": tx_attrs.get("status"),
        "user_id": user,
        "device_id": device,
        "ip_id": ip,
        "payment_instrument_id": pi,
        "shared_device_users": sorted(users_of(graph, device, NodeType.DEVICE)) if device else [],
        "shared_ip_users": sorted(users_of(graph, ip, NodeType.IP)) if ip else [],
        "shared_payment_users": sorted(users_of(graph, pi, NodeType.PAYMENT_INSTRUMENT)) if pi else [],
        "relationship_counts": _relationship_counts(graph, transaction_id),
        "temporal_density": _temporal_density(graph, transaction_id, ts, user_tx),
    }
    return evidence


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv(
        _PROJECT_ROOT / "data" / "samples" / "transactions.csv",
        parse_dates=["timestamp"],
        nrows=10000,
    )
    from engine.graph.graph_builder import build_graph

    g = build_graph(df)
    deg = device_degree(g)
    print("top-5 high-connectivity devices:", sorted(deg.items(), key=lambda kv: -kv[1])[:5])