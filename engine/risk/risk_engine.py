"""Campaign risk engine (Phase 4).

Consumes Phase-2 risk probabilities, Phase-3 candidate campaigns and graph
evidence, plus transaction metadata, and produces transparent CampaignAssessment
objects. Ground-truth fields are consumed ONLY in the dedicated evaluation
functions at the bottom of this module, never in scoring.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import pandas as pd

# Allow running directly: python engine/risk/risk_engine.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine.graph.graph_features import users_of, transaction_entities
from engine.graph.graph_builder import NodeType
from engine.risk.rules import (
    HIGH_RISK_TX_THRESHOLD,
    JACCARD_MERGE_THRESHOLD,
    SHARED_SEVERITY,
    jaccard,
    risk_level_for_score,
)
from engine.risk.scoring import (
    behavioral_risk_score,
    campaign_confidence,
    combined_campaign_score,
    concentration_risk_score,
    relationship_risk_score,
    temporal_risk_score,
    transaction_risk_score,
)
from engine.risk.result_types import CampaignAssessment, EvidenceItem

RISK_LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class TxIndex:
    """Lightweight per-transaction lookup built once from the CSV."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.set_index("transaction_id")
        self.amounts = self.frame["amount"].to_dict()
        self.statuses = self.frame["status"].astype(str).to_dict()
        self.timestamps = self.frame["timestamp"].to_dict()
        self.users = self.frame["user_id"].to_dict()
        self.devices = self.frame["device_id"].to_dict()
        self.ips = self.frame["ip_id"].to_dict()
        self.pis = self.frame["payment_instrument_id"].to_dict()

    def amount(self, tx_id: str) -> float:
        return float(self.amounts.get(tx_id, 0.0))

    def status(self, tx_id: str) -> str:
        return str(self.statuses.get(tx_id, "success"))

    def timestamp(self, tx_id: str) -> datetime:
        return self.timestamps[tx_id]

    def exposure(self, tx_ids: list[str]) -> float:
        return round(sum(self.amount(t) for t in tx_ids), 2)


def _candidate_shared_users(
    graph: nx.Graph,
    candidate_user_ids: set[str],
    entity_ids: list[str],
    entity_type: NodeType,
) -> dict[str, int]:
    """Count, per entity, how many *candidate* users share it (scoped)."""
    out: dict[str, int] = {}
    for entity_id in entity_ids:
        if entity_id is None:
            continue
        shared = users_of(graph, entity_id, entity_type)
        out[entity_id] = len(shared & candidate_user_ids)
    return out


def _tx_per_entity(candidate_tx: list[str], tx_index: TxIndex, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tx in candidate_tx:
        eid = getattr(tx_index, key).get(tx)
        if eid is None:
            continue
        counts[eid] = counts.get(eid, 0) + 1
    return counts


def _peak_5m_velocity(timestamps: list[datetime]) -> int:
    """Max transactions in any 5-minute rolling bucket (deterministic)."""
    ordered = sorted(timestamps)
    peak = 0
    for i, ts in enumerate(ordered):
        j = i
        while j < len(ordered) and ordered[j] - ts <= timedelta(minutes=5):
            j += 1
        peak = max(peak, j - i)
    return peak


def _severity_from_count(count: int) -> str:
    return _map_count_severity(count)


def _map_count_severity(count: int) -> str:
    if count >= SHARED_SEVERITY["high"]:
        return "high"
    if count >= SHARED_SEVERITY["medium"]:
        return "medium"
    return "low"


def _severity_for_threshold(value: float, medium: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def extract_evidence(
    campaign: dict,
    graph: nx.Graph,
    tx_index: TxIndex,
    risk_scores: dict[str, float],
) -> list[EvidenceItem]:
    """Build structured evidence items from real candidate data only."""
    tx_ids = list(campaign["transaction_ids"])
    user_ids = set(campaign["user_ids"])
    device_ids = list(campaign["device_ids"])
    ip_ids = list(campaign["ip_ids"])
    pi_ids = list(campaign["payment_instrument_ids"])

    device_shared = _candidate_shared_users(graph, user_ids, device_ids, NodeType.DEVICE)
    ip_shared = _candidate_shared_users(graph, user_ids, ip_ids, NodeType.IP)
    pi_shared = _candidate_shared_users(graph, user_ids, pi_ids, NodeType.PAYMENT_INSTRUMENT)

    evidence: list[EvidenceItem] = []

    # ---- shared entity evidence ------------------------------------------
    for label, shared_map, entities_label in (
        ("shared_device", device_shared, device_ids),
        ("shared_ip", ip_shared, ip_ids),
        ("shared_payment_instrument", pi_shared, pi_ids),
    ):
        top = max(shared_map.items(), key=lambda kv: (kv[1], kv[0]))[0] if shared_map else None
        count = shared_map.get(top, 0) if top else 0
        if top and count >= 2:
            supporting = [t for t in tx_ids if tx_index.devices.get(t) == top or tx_index.ips.get(t) == top or tx_index.pis.get(t) == top][:10]
            evidence.append(EvidenceItem(
                type=label,
                severity=_severity_from_count(count),
                description=f"{count} users share {top} in this campaign",
                entities=[top],
                supporting_transactions=supporting,
            ))

    # ---- temporal burst evidence ------------------------------------------
    duration_s = int(campaign["duration_seconds"])
    if len(tx_ids) >= 3 and duration_s > 0:
        per_hour = len(tx_ids) / (duration_s / 3600.0)
        if per_hour >= 30.0:  # documented reference: >=30 tx/hr is a clear burst
            evidence.append(EvidenceItem(
                type="temporal_burst",
                severity=_severity_for_threshold(per_hour, 30.0, 120.0),
                description=(
                    f"{len(tx_ids)} transactions occurred within "
                    f"{duration_s / 60.0:.1f} minutes"
                ),
                entities=sorted(user_ids)[:5],
                supporting_transactions=tx_ids[:10],
            ))

    # ---- high transaction risk evidence ---------------------------
    high_risk_tx = [t for t in tx_ids if risk_scores.get(t, 0.0) >= HIGH_RISK_TX_THRESHOLD]
    if high_risk_tx:
        evidence.append(EvidenceItem(
            type="high_transaction_risk",
            severity=_severity_for_threshold(len(high_risk_tx), 3, 10),
            description=(
                f"{len(high_risk_tx)}/{len(tx_ids)} transactions have elevated "
                "transaction-level risk"
            ),
            entities=[],
            supporting_transactions=high_risk_tx[:10],
        ))

    # ---- high-velocity evidence -----------------------------------
    tss = [tx_index.timestamp(t) for t in tx_ids if t in tx_index.timestamps]
    peak_5m = _peak_5m_velocity(tss) if len(tss) >= 2 else 0
    if peak_5m >= 4:
        per_min = peak_5m / 5.0
        evidence.append(EvidenceItem(
            type="high_velocity",
            severity=_severity_for_threshold(per_min, 1.0, 3.0),
            description=f"peak of {peak_5m} transactions within a 5-minute window",
            entities=[],
            supporting_transactions=tx_ids[:10],
        ))

    # ---- refund / failed patterns -----------------------------------
    statuses = [tx_index.status(t) for t in tx_ids]
    refund_ratio = sum(1 for s in statuses if s == "refunded") / max(len(statuses), 1)
    failed_ratio = sum(1 for s in statuses if s == "failed") / max(len(statuses), 1)
    if refund_ratio >= 0.3:
        evidence.append(EvidenceItem(
            type="refund_pattern",
            severity=_severity_for_threshold(refund_ratio, 0.3, 0.6),
            description=f"{refund_ratio:.0%} of campaign transactions are refunds",
            entities=[],
            supporting_transactions=tx_ids[:10],
        ))
    if failed_ratio >= 0.3:
        evidence.append(EvidenceItem(
            type="failed_payment_pattern",
            severity=_severity_for_threshold(failed_ratio, 0.3, 0.6),
            description=f"{failed_ratio:.0%} of campaign transactions failed",
            entities=[],
            supporting_transactions=tx_ids[:10],
        ))

    # ---- entity concentration ----------------------------------------
    concentration = _campaign_concentration_value(campaign, tx_index)
    if concentration >= 0.5:
        evidence.append(EvidenceItem(
            type="entity_concentration",
            severity=_severity_for_threshold(concentration, 0.5, 0.75),
            description=f"activity is concentrated around {len(device_ids)} device(s), "
                        f"{len(ip_ids)} IP(s), {len(pi_ids)} payment instrument(s)",
            entities=list(device_ids[:5] + ip_ids[:5] + pi_ids[:5]),
            supporting_transactions=tx_ids[:10],
        ))

    return evidence


def _campaign_concentration_value(campaign: dict, tx_index: TxIndex) -> float:
    """Log-scaled activity concentration for evidence/severity purposes."""
    tx_ids = list(campaign["transaction_ids"])
    n_entities = max(len(campaign["device_ids"]) + len(campaign["ip_ids"])
                     + len(campaign["payment_instrument_ids"]), 1)
    return min(1.0, (len(tx_ids) / max(n_entities * 2, 1)) / 5.0)


def assess_campaign(
    campaign: dict,
    graph: nx.Graph,
    tx_index: TxIndex,
    risk_scores: dict[str, float],
) -> CampaignAssessment:
    """Produce a full, deterministic, evidence-backed assessment.

    ``risk_scores`` is {transaction_id: fraud probability}. Ground-truth fields
    are never read here.
    """
    tx_ids = list(campaign["transaction_ids"])
    user_ids = set(campaign["user_ids"])
    device_ids = list(campaign["device_ids"])
    ip_ids = list(campaign["ip_ids"])
    pi_ids = list(campaign["payment_instrument_ids"])

    risks = [risk_scores.get(t, 0.0) for t in tx_ids]
    amounts = [tx_index.amount(t) for t in tx_ids]
    statuses = [tx_index.status(t) for t in tx_ids]

    # ---- dimension inputs ------------------------------------------------
    device_shared = _candidate_shared_users(graph, user_ids, device_ids, NodeType.DEVICE)
    ip_shared = _candidate_shared_users(graph, user_ids, ip_ids, NodeType.IP)
    pi_shared = _candidate_shared_users(graph, user_ids, pi_ids, NodeType.PAYMENT_INSTRUMENT)

    max_shared_device = max(device_shared.values(), default=0)
    max_shared_ip = max(ip_shared.values(), default=0)
    max_shared_pi = max(pi_shared.values(), default=0)

    tx_per_device = _tx_per_entity(tx_ids, tx_index, "devices")
    tx_per_ip = _tx_per_entity(tx_ids, tx_index, "ips")
    tx_per_pi = _tx_per_entity(tx_ids, tx_index, "pis")

    tx_dim = transaction_risk_score(risks)
    rel_dim = relationship_risk_score(
        float(campaign.get("relationship_density", 0.0)),
        max_shared_device,
        max_shared_ip,
        max_shared_pi,
    )
    temp_dim = temporal_risk_score(len(tx_ids), int(campaign["duration_seconds"]))
    conc_dim = concentration_risk_score(
        list(tx_per_device.values()),
        list(tx_per_ip.values()),
        list(tx_per_pi.values()),
        list(device_shared.values()),
        list(ip_shared.values()),
        list(pi_shared.values()),
    )
    behav_dim = behavioral_risk_score(statuses, amounts)

    dimensions = {
        "transaction_risk": round(float(tx_dim), 4),
        "relationship_risk": round(float(rel_dim), 4),
        "temporal_risk": round(float(temp_dim), 4),
        "concentration_risk": round(float(conc_dim), 4),
        "behavioral_risk": round(float(behav_dim), 4),
    }

    # ---- confidence (separate from risk) -------------------------------
    completeness = _completeness(user_ids, device_ids, ip_ids, pi_ids)
    risk_dispersion = _risk_dispersion(risks)
    graph_evidence_strength = round(
        min(1.0, max(max_shared_device, max_shared_ip, max_shared_pi) / 8.0), 4
    )
    confidence = campaign_confidence(
        dimensions, len(tx_ids), completeness, risk_dispersion, graph_evidence_strength
    )

    high_risk_tx = [t for t, r in zip(tx_ids, risks) if r >= HIGH_RISK_TX_THRESHOLD]
    score = combined_campaign_score(dimensions)
    evidence = extract_evidence(campaign, graph, tx_index, risk_scores)

    return CampaignAssessment(
        campaign_id=str(campaign["campaign_candidate_id"]),
        risk_score=score,
        risk_level=risk_level_for_score(score),
        confidence=confidence,
        transaction_count=len(tx_ids),
        user_count=len(user_ids),
        device_count=len(device_ids),
        ip_count=len(ip_ids),
        payment_instrument_count=len(pi_ids),
        start_time=str(campaign["start_time"]),
        end_time=str(campaign["end_time"]),
        duration_seconds=int(campaign["duration_seconds"]),
        estimated_exposure=tx_index.exposure(tx_ids),
        evidence=evidence,
        transaction_risk=dimensions["transaction_risk"],
        relationship_risk=dimensions["relationship_risk"],
        temporal_risk=dimensions["temporal_risk"],
        concentration_risk=dimensions["concentration_risk"],
        behavioral_risk=dimensions["behavioral_risk"],
        high_risk_transaction_count=len(high_risk_tx),
        transaction_ids=tx_ids,
        components=dimensions,
    )


def _completeness(user_ids, device_ids, ip_ids, pi_ids) -> float:
    present = sum(1 for seq in (user_ids, device_ids, ip_ids, pi_ids) if len(seq) > 0)
    return present / 4.0


def _risk_dispersion(risks: list[float]) -> float:
    """Normalized dispersion (0..1) of per-transaction risk scores."""
    if len(risks) < 2:
        return 0.0
    mean = sum(risks) / len(risks)
    if mean == 0:
        return 0.0
    spread = max(risks) - min(risks)
    return min(1.0, spread / 1.0)


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def assess_all(
    campaigns: list[dict],
    graph: nx.Graph,
    tx_index: TxIndex,
    risk_scores: dict[str, float],
) -> list[CampaignAssessment]:
    return [assess_campaign(c, graph, tx_index, risk_scores) for c in campaigns]


def deduplicate_candidates(
    campaigns: list[dict],
    jaccard_threshold: float = JACCARD_MERGE_THRESHOLD,
) -> list[dict]:
    """Deterministically remove heavily-overlapping candidate campaigns.

    Strategy: sort candidates by (max_risk desc, transaction_count desc,
    campaign_id). Scan in order; if a later candidate's transaction set has
    Jaccard overlap >= threshold with an already-kept representative, it is
    dropped and recorded on the representative's ``merged_candidates`` list.
    This is a documented heuristic — it never merges unrelated campaigns.
    """
    ordered = sorted(
        campaigns,
        key=lambda c: (-c["risk_score_summary"]["max"],
                       -len(c.get("transaction_ids", [])),
                       c["campaign_candidate_id"]),
    )
    representatives: list[dict] = []
    tx_sets = {c["campaign_candidate_id"]: set(c["transaction_ids"]) for c in ordered}

    for candidate in ordered:
        cid = candidate["campaign_candidate_id"]
        merged_into = None
        for rep in representatives:
            if jaccard(tx_sets[cid], tx_sets[rep["campaign_candidate_id"]]) >= jaccard_threshold:
                merged_into = rep
                break
        if merged_into is None:
            candidate["merged_campaign_ids"] = []
            representatives.append(candidate)
        else:
            merged_into.setdefault("merged_campaign_ids", []).append(cid)

    return representatives


def rank_campaigns(
    assessments: list[CampaignAssessment],
    min_risk_level: str | None = None,
    min_exposure: float | None = None,
    min_transactions: int | None = None,
) -> list[CampaignAssessment]:
    """Deterministic ranking: risk_score desc, exposure desc, campaign_id asc."""
    filtered = assessments
    if min_risk_level is not None:
        filtered = [a for a in filtered if RISK_LEVEL_ORDER[a.risk_level] >= RISK_LEVEL_ORDER[min_risk_level]]
    if min_exposure is not None:
        filtered = [a for a in filtered if a.estimated_exposure >= min_exposure]
    if min_transactions is not None:
        filtered = [a for a in filtered if a.transaction_count >= min_transactions]
    return sorted(
        filtered,
        key=lambda a: (-a.risk_score, -a.estimated_exposure, a.campaign_id),
    )


# ---------------------------------------------------------------------------
# Ground-truth evaluation. THE ONLY place ground-truth labels are consumed.
# ---------------------------------------------------------------------------

def evaluate_against_ground_truth(
    assessments: list[CampaignAssessment],
    tx_df: pd.DataFrame,
    min_risk_level: str = "HIGH",
) -> dict:
    """Campaign-level evaluation metrics against ground truth.

    Definitions (documented):
      - ``high_risk_campaign_tx``: transactions inside campaigns at/above
        ``min_risk_level``.
      - fraud_transaction_coverage:   |fraud tx in high-risk campaigns| / |all fraud tx|
      - legitimate_transaction_coverage: |legit tx in high-risk campaigns| / |all legit tx|
      - candidate_campaign_precision: |fraud tx in high-risk campaigns| /
        |tx in high-risk campaigns|   (transaction-level precision of flagged set)
      - campaign_recall: number of ground-truth fraud campaigns with at least
        one transaction covered by a high-risk candidate / total ground-truth
        fraud campaigns.
      - campaign_precision: number of high-risk candidates containing at least
        one fraud transaction / number of high-risk candidates.

    These definitions are stated explicitly; other definitions would yield
    different numbers.
    """
    is_fraud = tx_df.set_index("transaction_id")["is_fraud"].astype(int).to_dict()
    gt_campaign = (
        tx_df[tx_df["is_fraud"] == 1]
        .set_index("transaction_id")["fraud_campaign_id"]
        .to_dict()
    )

    selected = [a for a in assessments if RISK_LEVEL_ORDER[a.risk_level] >= RISK_LEVEL_ORDER[min_risk_level]]
    campaign_tx = {tx for a in selected for tx in a.transaction_ids}

    total_fraud = sum(is_fraud.values())
    total_legit = len(is_fraud) - total_fraud
    fraud_in = sum(1 for tx in campaign_tx if is_fraud.get(tx, 0) == 1)
    legit_in = sum(1 for tx in campaign_tx if is_fraud.get(tx, 0) == 0)

    all_gt_campaigns = set(gt_campaign.values())
    covered_gt_campaigns = set()
    for a in selected:
        for tx in a.transaction_ids:
            gt = gt_campaign.get(tx)
            if gt is not None:
                covered_gt_campaigns.add(gt)

    high_risk_candidates = len(selected)
    high_risk_with_fraud = 0
    for a in selected:
        if any(is_fraud.get(tx, 0) == 1 for tx in a.transaction_ids):
            high_risk_with_fraud += 1

    return {
        "min_risk_level": min_risk_level,
        "assessed_campaigns": len(assessments),
        "high_risk_campaigns": high_risk_candidates,
        "fraud_transaction_coverage": round(fraud_in / total_fraud, 4) if total_fraud else 0.0,
        "legitimate_transaction_coverage": round(legit_in / total_legit, 4) if total_legit else 0.0,
        "candidate_campaign_precision": round(fraud_in / len(campaign_tx), 4) if campaign_tx else 0.0,
        "campaign_recall": round(len(covered_gt_campaigns) / len(all_gt_campaigns), 4) if all_gt_campaigns else 0.0,
        "campaign_precision": round(high_risk_with_fraud / high_risk_candidates, 4) if high_risk_candidates else 0.0,
        "fraud_tx_in_high_risk": fraud_in,
        "legit_tx_in_high_risk": legit_in,
    }


def analyze_false_negatives(
    assessments: list[CampaignAssessment],
    tx_df: pd.DataFrame,
    predicted_labels: dict[str, int],
    risk_scores: dict[str, float] | None = None,
    min_risk_level: str = "HIGH",
) -> list[dict]:
    """Diagnose Phase-2 false negatives covered by high-risk campaigns.

    A false negative is a ground-truth fraud transaction the Phase-2 model
    predicted as legitimate (0). For each, find containing high-risk campaign
    assessments. This is a diagnostic experiment: labels are used here only,
    never to influence scoring.
    """
    if risk_scores is None:
        risk_scores = {}
    is_fraud = tx_df.set_index("transaction_id")["is_fraud"].astype(int).to_dict()
    rows: list[dict] = []
    for tx_id, actual in is_fraud.items():
        pred = predicted_labels.get(tx_id, 0)
        if actual == 1 and pred == 0:
            containing = [
                a for a in assessments
                if RISK_LEVEL_ORDER[a.risk_level] >= RISK_LEVEL_ORDER[min_risk_level]
                and tx_id in a.transaction_ids
            ]
            rows.append({
                "false_negative_transaction_id": tx_id,
                "transaction_risk": round(float(risk_scores.get(tx_id, 0.0)), 4),
                "campaigns": [
                    {
                        "campaign_id": a.campaign_id,
                        "campaign_risk": a.risk_score,
                        "risk_level": a.risk_level,
                        "evidence_types": [e.type for e in a.evidence],
                    }
                    for a in sorted(containing, key=lambda a: -a.risk_score)
                ],
            })
    return rows


if __name__ == "__main__":
    import json

    from engine.graph.campaign_detector import find_campaign_candidates, load_phase2_risk_scores
    from engine.graph.graph_builder import build_graph

    df = pd.read_csv(_PROJECT_ROOT / "data" / "samples" / "transactions.csv",
                     parse_dates=["timestamp"])
    graph = build_graph(df)
    risk_scores = load_phase2_risk_scores()
    candidates = find_campaign_candidates(graph, risk_scores=risk_scores)
    deduped = deduplicate_candidates(candidates)
    assessments = assess_all(deduped, graph, TxIndex(df), risk_scores)
    ranked = rank_campaigns(assessments, min_risk_level=None)
    print(f"Candidates: {len(candidates)} -> deduped: {len(deduped)}")
    print(f"Assessments: {len(assessments)}")
    for a in ranked[:3]:
        print(json.dumps(a.to_dict(), indent=2, default=str)[:1600])