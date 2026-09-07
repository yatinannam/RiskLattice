"""API integration tests for the RiskLattice FastAPI layer.

Uses the baseline dataset (fastest) via a shared TestClient. Verifies all
endpoints, error handling, that ground-truth labels never appear in any
response, and that simulation never executes real actions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Use the baseline dataset for fast, deterministic API tests.
os.environ["RISKLATTICE_DATASET"] = "baseline"

# Make the `app` package importable (apps/api) and the repo root (engines).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
API_DIR = ROOT / "apps" / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from fastapi.testclient import TestClient  # noqa: E402

GUARDED_KEYS = ("is_fraud", "fraud_campaign_id", "scenario")


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _assert_no_ground_truth(obj):
    if isinstance(obj, dict):
        for key in GUARDED_KEYS:
            assert key not in obj, f"ground-truth key {key} leaked"
        for value in obj.values():
            _assert_no_ground_truth(value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_ground_truth(item)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mode"] == "TEST_MODE"
    assert body["dataset"] == "baseline"


def test_overview(client):
    r = client.get("/api/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["active_campaigns"] > 0
    assert set(body["risk_distribution"]) >= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert isinstance(body["recent_campaigns"], list)
    _assert_no_ground_truth(body)


def test_campaign_listing(client):
    r = client.get("/api/campaigns")
    assert r.status_code == 200
    campaigns = r.json()
    assert len(campaigns) > 0
    _assert_no_ground_truth(campaigns)
    scores = [c["risk_score"] for c in campaigns]
    assert scores == sorted(scores, reverse=True)


def test_campaign_listing_filters(client):
    r = client.get("/api/campaigns", params={"risk_level": "HIGH"})
    assert r.status_code == 200
    assert all(c["risk_level"] == "HIGH" for c in r.json())
    r2 = client.get("/api/campaigns", params={"min_score": 70})
    assert r2.status_code == 200
    assert all(c["risk_score"] >= 70 for c in r2.json())


def test_campaign_detail(client, campaign_id):
    r = client.get(f"/api/campaigns/{campaign_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["campaign_id"] == campaign_id
    assert "risk_dimensions" in detail
    assert "evidence" in detail
    _assert_no_ground_truth(detail)


def test_campaign_graph(client, campaign_id):
    r = client.get(f"/api/campaigns/{campaign_id}/graph")
    assert r.status_code == 200
    g = r.json()
    assert "nodes" in g and "edges" in g
    _assert_no_ground_truth(g)
    assert {n["type"] for n in g["nodes"]}  # non-empty


def test_investigation(client, campaign_id):
    r = client.get(f"/api/campaigns/{campaign_id}/investigation")
    assert r.status_code == 200
    inv = r.json()
    assert inv["campaign_id"] == campaign_id
    # Report must be substantive, not an empty shell.
    assert inv["executive_summary"], "investigation must contain a summary"
    assert len(inv["why_flagged"]) >= 1, "investigation must contain findings"
    assert len(inv["uncertainty"]) >= 1, "investigation must surface uncertainty"
    for finding in inv["why_flagged"]:
        assert finding["type"] in ("FACT", "INFERENCE", "UNCERTAINTY")
        assert finding["evidence_ids"]
    _assert_no_ground_truth(inv)


def test_containment(client, campaign_id):
    r = client.get(f"/api/campaigns/{campaign_id}/containment")
    assert r.status_code == 200
    containment = r.json()
    assert containment["campaign_id"] == campaign_id
    assert containment["recommendation"] in ("CONTAIN", "NO_SAFE_ACTION")
    assert containment["test_mode"] is True
    _assert_no_ground_truth(containment)


def test_simulation(client, campaign_id):
    detail = client.get(f"/api/campaigns/{campaign_id}").json()
    uid = detail["user_ids"][0]
    r = client.post(
        f"/api/campaigns/{campaign_id}/containment/simulate",
        json={"action_type": "BLOCK_USER", "target_id": uid},
    )
    assert r.status_code == 200
    sim = r.json()
    assert sim["test_mode"] is True
    assert sim["fraud_containment_rate"] >= 0.0
    assert "SIMULATION" in sim["message"].upper()
    _assert_no_ground_truth(sim)


def test_audit(client):
    r = client.get("/api/audit")
    assert r.status_code == 200
    events = r.json()
    assert isinstance(events, list) and len(events) > 0
    for event in events:
        assert "event" in event
        assert "campaign" in event


def test_nonexistent_campaign(client):
    assert client.get("/api/campaigns/DOES_NOT_EXIST").status_code == 404
    assert client.get("/api/campaigns/DOES_NOT_EXIST/graph").status_code == 404
    assert client.get(
        "/api/campaigns/DOES_NOT_EXIST/investigation").status_code == 404


def test_invalid_action(client, campaign_id):
    r = client.post(
        f"/api/campaigns/{campaign_id}/containment/simulate",
        json={"action_type": "NOT_AN_ACTION", "target_id": "x"},
    )
    assert r.status_code == 400
    r2 = client.post(
        f"/api/campaigns/{campaign_id}/containment/simulate",
        json={"action_type": "BLOCK_USER", "target_id": "NOT_IN_CAMPAIGN"},
    )
    assert r2.status_code == 400


def test_simulation_never_executes_real_actions(client):
    campaigns = client.get("/api/campaigns").json()
    cid = campaigns[0]["campaign_id"]
    detail = client.get(f"/api/campaigns/{cid}").json()
    uid = detail["user_ids"][0]
    r = client.post(
        f"/api/campaigns/{cid}/containment/simulate",
        json={"action_type": "BLOCK_USER", "target_id": uid},
    )
    body = r.json()
    assert body["test_mode"] is True
    assert "no real action executed" in body["message"].lower()


def test_no_ground_truth_anywhere(client):
    for path in ("/api/overview", "/api/campaigns", "/api/audit"):
        r = client.get(path)
        assert r.status_code == 200
        _assert_no_ground_truth(r.json())
    campaigns = client.get("/api/campaigns").json()
    if campaigns:
        cid = campaigns[0]["campaign_id"]
        for suffix in ("", "/graph", "/investigation", "/containment"):
            response = client.get(f"/api/campaigns/{cid}{suffix}")
            assert response.status_code == 200
            _assert_no_ground_truth(response.json())


def test_openapi_docs_available(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    for required in ("/api/overview", "/api/campaigns",
                     "/api/campaigns/{campaign_id}",
                     "/api/campaigns/{campaign_id}/graph",
                     "/api/campaigns/{campaign_id}/investigation",
                     "/api/campaigns/{campaign_id}/containment",
                     "/api/campaigns/{campaign_id}/containment/simulate",
                     "/api/audit"):
        assert required in paths


@pytest.fixture
def campaign_id(client):
    campaigns = client.get("/api/campaigns").json()
    assert campaigns
    return campaigns[0]["campaign_id"]