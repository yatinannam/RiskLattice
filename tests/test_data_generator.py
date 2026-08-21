"""Phase 1 tests for the synthetic dataset generator.

These tests cover:
  * dataset size (exactly 10,000 transactions)
  * approximate 85% / 15% legitimate / fraud class balance
  * deterministic reproducibility with SEED = 42
  * presence of every defined fraud scenario
  * ground-truth field consistency (is_fraud / fraud_campaign_id / scenario)
  * schema validity and synthetic-only entity IDs (no real payment data)
  * legitimate edge cases that must NOT be mistaken for fraud
"""

from __future__ import annotations

import csv

import pytest

from data.generators import generate_dataset as gen
from data.schemas.transaction import TransactionStatus

FRAUD_SCENARIOS = {
    "account_farm",
    "payment_instrument_abuse",
    "coordinated_burst",
    "refund_abuse",
}

EXPECTED_CSV_COLUMNS = {
    "transaction_id",
    "timestamp",
    "merchant_id",
    "user_id",
    "device_id",
    "ip_id",
    "payment_instrument_id",
    "amount",
    "currency",
    "payment_method",
    "status",
    "is_fraud",
    "fraud_campaign_id",
    "scenario",
}


@pytest.fixture(scope="module")
def dataset():
    """Generate one deterministic dataset for the whole test module."""
    return gen.generate_dataset()


def test_exactly_10000_transactions(dataset):
    assert len(dataset) == gen.NUM_TRANSACTIONS == 10_000


def test_legit_fraud_ratio_matches_spec(dataset):
    fraud_count = sum(1 for tx in dataset if tx.is_fraud)
    ratio = fraud_count / len(dataset)
    assert 0.14 <= ratio <= 0.16, ratio


def test_seed_42_is_deterministic():
    first = gen.generate_dataset()
    second = gen.generate_dataset()

    def _dump(rows):
        return [row.model_dump(mode="json") for row in rows]

    assert _dump(first) == _dump(second)


def test_all_fraud_scenarios_present(dataset):
    present = {tx.scenario for tx in dataset if tx.is_fraud}
    assert FRAUD_SCENARIOS <= present


def test_fraud_has_campaign_and_scenario(dataset):
    fraud = [tx for tx in dataset if tx.is_fraud]
    assert fraud
    for tx in fraud:
        assert tx.fraud_campaign_id
        assert tx.scenario in FRAUD_SCENARIOS


def test_legitimate_has_no_campaign(dataset):
    legit = [tx for tx in dataset if not tx.is_fraud]
    assert legit
    for tx in legit:
        assert tx.fraud_campaign_id is None
        assert tx.scenario == "legitimate"


def test_schema_validates_all_rows(dataset):
    for tx in dataset:
        assert tx.amount > 0
        assert tx.currency == "INR"
        assert tx.payment_method in {"upi", "card", "netbanking", "wallet"}
        assert tx.status in {"success", "failed", "refunded"}


def test_synthetic_ids_only_no_real_payment_info(dataset):
    for tx in dataset:
        assert tx.user_id.startswith("USER_")
        assert tx.device_id.startswith("DEV_")
        assert tx.ip_id.startswith("IP_")
        assert tx.payment_instrument_id.startswith("PI_")


def test_legit_edge_cases_are_represented(dataset):
    legit = [tx for tx in dataset if not tx.is_fraud]

    # Refunds among legitimate traffic.
    assert any(tx.status == TransactionStatus.REFUNDED for tx in legit)
    # Failed / declined payment attempts among legitimate traffic.
    assert any(tx.status == TransactionStatus.FAILED for tx in legit)
    # High-value legitimate transactions.
    assert any(tx.amount >= 50_000 for tx in legit)


def test_shared_ip_occurs_among_legitimate(dataset):
    """Shared IP must exist among legitimate users.

    This is a data-level guard: a shared IP in the dataset is by itself not a
    fraud label. (The graph / risk layers later consume this without treating
    a shared IP as true proof of fraud.)
    """
    ip_users: dict[str, set[str]] = {}
    for tx in dataset:
        if not tx.is_fraud:
            ip_users.setdefault(tx.ip_id, set()).add(tx.user_id)

    shared = {
        ip: users for ip, users in ip_users.items() if len(users) > 1
    }
    assert shared, "expected legitimate shared-infrastructure groups"


def test_shared_device_occurs_among_legitimate(dataset):
    """Shared devices appear among legitimate traffic too."""
    device_users: dict[str, set[str]] = {}
    for tx in dataset:
        if not tx.is_fraud:
            device_users.setdefault(tx.device_id, set()).add(tx.user_id)

    shared = {
        dev: users
        for dev, users in device_users.items()
        if len(users) > 1
    }
    assert shared, "expected legitimate shared-device groups"


def test_save_csv_writes_expected_header_and_rows(tmp_path, dataset):
    dest = tmp_path / "transactions.csv"
    gen.save_csv(dataset, output_path=dest)
    assert dest.exists()

    with dest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 10_000
    assert set(rows[0].keys()) == EXPECTED_CSV_COLUMNS
    assert len(rows[0].keys()) == 14