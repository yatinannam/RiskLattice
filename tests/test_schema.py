"""Schema validation tests for the Transaction model."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from data.schemas.transaction import (
    PaymentMethod,
    Transaction,
    TransactionStatus,
)


def _minimal(**overrides):
    data = {
        "transaction_id": "TXN_000001",
        "timestamp": "2026-08-01T00:00:00",
        "merchant_id": "MERCH_001",
        "user_id": "USER_0001",
        "device_id": "DEV_0001",
        "ip_id": "IP_0001",
        "payment_instrument_id": "PI_0001",
        "amount": 100.0,
        "currency": "INR",
        "payment_method": "upi",
        "status": "success",
    }
    data.update(overrides)
    return data


def test_required_fields_present():
    fields = set(Transaction.model_fields)
    expected = {
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
    assert expected == fields


def test_optional_fields_have_safe_defaults():
    tx = Transaction.model_validate(_minimal())
    assert tx.is_fraud is False
    assert tx.currency == "INR"
    assert tx.fraud_campaign_id is None
    assert tx.scenario == "legitimate"


def test_timestamp_parsed_to_datetime():
    tx = Transaction.model_validate(_minimal())
    assert isinstance(tx.timestamp, datetime)


def test_amount_must_be_positive():
    with pytest.raises(ValidationError):
        Transaction.model_validate(_minimal(amount=0))
    with pytest.raises(ValidationError):
        Transaction.model_validate(_minimal(amount=-10))


def test_enums_validate():
    assert TransactionStatus.SUCCESS.value == "success"
    assert TransactionStatus.REFUNDED.value == "refunded"
    assert PaymentMethod.CARD.value == "card"

    with pytest.raises(ValidationError):
        Transaction.model_validate(_minimal(status="processing"))


def test_ground_truth_fields_accept_eval_labels():
    """Ground-truth fields exist on the schema but are evaluation-only.

    These fields must never be used as model features (enforced in Phase 2).
    Here we only confirm the schema can carry them for labeling/reporting.
    """
    tx = Transaction.model_validate(
        _minimal(
            is_fraud=True,
            fraud_campaign_id="FARM_1234",
            scenario="account_farm",
        )
    )
    assert tx.is_fraud is True
    assert tx.fraud_campaign_id == "FARM_1234"
    assert tx.scenario == "account_farm"


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        Transaction.model_validate({})
    with pytest.raises(ValidationError):
        Transaction.model_validate({"transaction_id": "X", "amount": 1.0})