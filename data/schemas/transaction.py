from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TransactionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    REFUNDED = "refunded"


class PaymentMethod(str, Enum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class Transaction(BaseModel):
    transaction_id: str
    timestamp: datetime

    merchant_id: str
    user_id: str
    device_id: str
    ip_id: str
    payment_instrument_id: str

    amount: float = Field(gt=0)
    currency: str = "INR"
    payment_method: PaymentMethod
    status: TransactionStatus

    # Ground truth — used ONLY for evaluation.
    # This must never be fed into the model as a feature.
    is_fraud: bool = False

    # Campaign that generated the fraud.
    # None means legitimate or isolated activity.
    fraud_campaign_id: str | None = None

    # Useful for understanding why a transaction was generated.
    scenario: str = "legitimate"