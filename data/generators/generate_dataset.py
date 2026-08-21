import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# ============================================================
# Project path setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from data.schemas.transaction import (
    PaymentMethod,
    Transaction,
    TransactionStatus,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

NUM_TRANSACTIONS = 10_000

NUM_USERS = 2_000
NUM_DEVICES = 1_000
NUM_IPS = 1_500
NUM_PAYMENT_INSTRUMENTS = 2_500

MERCHANT_ID = "MERCH_001"

START_TIME = datetime(2026, 8, 1, 0, 0, 0)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "transactions.csv"
)

random.seed(SEED)


# ============================================================
# Entity pools
# ============================================================

USERS = [
    f"USER_{i:04d}"
    for i in range(1, NUM_USERS + 1)
]

DEVICES = [
    f"DEV_{i:04d}"
    for i in range(1, NUM_DEVICES + 1)
]

IPS = [
    f"IP_{i:04d}"
    for i in range(1, NUM_IPS + 1)
]

PAYMENT_INSTRUMENTS = [
    f"PI_{i:04d}"
    for i in range(1, NUM_PAYMENT_INSTRUMENTS + 1)
]


# ============================================================
# Helper functions
# ============================================================

def random_timestamp() -> datetime:
    """
    Generate a random timestamp within a 30-day period.
    """

    seconds = random.randint(
        0,
        30 * 24 * 60 * 60,
    )

    return START_TIME + timedelta(
        seconds=seconds
    )


def random_amount() -> float:
    """
    Generate a realistic-looking INR transaction amount.
    """

    amount = random.choice(
        [
            random.uniform(100, 1_000),
            random.uniform(1_000, 5_000),
            random.uniform(5_000, 20_000),
            random.uniform(20_000, 75_000),
        ]
    )

    return round(amount, 2)


def random_payment_method() -> PaymentMethod:
    """
    Generate a payment method using approximate
    real-world-like proportions.
    """

    return random.choices(
        [
            PaymentMethod.UPI,
            PaymentMethod.CARD,
            PaymentMethod.NETBANKING,
            PaymentMethod.WALLET,
        ],
        weights=[
            45,
            35,
            15,
            5,
        ],
        k=1,
    )[0]


def create_transaction(
    transaction_number: int,
    timestamp: datetime,
    user_id: str,
    device_id: str,
    ip_id: str,
    payment_instrument_id: str,
    amount: float,
    status: TransactionStatus,
    is_fraud: bool,
    fraud_campaign_id: str | None,
    scenario: str,
) -> Transaction:

    return Transaction(
        transaction_id=f"TXN_{transaction_number:06d}",
        timestamp=timestamp,
        merchant_id=MERCHANT_ID,
        user_id=user_id,
        device_id=device_id,
        ip_id=ip_id,
        payment_instrument_id=payment_instrument_id,
        amount=round(amount, 2),
        currency="INR",
        payment_method=random_payment_method(),
        status=status,
        is_fraud=is_fraud,
        fraud_campaign_id=fraud_campaign_id,
        scenario=scenario,
    )


# ============================================================
# Legitimate transactions
# ============================================================

def generate_legitimate_transaction(
    transaction_number: int,
) -> Transaction:

    user_id = random.choice(USERS)

    user_number = int(
        user_id.split("_")[1]
    )

    # Most users have a relatively stable device.
    device_id = DEVICES[
        (user_number * 7) % NUM_DEVICES
    ]

    # Some legitimate users share infrastructure.
    #
    # This intentionally creates realistic situations such as:
    # - universities
    # - offices
    # - apartments
    # - mobile networks
    if random.random() < 0.20:
        ip_id = random.choice(IPS[:100])
    else:
        ip_id = random.choice(IPS)

    payment_instrument_id = random.choice(
        PAYMENT_INSTRUMENTS
    )

    status = random.choices(
        [
            TransactionStatus.SUCCESS,
            TransactionStatus.FAILED,
            TransactionStatus.REFUNDED,
        ],
        weights=[
            90,
            7,
            3,
        ],
        k=1,
    )[0]

    return create_transaction(
        transaction_number=transaction_number,
        timestamp=random_timestamp(),
        user_id=user_id,
        device_id=device_id,
        ip_id=ip_id,
        payment_instrument_id=payment_instrument_id,
        amount=random_amount(),
        status=status,
        is_fraud=False,
        fraud_campaign_id=None,
        scenario="legitimate",
    )


# ============================================================
# Fraud Campaign 1
# Account Farm
# ============================================================

def generate_account_farm(
    starting_transaction_number: int,
) -> list[Transaction]:

    campaign_id = (
        f"FARM_{random.randint(1000, 9999)}"
    )

    # Multiple accounts controlled through
    # shared infrastructure.
    device_id = random.choice(DEVICES)
    ip_id = random.choice(IPS)

    payment_instrument_id = random.choice(
        PAYMENT_INSTRUMENTS
    )

    users = random.sample(
        USERS,
        random.randint(5, 12),
    )

    transactions = []

    base_time = random_timestamp()

    for user_index, user_id in enumerate(users):

        transaction_count = random.randint(2, 5)

        for transaction_index in range(
            transaction_count
        ):

            transaction_number = (
                starting_transaction_number
                + len(transactions)
            )

            timestamp = (
                base_time
                + timedelta(
                    minutes=(
                        user_index * 3
                        + transaction_index
                    )
                )
            )

            transactions.append(
                create_transaction(
                    transaction_number=transaction_number,
                    timestamp=timestamp,
                    user_id=user_id,
                    device_id=device_id,
                    ip_id=ip_id,
                    payment_instrument_id=payment_instrument_id,
                    amount=random.uniform(
                        2_000,
                        15_000,
                    ),
                    status=TransactionStatus.SUCCESS,
                    is_fraud=True,
                    fraud_campaign_id=campaign_id,
                    scenario="account_farm",
                )
            )

    return transactions


# ============================================================
# Fraud Campaign 2
# Payment Instrument Abuse
# ============================================================

def generate_payment_instrument_abuse(
    starting_transaction_number: int,
) -> list[Transaction]:

    campaign_id = (
        f"PAY_{random.randint(1000, 9999)}"
    )

    payment_instrument_id = random.choice(
        PAYMENT_INSTRUMENTS
    )

    users = random.sample(
        USERS,
        random.randint(5, 10),
    )

    transactions = []

    base_time = random_timestamp()

    for user_index, user_id in enumerate(users):

        transaction_number = (
            starting_transaction_number
            + len(transactions)
        )

        transactions.append(
            create_transaction(
                transaction_number=transaction_number,
                timestamp=(
                    base_time
                    + timedelta(
                        minutes=user_index * 4
                    )
                ),
                user_id=user_id,
                device_id=random.choice(DEVICES),
                ip_id=random.choice(IPS),
                payment_instrument_id=payment_instrument_id,
                amount=random.uniform(
                    1_500,
                    12_000,
                ),
                status=TransactionStatus.SUCCESS,
                is_fraud=True,
                fraud_campaign_id=campaign_id,
                scenario="payment_instrument_abuse",
            )
        )

    return transactions


# ============================================================
# Fraud Campaign 3
# Coordinated Burst
# ============================================================

def generate_coordinated_burst(
    starting_transaction_number: int,
) -> list[Transaction]:

    campaign_id = (
        f"BURST_{random.randint(1000, 9999)}"
    )

    users = random.sample(
        USERS,
        random.randint(10, 20),
    )

    # Shared IP across the burst.
    ip_id = random.choice(IPS)

    transactions = []

    base_time = random_timestamp()

    for index, user_id in enumerate(users):

        transaction_number = (
            starting_transaction_number
            + len(transactions)
        )

        transactions.append(
            create_transaction(
                transaction_number=transaction_number,
                timestamp=(
                    base_time
                    + timedelta(
                        seconds=(
                            index
                            * random.randint(
                                5,
                                20,
                            )
                        )
                    )
                ),
                user_id=user_id,
                device_id=random.choice(DEVICES),
                ip_id=ip_id,
                payment_instrument_id=random.choice(
                    PAYMENT_INSTRUMENTS
                ),
                amount=random.uniform(
                    4_000,
                    6_000,
                ),
                status=TransactionStatus.SUCCESS,
                is_fraud=True,
                fraud_campaign_id=campaign_id,
                scenario="coordinated_burst",
            )
        )

    return transactions


# ============================================================
# Fraud Campaign 4
# Refund Abuse
# ============================================================

def generate_refund_abuse(
    starting_transaction_number: int,
) -> list[Transaction]:

    campaign_id = (
        f"REFUND_{random.randint(1000, 9999)}"
    )

    user_id = random.choice(USERS)
    device_id = random.choice(DEVICES)
    ip_id = random.choice(IPS)

    payment_instrument_id = random.choice(
        PAYMENT_INSTRUMENTS
    )

    transactions = []

    base_time = random_timestamp()

    for index in range(
        random.randint(5, 10)
    ):

        transaction_number = (
            starting_transaction_number
            + len(transactions)
        )

        if index % 2 == 1:
            status = TransactionStatus.REFUNDED
        else:
            status = TransactionStatus.SUCCESS

        transactions.append(
            create_transaction(
                transaction_number=transaction_number,
                timestamp=(
                    base_time
                    + timedelta(
                        hours=index * 2
                    )
                ),
                user_id=user_id,
                device_id=device_id,
                ip_id=ip_id,
                payment_instrument_id=payment_instrument_id,
                amount=random.uniform(
                    1_000,
                    8_000,
                ),
                status=status,
                is_fraud=True,
                fraud_campaign_id=campaign_id,
                scenario="refund_abuse",
            )
        )

    return transactions


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset() -> list[Transaction]:
    # Reset the PRNG so that every call (including repeated calls within one
    # process) produces the exact same deterministic dataset for SEED = 42.
    random.seed(SEED)

    transactions: list[Transaction] = []

    transaction_number = 1

    # --------------------------------------------------------
    # 85% legitimate baseline
    # --------------------------------------------------------

    legitimate_target = int(
        NUM_TRANSACTIONS * 0.85
    )

    for _ in range(
        legitimate_target
    ):

        transactions.append(
            generate_legitimate_transaction(
                transaction_number
            )
        )

        transaction_number += 1

    # --------------------------------------------------------
    # Generate coordinated fraud campaigns
    # --------------------------------------------------------

    campaign_generators = [
        generate_account_farm,
        generate_payment_instrument_abuse,
        generate_coordinated_burst,
        generate_refund_abuse,
    ]

    while len(transactions) < NUM_TRANSACTIONS:

        generator = random.choice(
            campaign_generators
        )

        campaign_transactions = generator(
            transaction_number
        )

        remaining = (
            NUM_TRANSACTIONS
            - len(transactions)
        )

        transactions.extend(
            campaign_transactions[
                :remaining
            ]
        )

        transaction_number = (
            len(transactions) + 1
        )

    # --------------------------------------------------------
    # Shuffle transactions
    # --------------------------------------------------------

    random.shuffle(
        transactions
    )

    # --------------------------------------------------------
    # Reassign transaction IDs
    # --------------------------------------------------------

    for index, transaction in enumerate(
        transactions,
        start=1,
    ):

        transaction.transaction_id = (
            f"TXN_{index:06d}"
        )

    return transactions


# ============================================================
# CSV export
# ============================================================

def save_csv(
    transactions: list[Transaction],
    output_path: Path | None = None,
) -> None:

    destination = output_path or OUTPUT_PATH

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
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
    ]

    with open(
        destination,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for transaction in transactions:

            writer.writerow(
                transaction.model_dump(
                    mode="json"
                )
            )

    print(
        f"Generated {len(transactions):,} transactions."
    )

    print(
        f"Saved to: {destination}"
    )


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":

    dataset = generate_dataset()

    save_csv(dataset)