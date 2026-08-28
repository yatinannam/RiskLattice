"""Hardened synthetic dataset generator (Phase 5.5).

Produces ``data/samples/transactions_hardened.csv`` — a harder, deterministic
synthetic dataset designed to stress-test RiskLattice:

  A. baseline legitimate traffic + baseline fraud
  B. low-signal coordinated fraud (individual transactions look normal)
  C. legitimate shared infrastructure (office/university/household/business)
  D. legitimate bursts
  E. mixed fraud/legitimate entity relationships
  F. slow / periodic / multi-stage fraud campaign styles

The baseline generator (``generate_dataset.py``) is untouched and remains
byte-reproducible with SEED=42. This module uses its own fixed seed (2026).

Ground-truth fields (is_fraud, fraud_campaign_id, scenario) are evaluation-only.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.schemas.transaction import (
    PaymentMethod,
    Transaction,
    TransactionStatus,
)

HARDENED_SEED = 2026
HARDENED_OUTPUT = PROJECT_ROOT / "data" / "samples" / "transactions_hardened.csv"

NUM_TRANSACTIONS = 12_000
NUM_USERS = 2_400
NUM_DEVICES = 1_200
NUM_IPS = 1_800
NUM_PAYMENT_INSTRUMENTS = 3_000

MERCHANT_ID = "MERCH_001"
START_TIME = datetime(2026, 8, 1, 0, 0, 0)

LEGIT_RATIO = 0.70       # baseline legitimate traffic
FRAUD_RATIO = 0.20       # low-signal + mixed-entity fraud
SHARED_RATIO = 0.10      # legitimate shared infrastructure / bursts


def _build_pools():
    users = [f"USER_{i:04d}" for i in range(1, NUM_USERS + 1)]
    devices = [f"DEV_{i:04d}" for i in range(1, NUM_DEVICES + 1)]
    ips = [f"IP_{i:04d}" for i in range(1, NUM_IPS + 1)]
    pis = [f"PI_{i:04d}" for i in range(1, NUM_PAYMENT_INSTRUMENTS + 1)]
    return users, devices, ips, pis


def _random_timestamp() -> datetime:
    return START_TIME + timedelta(seconds=random.randint(0, 30 * 24 * 60 * 60))


def _normal_amount() -> float:
    band = random.choice([
        random.uniform(100, 1_000),
        random.uniform(1_000, 5_000),
        random.uniform(5_000, 20_000),
    ])
    return round(band, 2)


def _normal_status() -> TransactionStatus:
    return random.choices(
        [TransactionStatus.SUCCESS, TransactionStatus.FAILED,
         TransactionStatus.REFUNDED],
        weights=[88, 8, 4],
        k=1,
    )[0]


def _payment_method():
    return random.choices(
        [PaymentMethod.UPI, PaymentMethod.CARD, PaymentMethod.NETBANKING,
         PaymentMethod.WALLET],
        weights=[45, 35, 15, 5],
        k=1,
    )[0]


_COUNTER = {"n": 0}


def _next_number() -> int:
    _COUNTER["n"] += 1
    return _COUNTER["n"]


def _make_tx(ts, user, dev, ip, pi, amount, status, is_fraud,
             scenario, campaign_id=None) -> Transaction:
    return Transaction(
        transaction_id=f"TXN_{_next_number():06d}",
        timestamp=ts,
        merchant_id=MERCHANT_ID,
        user_id=user,
        device_id=dev,
        ip_id=ip,
        payment_instrument_id=pi,
        amount=round(amount, 2),
        currency="INR",
        payment_method=_payment_method(),
        status=status,
        is_fraud=is_fraud,
        fraud_campaign_id=campaign_id,
        scenario=scenario,
    )


def _stable_device(devices, user_id: str) -> str:
    return devices[(int(user_id.split("_")[1]) * 7) % NUM_DEVICES]


def _legit_tx(users, devices, ips, pis, scenario="legitimate"):
    user = random.choice(users)
    dev = _stable_device(devices, user)
    ip = random.choice(ips[:100]) if random.random() < 0.18 else random.choice(ips)
    pi = random.choice(pis)
    return _make_tx(_random_timestamp(), user, dev, ip, pi,
                    _normal_amount(), _normal_status(), False, scenario)


def _shared_legit(users, devices, ips, pis, n_users, scenario, shared_ip,
                  shared_device=None, per_user=3):
    out = []
    group = random.sample(users, n_users)
    base = _random_timestamp()
    for i, user in enumerate(group):
        for j in range(per_user):
            dev = shared_device or _stable_device(devices, user)
            pi = random.choice(pis)
            out.append(_make_tx(
                base + timedelta(hours=i * 6 + j), user, dev, shared_ip, pi,
                _normal_amount(), _normal_status(), False, scenario,
            ))
    return out


def _generate_legit_burst(users, devices, ips, pis, n=15):
    ip = random.choice(ips[:100])
    group = random.sample(users, n)
    base = _random_timestamp()
    out = []
    for i, user in enumerate(group):
        dev = random.choice(devices)
        pi = random.choice(pis)
        out.append(_make_tx(
            base + timedelta(minutes=i), user, dev, ip, pi,
            _normal_amount(), _normal_status(), False, "legitimate_burst",
        ))
    return out


def _legit_scenario_groups(users, devices, ips, pis):
    """One batch of legitimate shared-infrastructure/burst scenario groups."""
    groups = []
    # Office: 20 users share IP, normal historical spread.
    for _ in range(3):
        groups.append(_shared_legit(users, devices, ips, pis, 20,
                                    "legitimate_shared_office",
                                    random.choice(ips[:50]), per_user=2))
    # University: 15 users share IP + one shared device.
    for _ in range(3):
        groups.append(_shared_legit(users, devices, ips, pis, 15,
                                    "legitimate_shared_university",
                                    random.choice(ips[:50]),
                                    shared_device=random.choice(devices),
                                    per_user=2))
    # Household: 4 users share IP + device.
    for _ in range(4):
        groups.append(_shared_legit(users, devices, ips, pis, 4,
                                    "legitimate_household",
                                    random.choice(ips[:50]),
                                    shared_device=random.choice(devices),
                                    per_user=5))
    # Business: 8 users share IP + payment instrument.
    for _ in range(3):
        group = random.sample(users, 8)
        ip = random.choice(ips[:50])
        pi_shared = random.choice(pis)
        base = _random_timestamp()
        rows = []
        for i, user in enumerate(group):
            rows.append(_make_tx(
                base + timedelta(hours=i), user, _stable_device(devices, user),
                ip, pi_shared, _normal_amount(), _normal_status(), False,
                "legitimate_business",
            ))
        groups.append(rows)
    # Legitimate bursts.
    for _ in range(4):
        groups.append(_generate_legit_burst(users, devices, ips, pis))
    return groups


# ---------------------------------------------------------------------------
# Hardened fraud (low-signal, temporal-patterned)
# ---------------------------------------------------------------------------

def _low_signal_campaign(users, devices, ips, pis, scenario, n_users=5,
                         shared_entity=("device",), n_tx_per_user=(1, 3),
                         pattern="dense"):
    """Fraud campaign whose per-transaction attributes look completely normal.

    pattern:
      dense      -> many tx in a short burst
      slow       -> spread over 5-15 min intervals
      periodic   -> repeat every few minutes, sustained
      multistage -> two active windows separated by a gap
    """
    scenario_tag = scenario.replace("_", "")[:6].upper() or "CAMP"
    campaign_id = f"HARD_{scenario_tag}_{random.randint(1000, 9999)}"
    group = random.sample(users, n_users)
    shared_dev = random.choice(devices) if "device" in shared_entity else None
    shared_ip = random.choice(ips) if "ip" in shared_entity else None
    shared_pi = random.choice(pis) if "payment" in shared_entity else None
    base = _random_timestamp()
    out = []
    for i, user in enumerate(group):
        n_tx = random.randint(*n_tx_per_user)
        for j in range(n_tx):
            dev = shared_dev if shared_dev else _stable_device(devices, user)
            ip = shared_ip if shared_ip else random.choice(ips)
            pi = shared_pi if shared_pi else random.choice(pis)
            if pattern == "slow":
                offset = i * random.randint(300, 900) + j * 15
            elif pattern == "periodic":
                offset = i * random.randint(120, 300) + j * 30
            elif pattern == "multistage":
                gap = random.randint(3, 6) * 3600
                offset = (i * 60 + j * 5) if i % 2 == 0 else (gap + i * 60 + j * 5)
            else:  # dense
                offset = i * 60 + j * random.randint(1, 5)
            out.append(_make_tx(
                base + timedelta(seconds=offset), user, dev, ip, pi,
                _normal_amount(), _normal_status(), True, scenario, campaign_id,
            ))
    return out


def _mixed_entity(users, devices, ips, pis):
    """Shared device with 5 fraud + 3 legitimate users (collateral test)."""
    shared_dev = random.choice(devices)
    shared_ip = random.choice(ips)
    fraud_users = random.sample(users, 5)
    legit_users = random.sample([u for u in users if u not in fraud_users], 3)
    out = []
    base = _random_timestamp()
    for i, user in enumerate(fraud_users):
        pi = random.choice(pis)
        out.append(_make_tx(
            base + timedelta(minutes=i), user, shared_dev, shared_ip, pi,
            _normal_amount(), _normal_status(), True,
            "mixed_entity_campaign", f"MIXED_{random.randint(1000, 9999)}",
        ))
    for j, user in enumerate(legit_users):
        pi = random.choice(pis)
        out.append(_make_tx(
            base + timedelta(hours=6 + j), user, shared_dev, shared_ip, pi,
            _normal_amount(), _normal_status(), False, "legitimate",
        ))
    return out


# Fraud scenario kinds used by the deterministic generator loop.
FRAUD_KINDS = [
    ("low_signal_account_farm", {"shared_entity": ("device",), "pattern": "dense",
                                 "n_tx_per_user": (2, 3), "n_users": 5}),
    ("low_signal_payment_abuse", {"shared_entity": ("payment",), "pattern": "periodic",
                                  "n_tx_per_user": (1, 2), "n_users": 6}),
    ("low_signal_coordinated_burst", {"shared_entity": ("ip",), "pattern": "slow",
                                      "n_tx_per_user": (1, 2), "n_users": 5}),
    ("mixed_entity_campaign", {"shared_entity": ("device", "ip"), "pattern": "multistage",
                               "n_tx_per_user": (2, 3), "n_users": 5}),
    ("slow_coordinated_campaign", {"shared_entity": ("payment", "ip"), "pattern": "slow",
                                   "n_tx_per_user": (1, 2), "n_users": 7}),
]


def generate_hardened_dataset() -> list[Transaction]:
    """Generate the full hardened dataset deterministically (seed 2026)."""
    random.seed(HARDENED_SEED)
    _COUNTER["n"] = 0

    users, devices, ips, pis = _build_pools()
    transactions: list[Transaction] = []

    # A. Baseline legitimate traffic.
    leg_target = int(NUM_TRANSACTIONS * LEGIT_RATIO)
    for _ in range(leg_target):
        transactions.append(_legit_tx(users, devices, ips, pis))

    # C/D. Legitimate shared infrastructure and bursts.
    for group in _legit_scenario_groups(users, devices, ips, pis):
        transactions.extend(group)

    # B/E. Fraud: generate campaigns until ~20% fraud rows (before top-up).
    fraud_target = int(NUM_TRANSACTIONS * FRAUD_RATIO)
    fraud_count = 0
    kind_idx = 0
    while fraud_count < fraud_target:
        scenario, params = FRAUD_KINDS[kind_idx % len(FRAUD_KINDS)]
        kind_idx += 1
        group = _low_signal_campaign(users, devices, ips, pis, scenario,
                                     n_users=params["n_users"],
                                     shared_entity=params["shared_entity"],
                                     n_tx_per_user=params["n_tx_per_user"],
                                     pattern=params["pattern"])
        transactions.extend(group)
        fraud_count += sum(1 for t in group if t.is_fraud)

    # Explicit mixed-entity shared-device relationships (fraud + legit users).
    for _ in range(5):
        transactions.extend(_mixed_entity(users, devices, ips, pis))

    # Top up and trim to the exact target.
    while len(transactions) < NUM_TRANSACTIONS:
        transactions.append(_legit_tx(users, devices, ips, pis))
    if len(transactions) > NUM_TRANSACTIONS:
        transactions = transactions[:NUM_TRANSACTIONS]

    random.shuffle(transactions)
    for index, tx in enumerate(transactions, start=1):
        tx.transaction_id = f"TXN_{index:06d}"

    return transactions


def save_csv(transactions: list[Transaction], output_path=None) -> None:
    import csv

    dest = output_path or HARDENED_OUTPUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(transactions[0].model_dump(mode="json").keys())
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for tx in transactions:
            writer.writerow(tx.model_dump(mode="json"))
    print(f"Saved {len(transactions)} hardened transactions to {dest}")


if __name__ == "__main__":
    dataset = generate_hardened_dataset()
    save_csv(dataset)