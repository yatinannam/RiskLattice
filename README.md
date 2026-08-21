# RiskLattice

AI-powered fraud **containment** intelligence for merchants.

> **Status: Phase 1 (Foundation + Dataset).** This README documents what is
> implemented today. Later phases (risk model, graph, campaign detection,
> containment, AI investigator, API, web dashboard, evaluation) will extend it
> as they are built. Nothing below is a fabricated metric or a claim of
> production readiness.

---

## What is RiskLattice?

Traditional fraud systems answer: *"Is this transaction fraudulent?"*
RiskLattice focuses on the harder operational question:

> **What is the smallest safe intervention that can contain coordinated fraud
> while minimizing legitimate-customer impact?**

The system is defense-only. It detects suspicious behavior, connects related
entities into an evidence graph, finds coordinated campaigns, and proposes
bounded, explicitly approved containment actions — never autonomous financial
actions, never offensive tooling, and never real card data.

Primary decision outcomes: `ALLOW`, `REVIEW`, `BLOCK`.

---

## Problem

Merchants face coordinated fraud (account farms, payment-instrument abuse,
coordinated bursts, refund abuse) that individual transaction scoring misses.
Blocking everything is safe against fraud but damages legitimate customers;
blocking nothing maximizes revenue but leaks loss. RiskLattice models this
trade-off and recommends the minimum effective intervention.

## Why transaction-level detection is insufficient (the RiskLattice thesis)

A single transaction often looks legitimate. Fraud emerges from *relationships*
and *patterns* — many accounts sharing a device, unusual temporal density, or
one payment instrument reused across accounts. RiskLattice layers a
relationship graph over transaction risk so the decision is network-aware, not
per-row.

## Architecture (target)

```
Raw transaction data -> Feature engineering -> {Transaction risk model,
Relationship graph} -> Campaign detection -> Containment engine ->
AI Investigator -> Merchant recommendation
```

The ML/statistical/graph layers generate the evidence. The AI layer explains,
investigates, summarizes, and recommends **based only on structured evidence**
and must never invent evidence.

---

## Current implementation (Phase 1)

### Dataset

- Deterministic synthetic generator: `data/generators/generate_dataset.py`
- `SEED = 42`, exactly **10,000 transactions**, target **~85% legitimate /
  ~15% fraudulent**.
- Four fraud scenarios:
  - `account_farm`
  - `payment_instrument_abuse`
  - `coordinated_burst`
  - `refund_abuse`
- Includes deliberate legitimate edge cases to prevent one-trivial-feature
  detection: shared IPs behind legitimate multiple users, shared devices,
  high-value purchases, refunds, normal payment failures, repeated purchases.
- Output: `data/samples/transactions.csv`

### Data schema

- `data/schemas/transaction.py` — Pydantic `Transaction` model with strong
  typing for all core fields.
- Ground-truth fields `is_fraud`, `fraud_campaign_id`, `scenario` are
  evaluation/labeling only. **They must never be used as model features**
  (enforced in later phases).

### Tests (Phase 1)

- `tests/test_data_generator.py` — size, class mix, determinism, scenario
  coverage, ground-truth consistency, schema validity, synthetic-only IDs,
  legitimate edge cases, CSV write.
- `tests/test_schema.py` — required fields, defaults, enums, amount bounds,
  ground-truth fields carry eval labels.

---

## Running locally

### Environment

```
cd apps/api
python -m venv .venv

# Windows CMD:
.venv\Scripts\activate

pip install -r apps/api/requirements.txt
```

### Generate the dataset (deterministic with seed 42)

```
python data/generators/generate_dataset.py
```

### Run the tests (Phase 1)

```
pytest
```

## Demo workflow (Phase 1)

1. Generate the dataset (above).
2. Run `pytest` to confirm data quality and reproducibility.

An API, dashboard, graph, and evaluation screens will arrive in later phases.

---

## Known limitations (Phase 1)

- Only the foundation and dataset exist so far. Feature engineering, the
  baseline model, graph engine, campaign detection, containment, AI
  investigator, API routes, and the dashboard are **not yet built**.
- No Razorpay integration or AI-provider integration exists yet; no real
  credentials are used anywhere.
- The dataset is fully synthetic, seeded with `SEED = 42` for deterministic
  reproducibility, and is **not** real Razorpay transaction data.

## Security notes

- No real payment card numbers, CVVs, or credentials are stored. All entity
  IDs are synthetic (`USER_*`, `DEV_*`, `IP_*`, `PI_*`).
- No offensive/evasion/payment-abuse tooling is included or planned.
