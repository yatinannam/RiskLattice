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

### Build features + train the baseline (Phase 2)

```
python ml/features/build_features.py
python ml/training/train_baseline.py --model logistic_regression
python ml/training/train_baseline.py --model random_forest
```

### Evaluate on the held-out split (Phase 2)

```
python ml/evaluation/evaluate.py --model logistic_regression
python ml/evaluation/evaluate.py --model random_forest
```

Artifacts (model pipelines, split metadata, thresholds, metrics JSON) are
written under `ml/artifacts/` (git-ignored).

### Run the tests

```
pytest
```

### Build the relationship graph + candidate campaigns (Phase 3)

```
python engine/graph/graph_builder.py
python engine/graph/graph_features.py
python engine/graph/experiment.py
python engine/graph/campaign_detector.py
```

`experiment.py` prints graph statistics, high-connectivity entities, and
evidence examples. `campaign_detector.py` loads the Phase-2 model scores and
prints candidate campaign structures.

### Campaign intelligence + risk scoring (Phase 4)

```
python engine/risk/risk_engine.py
python engine/risk/experiment.py
```

`experiment.py` produces the Phase-4 report: candidate/assessed counts,
risk-level distribution, exposure, fraud/legitimate coverage, campaign
precision/recall, and Phase-2 false-negative recovery analysis.

### Containment optimizer (Phase 5)

```
python engine/containment/experiment.py
```

`experiment.py` runs the simulated-containment optimizer over all assessed
campaigns and writes `ml/artifacts/containment_experiment.json` — strategy
distribution, average containment/collateral, NO_SAFE_ACTION count, the
"block everything vs RiskLattice heuristic" comparison, and a ground-truth
evaluation of the recommended strategies.

## Hardening + adversarial evaluation (Phase 5.5)

Generate the harder dataset (separate from the unchanged baseline):

```
python data/generators/generate_hardened.py
```

Run the evaluation:

```
python engine/hardening/experiment.py
python engine/hardening/scenario_report.py
```

Writes `ml/artifacts/hardening_report.json` (baseline metrics + lattice
recovery/coverage) and `ml/artifacts/hardening_scenario_report.json` (per
scenario: `baseline_recall`, `campaign_detection_rate`, `baseline_false_negatives`).

## Demo workflow

1. Generate the dataset(s) (baseline + optional hardened).
2. Build features and train one or both baselines.
3. Evaluate on the held-out test split and review `ml/artifacts/*_metrics.json`.
4. Build the graph and inspect candidate campaigns (`experiment.py`,
   `campaign_detector.py`).
5. Assess campaigns (`engine/risk/experiment.py`) and inspect the ranked
   assessments / evidence.
6. Run containment (`engine/containment/experiment.py`) and review the
   recommended strategies vs. the "block everything" baseline.
7. Run the hardening evaluation (`engine/hardening/experiment.py`).
8. Run `pytest` to confirm data quality, leakage guards, and reproducibility.

## Phase 6 status

Implemented: a grounded **AI investigation** layer (`engine/investigator/`).
`InvestigationEvidence` packages deterministic risk/graph/containment data;
`InvestigatorProvider` (mock, no API key) generates a reproducible report with
FACT/INFERENCE/UNCERTAINTY findings, each citing evidence IDs; `validate_report`
is a hard hallucination guard (rejects unknown IDs / off-package numbers);
`investigate_campaign` runs the full workflow end-to-end. It never executes
containment and never makes the final fraud determination. Run it with:

```
python engine/investigator/experiment.py
```

Output: `ml/artifacts/investigator_report.json`.

## Phase 5.5 status

Implemented: a separate deterministic **hardened** dataset
(`transactions_hardened.csv`, seed 2026, 12,000 rows) with low-signal fraud,
legitimate shared infrastructure, legitimate bursts, and mixed fraud/legit
entities; plus an honest adversarial evaluation (`engine/hardening/`). The
baseline dataset stays byte-identical (SEED=42). Findings are reported as-is:
on the hardened set the fresh baseline reached ROC-AUC ≈ 0.955 / recall ≈ 0.82,
produced **89 test false negatives**, and the lattice recovered only **2
(2.25%)** via high-risk campaigns — an honest, limited result, not a claim of
superiority.

## Phase 5 status

Implemented: a simulated containment optimizer (`engine/containment/optimizer.py`)
over Phase-4 campaign assessments. It generates candidate actions from the
campaign's own entities, simulates each against the **whole dataset**
(so legitimate collateral outside the campaign is counted), evaluates bounded
combinations (max 3 actions, top-10 entities), enforces constraints
(max 5 legit users, min 70% fraud containment, max 3 actions), removes
dominated strategies, and either recommends the best strategy or returns
`NO_SAFE_ACTION` (never forcing an unsafe block). Every recommendation produces
an audit record with `execution_status: SIMULATED` and `approval_required`.

## Phase 4 status

Implemented: a deterministic campaign risk engine (`engine/risk/risk_engine.py`)
over Phase-3 candidate campaigns. Each assessment carries a transparent 0–100
score ("transparent heuristic campaign score") blending five documented 0..1
dimensions — transaction, relationship, temporal, concentration, behavioral —
plus a **confidence** separate from risk, a documented risk-level band
(LOW/MEDIUM/HIGH/CRITICAL), and structured evidence items referencing only real
entities/transactions. Deduplication (Jaccard overlap), deterministic ranking
with filters, ground-truth evaluation (dedicated eval API only), and Phase-2
false-negative diagnostics are included. The exact formulas are in
`docs/architecture/architecture.md`.

## Phase 3 status

Implemented: a typed, temporal-aware relationship graph
(`engine/graph/graph_builder.py`) over the 10,000-transaction dataset —
16,897 nodes / 41,165 edges across USER/DEVICE/IP/PAYMENT_INSTRUMENT/
TRANSACTION/MERCHANT, with aggregate edges carrying
`relationship_type`, `first_seen`, `last_seen`, `transaction_count`.
Graph features (`graph_features.py`) compute degree, users-per-entity,
transactions-per-entity, connected components, relationship density, and
past-only 5m/1h/24h window helpers. Structured evidence extraction
(`extract_graph_evidence`) and subgraph extraction are available.
`campaign_detector.py` finds **candidate** campaigns from Phase-2 risk scores +
shared entities + temporal proximity (ground-truth fields never used).

## Phase 2 status

Implemented: past-only (leakage-free) feature engineering (38 features),
chronological 80/20 split, Logistic Regression + Random Forest baselines,
threshold selection on training out-of-fold predictions, held-out evaluation
with precision/recall/F1/ROC-AUC/PR-AUC, confusion counts, and a documented
**demo cost model** (INR 500 / false positive, INR 2500 / false negative).

**Explicit limitation:** the baseline is transaction-level only. It does not
yet understand fraud campaigns, graph relationships, connected components,
coordinated entities, or containment (Phases 3–5). Whether network-aware
detection improves decision quality is measured honestly in Phase 9.

---

## Known limitations (current)

- The **full API routes** and **dashboard/frontend** are **not yet built**.
  Phase 6 delivers the AI investigation layer (mock provider, validated);
  no real payment action is ever executed, and the investigator never executes
  containment.
- Real LLM provider adapters (OpenAI/Anthropic) are **not wired up** — the
  application runs offline with the deterministic mock provider, and any
  requested provider name falls back to mock (documented in
  `engine/investigator/`).
- **Phase-5.5 honest finding:** on the hardened dataset the lattice recovered
  only **2 of 89 (2.25%)** baseline test false negatives via high-risk
  campaigns, and fraud-transaction coverage of high-risk campaigns was ~12.7%.
  The lattice does NOT clearly beat the transaction-level baseline on this
  low-signal set; the lattice's clearest measured value remains **collateral
  reduction** in containment (Phase 5) and **safe refusal** (NO_SAFE_ACTION).
- On the easy baseline dataset, false-negative recovery was likewise measured
  as 0 (109 FNs, of which 13 had risk ≥ 0.5, none inside any candidate).
- The current synthetic fraud is relatively separable at the transaction level
  on the Phase-1 dataset (high ROC-AUC), which is why the lattice's marginal
  value is small there; the hardened set is harder but still synthetic.
- The containment optimizer is a documented **bounded heuristic**, not a
  provably optimal solver.
- No Razorpay integration or AI-provider integration exists yet; no real
  credentials are used anywhere.
- The datasets are fully synthetic (SEED=42 baseline, seed-2026 hardened) and
  are **not** real Razorpay transaction data.

## Security notes

- No real payment card numbers, CVVs, or credentials are stored. All entity
  IDs are synthetic (`USER_*`, `DEV_*`, `IP_*`, `PI_*`).
- No offensive/evasion/payment-abuse tooling is included or planned.
