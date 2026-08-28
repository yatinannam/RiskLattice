"""Hardening + adversarial evaluation (Phase 5.5).

Stress-tests RiskLattice against a harder deterministic synthetic dataset
(``transactions_hardened.csv``) and compares the Phase-2 transaction-level
baseline against the full lattice (graph + campaign + containment) honestly.
Ground-truth labels are used ONLY for evaluation.
"""