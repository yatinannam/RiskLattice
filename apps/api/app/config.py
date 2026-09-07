"""Application configuration for the RiskLattice API.

Dataset selection is via the RISKLATTICE_DATASET environment variable:
  baseline  -> data/samples/transactions.csv (SEED=42)
  hardened  -> data/samples/transactions_hardened.csv (seed 2026)  [default]

The API deliberately refuses to expose ground-truth labels.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SUPPORTED_DATASETS = ("baseline", "hardened")


class Settings:
    dataset_name: str
    dataset_path: Path

    def __init__(self) -> None:
        self.dataset_name = os.environ.get("RISKLATTICE_DATASET", "hardened")
        if self.dataset_name not in SUPPORTED_DATASETS:
            self.dataset_name = "hardened"
        if self.dataset_name == "baseline":
            self.dataset_path = PROJECT_ROOT / "data" / "samples" / "transactions.csv"
        else:
            self.dataset_path = PROJECT_ROOT / "data" / "samples" / "transactions_hardened.csv"


settings = Settings()