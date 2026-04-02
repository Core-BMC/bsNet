"""Generate Figure 1: Duration sweep results from ds000243 (WashU resting-state).

Thin adapter over plot_figure1_ds007535.py — patches dataset-specific constants
and TR (2.5 s) while reusing all panel logic.

Dataset:
  ds000243, N=52, TR=2.5s, MNI152NLin6Asym_res-2
  sub-001~014: single-run (480–724 vols), sub-015~052: run-1+run-2 concat (260–266 vols)
  6 atlases × 10 seeds × 8 durations (30–450 s)

Changes vs. ds007535 version:
  - RESULTS_DIR  : data/ds000243/results/
  - CACHE_DIR    : data/ds000243/timeseries_cache/
  - OUTPUT_NAME  : Figure1_ds000243_DurationSweep.png
  - EXEMPLAR_SUB : sub-001  (longest scan, 1200 s, representative r_FC)
  - TR           : 2.5 s (confirmed from NIfTI header get_zooms()[3])
  - CSV prefix   : ds000243_duration_sweep_{atlas}.csv

Usage:
    python src/visualization/plot_figure1_ds000243.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── Add project root to sys.path (same pattern as ds007535 script) ───────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Import base module and patch constants ────────────────────────────────────
import src.visualization.plot_figure1_ds007535 as _base

_base.RESULTS_DIR   = Path("data/ds000243/results")
_base.CACHE_DIR     = Path("data/ds000243/timeseries_cache")
_base.OUTPUT_NAME   = "Figure1_ds000243_DurationSweep.png"
_base.EXEMPLAR_SUB  = "sub-001"

# TR 2.5 s: patch function defaults (default args are evaluated at definition
# time, so we re-wrap rather than reassign)

_orig_rel = _base._load_reliability_matrices
def _load_reliability_matrices_ds000243(
    atlas: str,
    short_sec: int,
    tr: float = 2.5,
    correction_method: str = "fisher_z",
):
    return _orig_rel(atlas, short_sec, tr=tr, correction_method=correction_method)
_base._load_reliability_matrices = _load_reliability_matrices_ds000243  # type: ignore[assignment]

_orig_exemplar = _base._load_exemplar_fc
def _load_exemplar_fc_ds000243(
    atlas: str,
    sub_id: str,
    short_sec: int,
    tr: float = 2.5,
):
    return _orig_exemplar(atlas, sub_id, short_sec, tr=tr)
_base._load_exemplar_fc = _load_exemplar_fc_ds000243  # type: ignore[assignment]

# CSV filename prefix is hardcoded in _load_per_record — override the function
def _load_per_record_ds000243(atlas: str) -> pd.DataFrame:
    """Load per-record (raw) sweep CSV for ds000243.

    Args:
        atlas: Atlas name key.

    Returns:
        DataFrame with sub_id, duration_sec, seed, rho_hat_T, r_fc_raw, etc.
    """
    path = _base.RESULTS_DIR / f"ds000243_duration_sweep_{atlas}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Per-record CSV not found: {path}")
    return pd.read_csv(path)
_base._load_per_record = _load_per_record_ds000243  # type: ignore[assignment]


# ── Entry point ───────────────────────────────────────────────────────────────
# single_atlas_only=True: skip multi-atlas panels D/E/F, show Schaefer 200 only.
# Layout: 2×3 (Row 0: A B C, Row 1: G1 G2 G3), figsize=(18,11).

if __name__ == "__main__":
    _base.plot_figure1(exemplar_sub=_base.EXEMPLAR_SUB, single_atlas_only=True)
