#!/usr/bin/env python3
"""Prepare ds000243 data for Diffusion-TS signal recovery experiment.

Phase 1: harvard_oxford atlas (48 ROI, 724 TR, TR=2.5s)
Phase 2: schaefer200 atlas (200 ROI, 266 TR)

Outputs:
  - train.npy: [N_windows, seq_len, n_roi] for Diffusion-TS training
  - test_short.npy: [N_test, short_len, n_roi] observed short scans
  - test_long.npy: [N_test, seq_len, n_roi] reference long scans
  - test_fc_ref.npy: [N_test, n_edges] reference FC (upper triangle)
  - subject_ids.json: train/test split subject IDs

Usage:
  python -m src.scripts.prepare_signal_recovery_data \\
      --atlas harvard_oxford --seq-len 180 --short-len 48 \\
      --n-windows 20 --test-ratio 0.2 --seed 42

  (서버에서) PYTHONPATH=. python3 src/scripts/prepare_signal_recovery_data.py ...
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def discover_subjects(
    cache_dir: str | Path,
    atlas: str,
) -> list[dict[str, Any]]:
    """Find all subject time-series files for a given atlas."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_path}")

    atlas_dir = cache_path / atlas
    if not atlas_dir.exists():
        # try flat layout
        atlas_dir = cache_path

    patterns = [f"sub-*_{atlas}.npy", "sub-*_ts.npy", "sub-*.npy"]
    files: list[Path] = []
    for pat in patterns:
        files = sorted(atlas_dir.glob(pat))
        if files:
            break

    if not files:
        available = sorted({p.parent.name for p in cache_path.rglob("*.npy")})
        raise FileNotFoundError(
            f"No files found for atlas '{atlas}' in {cache_path}. "
            f"Available: {available}"
        )

    subjects = []
    for f in files:
        sub_id = f.stem.split("_")[0]
        subjects.append({"sub_id": sub_id, "ts_path": str(f)})

    logger.info(f"Found {len(subjects)} subjects for atlas={atlas}")
    return subjects


# ---------------------------------------------------------------------------
# FC computation (minimal, no BS-NET dependency)
# ---------------------------------------------------------------------------

def compute_fc_upper_triangle(ts: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation FC and return upper triangle vector.

    Args:
        ts: [T, N_ROI] time series

    Returns:
        [N_edges] upper triangle of correlation matrix
    """
    fc = np.corrcoef(ts.T)  # [N_ROI, N_ROI]
    idx = np.triu_indices_from(fc, k=1)
    return fc[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Windowed sampling for training
# ---------------------------------------------------------------------------

def extract_training_windows(
    ts: np.ndarray,
    seq_len: int,
    n_windows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Extract random windows from a full time series for training.

    Args:
        ts: [T, N_ROI] full time series
        seq_len: window length
        n_windows: number of windows to extract
        rng: random number generator

    Returns:
        [n_windows, seq_len, N_ROI]
    """
    t_total = ts.shape[0]
    if t_total < seq_len:
        raise ValueError(f"Time series length {t_total} < seq_len {seq_len}")

    max_start = t_total - seq_len
    starts = rng.integers(0, max_start + 1, size=n_windows)
    windows = np.stack([ts[s : s + seq_len] for s in starts])
    return windows.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data for signal recovery experiment")
    parser.add_argument("--cache-dir", type=str, default="data/ds000243/timeseries_cache",
                        help="Path to timeseries cache directory")
    parser.add_argument("--atlas", type=str, default="harvard_oxford",
                        help="Atlas name (harvard_oxford for Phase 1, schaefer200 for Phase 2)")
    parser.add_argument("--output-dir", type=str, default="data/ds000243/signal_recovery",
                        help="Output directory for prepared data")
    parser.add_argument("--seq-len", type=int, default=180,
                        help="Target sequence length (TRs) for Diffusion-TS training")
    parser.add_argument("--short-len", type=int, default=48,
                        help="Short scan length (TRs) for imputation test")
    parser.add_argument("--n-windows", type=int, default=20,
                        help="Number of training windows per subject")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of subjects for test set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir) / args.atlas
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Discover subjects ---
    subjects = discover_subjects(args.cache_dir, args.atlas)
    n_test = max(1, int(len(subjects) * args.test_ratio))
    n_train = len(subjects) - n_test

    # Shuffle and split
    indices = rng.permutation(len(subjects))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    logger.info(f"Train: {n_train}, Test: {n_test}")

    # --- Load and check shapes ---
    sample_ts = np.load(subjects[0]["ts_path"])
    n_tr_total, n_roi = sample_ts.shape
    logger.info(f"Atlas={args.atlas}: {n_tr_total} TRs, {n_roi} ROIs")

    if n_tr_total < args.seq_len:
        logger.warning(
            f"Total TRs ({n_tr_total}) < seq_len ({args.seq_len}). "
            f"Adjusting seq_len to {n_tr_total}."
        )
        args.seq_len = n_tr_total

    # --- Training data: windowed samples ---
    train_windows = []
    for i in train_idx:
        ts = np.load(subjects[i]["ts_path"]).astype(np.float64)
        windows = extract_training_windows(ts, args.seq_len, args.n_windows, rng)
        train_windows.append(windows)

    train_data = np.concatenate(train_windows, axis=0)  # [N_total_windows, seq_len, n_roi]
    logger.info(f"Training data shape: {train_data.shape}")

    # --- Test data: short + long + reference FC ---
    test_short = []
    test_long = []
    test_fc_ref = []
    test_ids = []

    for i in test_idx:
        ts = np.load(subjects[i]["ts_path"]).astype(np.float64)
        short = ts[:args.short_len].astype(np.float32)
        long = ts[:args.seq_len].astype(np.float32)
        fc_ref = compute_fc_upper_triangle(ts[:args.seq_len])

        test_short.append(short)
        test_long.append(long)
        test_fc_ref.append(fc_ref)
        test_ids.append(subjects[i]["sub_id"])

    test_short_arr = np.stack(test_short)   # [N_test, short_len, n_roi]
    test_long_arr = np.stack(test_long)     # [N_test, seq_len, n_roi]
    test_fc_ref_arr = np.stack(test_fc_ref) # [N_test, n_edges]

    logger.info(f"Test short: {test_short_arr.shape}, long: {test_long_arr.shape}, FC ref: {test_fc_ref_arr.shape}")

    # --- Save ---
    np.save(out_dir / "train.npy", train_data)
    np.save(out_dir / "test_short.npy", test_short_arr)
    np.save(out_dir / "test_long.npy", test_long_arr)
    np.save(out_dir / "test_fc_ref.npy", test_fc_ref_arr)

    split_info = {
        "atlas": args.atlas,
        "n_roi": n_roi,
        "n_tr_total": n_tr_total,
        "seq_len": args.seq_len,
        "short_len": args.short_len,
        "n_windows_per_subject": args.n_windows,
        "seed": args.seed,
        "train_subjects": [subjects[i]["sub_id"] for i in train_idx],
        "test_subjects": test_ids,
        "train_shape": list(train_data.shape),
        "test_short_shape": list(test_short_arr.shape),
        "test_long_shape": list(test_long_arr.shape),
        "test_fc_ref_shape": list(test_fc_ref_arr.shape),
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    logger.info(f"Saved to {out_dir}/")
    logger.info(f"  train.npy: {train_data.shape}")
    logger.info(f"  test_short.npy: {test_short_arr.shape}")
    logger.info(f"  test_long.npy: {test_long_arr.shape}")
    logger.info(f"  test_fc_ref.npy: {test_fc_ref_arr.shape}")
    logger.info(f"  split_info.json")


if __name__ == "__main__":
    main()
