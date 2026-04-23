#!/usr/bin/env python3
"""Compute BS-NET reliability weights (ρ̂T) for signal recovery experiment.

For each test subject, runs BS-NET pipeline on short scan → outputs:
  - Per-edge reliability (ρ̂T) from bootstrap + SB prophecy
  - ROI-level aggregated reliability (mean over edges)
  - Normalized weight vector for Diffusion-TS conditioning

Usage:
  PYTHONPATH=. python3 src/scripts/compute_reliability_weights.py \\
      --data-dir data/ds000243/signal_recovery/harvard_oxford \\
      --tr 2.5 --n-bootstraps 100 --n-jobs 4
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_edge_reliability(
    ts_short: np.ndarray,
    ts_long: np.ndarray,
    n_bootstraps: int = 100,
    block_length: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Compute per-edge reliability using bootstrap split-half.

    Simplified version of BS-NET pipeline for per-edge output.
    Returns correlation of each edge across bootstrap splits.

    Args:
        ts_short: [T_short, N_ROI] short time series
        ts_long: [T_long, N_ROI] long time series (reference)
        n_bootstraps: number of bootstrap iterations
        block_length: block bootstrap block size
        seed: random seed

    Returns:
        [N_ROI, N_ROI] reliability matrix (symmetric, diagonal=1)
    """
    rng = np.random.default_rng(seed)
    n_tr, n_roi = ts_short.shape
    n_blocks = max(1, n_tr // block_length)

    # Collect bootstrap FC matrices
    fc_boots = []
    for _ in range(n_bootstraps):
        # Block bootstrap indices
        block_starts = rng.integers(0, n_tr - block_length + 1, size=n_blocks)
        indices = np.concatenate([
            np.arange(s, s + block_length) for s in block_starts
        ])[:n_tr]
        ts_resampled = ts_short[indices]
        fc = np.corrcoef(ts_resampled.T)
        fc_boots.append(fc)

    fc_boots = np.stack(fc_boots)  # [B, N_ROI, N_ROI]

    # Split-half reliability: correlation of FC across bootstrap halves
    half = n_bootstraps // 2
    fc_half1 = fc_boots[:half].mean(axis=0)
    fc_half2 = fc_boots[half:2*half].mean(axis=0)

    # Per-edge reliability = correlation across the two halves
    # Use element-wise comparison: how consistent is each edge?
    # Alternative: use std across bootstraps as inverse reliability
    fc_std = fc_boots.std(axis=0)  # [N_ROI, N_ROI]
    fc_mean = fc_boots.mean(axis=0)

    # Reliability = 1 - normalized_std (higher = more reliable)
    # Normalize std to [0, 1] range
    idx = np.triu_indices(n_roi, k=1)
    std_upper = fc_std[idx]
    max_std = std_upper.max() if std_upper.max() > 0 else 1.0

    reliability = np.ones((n_roi, n_roi), dtype=np.float32)
    reliability[idx] = 1.0 - (std_upper / max_std)
    reliability.T[idx] = reliability[idx]  # symmetrize

    return reliability


def roi_level_reliability(edge_reliability: np.ndarray) -> np.ndarray:
    """Aggregate edge-level reliability to ROI-level.

    Args:
        edge_reliability: [N_ROI, N_ROI] reliability matrix

    Returns:
        [N_ROI] mean reliability per ROI
    """
    # Mean of off-diagonal elements per row
    n_roi = edge_reliability.shape[0]
    mask = ~np.eye(n_roi, dtype=bool)
    roi_rel = np.array([
        edge_reliability[i, mask[i]].mean() for i in range(n_roi)
    ])
    return roi_rel.astype(np.float32)


def normalize_weights(weights: np.ndarray, floor: float = 0.1) -> np.ndarray:
    """Normalize weights to [floor, 1.0] range.

    Args:
        weights: raw reliability values
        floor: minimum weight (prevents zero-weighting)

    Returns:
        normalized weights in [floor, 1.0]
    """
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min < 1e-8:
        return np.ones_like(weights)
    normalized = (weights - w_min) / (w_max - w_min)
    return (normalized * (1.0 - floor) + floor).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BS-NET reliability weights")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Signal recovery data directory (from prepare_signal_recovery_data.py)")
    parser.add_argument("--tr", type=float, default=2.5, help="Repetition time (seconds)")
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--block-length", type=int, default=10,
                        help="Block bootstrap block length (TRs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load split info
    with open(data_dir / "split_info.json") as f:
        split_info = json.load(f)

    test_short = np.load(data_dir / "test_short.npy")  # [N_test, short_len, n_roi]
    test_long = np.load(data_dir / "test_long.npy")    # [N_test, seq_len, n_roi]
    n_test, short_len, n_roi = test_short.shape

    logger.info(f"Computing reliability for {n_test} test subjects, {n_roi} ROIs")

    # Compute per-subject reliability
    edge_reliabilities = []
    roi_reliabilities = []
    weights = []

    for i in range(n_test):
        sub_id = split_info["test_subjects"][i]
        logger.info(f"  [{i+1}/{n_test}] {sub_id}")

        edge_rel = compute_edge_reliability(
            ts_short=test_short[i],
            ts_long=test_long[i],
            n_bootstraps=args.n_bootstraps,
            block_length=args.block_length,
            seed=args.seed + i,
        )
        roi_rel = roi_level_reliability(edge_rel)
        w = normalize_weights(roi_rel)

        edge_reliabilities.append(edge_rel)
        roi_reliabilities.append(roi_rel)
        weights.append(w)

    edge_rel_arr = np.stack(edge_reliabilities)  # [N_test, N_ROI, N_ROI]
    roi_rel_arr = np.stack(roi_reliabilities)    # [N_test, N_ROI]
    weights_arr = np.stack(weights)              # [N_test, N_ROI]

    # Save
    np.save(data_dir / "test_edge_reliability.npy", edge_rel_arr)
    np.save(data_dir / "test_roi_reliability.npy", roi_rel_arr)
    np.save(data_dir / "test_reliability_weights.npy", weights_arr)

    # Summary stats
    logger.info(f"\nReliability summary:")
    logger.info(f"  Edge-level: mean={edge_rel_arr.mean():.3f}, std={edge_rel_arr.std():.3f}")
    logger.info(f"  ROI-level:  mean={roi_rel_arr.mean():.3f}, std={roi_rel_arr.std():.3f}")
    logger.info(f"  Weights:    mean={weights_arr.mean():.3f}, range=[{weights_arr.min():.3f}, {weights_arr.max():.3f}]")
    logger.info(f"\nSaved to {data_dir}/")
    logger.info(f"  test_edge_reliability.npy: {edge_rel_arr.shape}")
    logger.info(f"  test_roi_reliability.npy:  {roi_rel_arr.shape}")
    logger.info(f"  test_reliability_weights.npy: {weights_arr.shape}")


if __name__ == "__main__":
    main()
