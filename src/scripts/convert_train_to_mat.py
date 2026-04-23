#!/usr/bin/env python3
"""Convert BS-NET training subjects to .mat format for Diffusion-TS.

Diffusion-TS fMRIDataset expects:
  data_root/sim4.mat  with key 'ts' → [T_total, N_features]

This script concatenates all training subjects' full time series into
a single long array and saves as .mat.

Usage:
  PYTHONPATH=. python3 src/scripts/convert_train_to_mat.py \
      --data-dir data/ds000243/signal_recovery/harvard_oxford \
      --cache-dir data/ds000243/timeseries_cache \
      --atlas harvard_oxford
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy import io as sio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert training data to .mat for Diffusion-TS")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Signal recovery data directory (with split_info.json)")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Timeseries cache directory")
    parser.add_argument("--atlas", type=str, default="harvard_oxford")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)

    # Load split info to get training subjects
    with open(data_dir / "split_info.json") as f:
        split_info = json.load(f)

    train_subjects = split_info["train_subjects"]
    logger.info(f"Training subjects: {len(train_subjects)}")

    # Find and concatenate full time series
    atlas_dir = cache_dir / args.atlas
    if not atlas_dir.exists():
        atlas_dir = cache_dir  # flat layout

    all_ts = []
    for sub_id in train_subjects:
        # Try different naming patterns
        candidates = [
            atlas_dir / f"{sub_id}_{args.atlas}.npy",
            atlas_dir / f"{sub_id}_ts.npy",
            atlas_dir / f"{sub_id}.npy",
        ]
        ts_path = None
        for c in candidates:
            if c.exists():
                ts_path = c
                break

        if ts_path is None:
            logger.warning(f"  {sub_id}: not found, skipping")
            continue

        ts = np.load(ts_path).astype(np.float64)
        all_ts.append(ts)
        logger.info(f"  {sub_id}: {ts.shape}")

    # Concatenate: [T1 + T2 + ... + Tn, N_ROI]
    concat_ts = np.concatenate(all_ts, axis=0)
    logger.info(f"\nConcatenated: {concat_ts.shape}")

    # Save as .mat in Diffusion-TS expected format
    mat_dir = data_dir / "diffusion_ts_data"
    mat_dir.mkdir(parents=True, exist_ok=True)
    mat_path = mat_dir / "sim4.mat"

    sio.savemat(str(mat_path), {"ts": concat_ts})
    logger.info(f"Saved: {mat_path}")
    logger.info(f"  Shape: {concat_ts.shape}")
    logger.info(f"  Range: [{concat_ts.min():.4f}, {concat_ts.max():.4f}]")
    logger.info(f"\nUse in config: data_root: {mat_dir}")


if __name__ == "__main__":
    main()
