#!/usr/bin/env python3
"""Run Diffusion-TS imputation for BS-NET signal recovery experiment.

Loads trained Diffusion-TS model and runs imputation on our test subjects
for Condition A (naive, uniform weights) and Condition B (reliability-guided).

Must be run from external/Diffusion-TS directory.

Usage:
  cd external/Diffusion-TS
  PYTHONPATH=. python3 ../../src/scripts/run_signal_recovery.py \
      --config ../../configs/diffusion_ts_fmri_phase1.yaml \
      --data-dir ../../data/ds000243/signal_recovery/harvard_oxford \
      --milestone 10 \
      --output-dir ../../results/signal_recovery \
      --gpu 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Diffusion-TS normalization helpers (copied from their codebase)
def normalize_to_neg_one_to_one(x: np.ndarray) -> np.ndarray:
    return x * 2.0 - 1.0

def unnormalize_to_zero_to_one(x: np.ndarray) -> np.ndarray:
    return (x + 1.0) * 0.5


def build_test_dataloader(
    test_short: np.ndarray,
    seq_len: int,
    scaler: MinMaxScaler,
    batch_size: int = 8,
    neg_one_to_one: bool = True,
) -> DataLoader:
    """Build DataLoader matching Diffusion-TS restore() API.

    restore() expects (x, t_m) pairs:
      x: [B, seq_len, N_ROI] — full normalized data (zeros for missing)
      t_m: [B, seq_len, N_ROI] — boolean mask (True = observed)

    Args:
        test_short: [N_test, short_len, N_ROI] raw test data
        seq_len: target sequence length
        scaler: fitted MinMaxScaler from training data
        batch_size: batch size
        neg_one_to_one: if True, scale to [-1, 1]

    Returns:
        DataLoader yielding (x_normalized, mask) pairs
    """
    n_test, short_len, n_roi = test_short.shape

    # Pad to full sequence length
    x_padded = np.zeros((n_test, seq_len, n_roi), dtype=np.float64)
    x_padded[:, :short_len, :] = test_short

    # Normalize: MinMaxScaler per feature, then optional [-1, 1]
    # scaler expects [N, n_features], so reshape
    for i in range(n_test):
        x_padded[i] = scaler.transform(x_padded[i])

    if neg_one_to_one:
        x_padded = normalize_to_neg_one_to_one(x_padded)

    # Mask: True for observed timepoints
    mask = np.zeros((n_test, seq_len, n_roi), dtype=bool)
    mask[:, :short_len, :] = True

    x_tensor = torch.from_numpy(x_padded).float()
    mask_tensor = torch.from_numpy(mask).bool()

    dataset = TensorDataset(x_tensor, mask_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def fit_scaler(data_dir: Path) -> MinMaxScaler:
    """Fit MinMaxScaler on training data (same as Diffusion-TS does).

    Reads the concatenated .mat file and fits scaler.
    """
    from scipy import io as sio

    mat_path = data_dir / "diffusion_ts_data" / "sim4.mat"
    if mat_path.exists():
        raw = sio.loadmat(str(mat_path))["ts"]
    else:
        # Fallback: use train.npy (reshape to 2D)
        train = np.load(data_dir / "train.npy")
        raw = train.reshape(-1, train.shape[-1])

    scaler = MinMaxScaler()
    scaler.fit(raw)
    return scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run signal recovery imputation")
    parser.add_argument("--config", type=str, required=True,
                        help="Diffusion-TS config YAML")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Signal recovery data directory")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--milestone", type=int, default=10,
                        help="Checkpoint milestone to load")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Number of imputation runs to average")
    parser.add_argument("--sampling-steps", type=int, default=250)
    parser.add_argument("--coef", type=float, default=1e-2,
                        help="Guidance coefficient (from config test_dataset)")
    parser.add_argument("--stepsize", type=float, default=5e-2,
                        help="Langevin step size (from config test_dataset)")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-naive", action="store_true")
    parser.add_argument("--skip-guided", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # --- Load config and create model ---
    from Utils.io_utils import load_yaml_config, instantiate_from_config, seed_everything
    from engine.solver import Trainer
    from engine.logger import Logger
    from Data.build_dataloader import build_dataloader

    seed_everything(42)
    config = load_yaml_config(args.config)

    # Create model
    model = instantiate_from_config(config["model"]).cuda()

    # Create a minimal args namespace for Trainer
    class FakeArgs:
        def __init__(self, name, output_dir):
            self.save_dir = str(output_dir)
            self.name = name
            self.tensorboard = False
            self.output = str(output_dir)
    fake_args = FakeArgs("bsnet_phase1", output_dir)
    os.makedirs(fake_args.save_dir, exist_ok=True)

    # Build training dataloader (needed for Trainer init)
    logger.info("Building training dataloader (for Trainer init)...")
    config["dataloader"]["train_dataset"]["params"]["output_dir"] = fake_args.save_dir
    dataloader_info = build_dataloader(config, fake_args)

    # Create Trainer and load checkpoint
    trainer = Trainer(
        config=config, args=fake_args, model=model,
        dataloader=dataloader_info, logger=None,
    )
    trainer.load(args.milestone)
    logger.info(f"Loaded checkpoint milestone {args.milestone}")

    # --- Prepare test data ---
    test_short = np.load(data_dir / "test_short.npy")
    with open(data_dir / "split_info.json") as f:
        split_info = json.load(f)

    seq_len = split_info["seq_len"]
    n_test, short_len, n_roi = test_short.shape
    shape = [seq_len, n_roi]
    logger.info(f"Test: {n_test} subjects, {short_len} → {seq_len} TRs, {n_roi} ROIs")

    # Fit scaler from training data
    scaler = fit_scaler(data_dir)
    logger.info(f"Scaler fitted: min={scaler.data_min_[:3]}..., max={scaler.data_max_[:3]}...")

    # Build test DataLoader
    test_dl = build_test_dataloader(
        test_short, seq_len, scaler,
        batch_size=args.batch_size, neg_one_to_one=True,
    )

    # --- Condition A: Naive imputation ---
    if not args.skip_naive:
        logger.info("=== Condition A: Naive imputation (uniform weights) ===")
        all_samples_a = []
        for run_idx in range(args.n_runs):
            logger.info(f"  Run {run_idx + 1}/{args.n_runs}")
            samples, reals, masks = trainer.restore(
                test_dl, shape=shape,
                coef=args.coef, stepsize=args.stepsize,
                sampling_steps=args.sampling_steps,
            )
            # Unnormalize: [-1,1] → [0,1] → original scale
            samples = unnormalize_to_zero_to_one(samples)
            samples_orig = scaler.inverse_transform(
                samples.reshape(-1, n_roi)
            ).reshape(samples.shape)
            all_samples_a.append(samples_orig)

        imputed_a = np.mean(all_samples_a, axis=0).astype(np.float32)
        out_a = output_dir / "imputed_naive.npy"
        np.save(out_a, imputed_a)
        logger.info(f"Saved: {out_a} {imputed_a.shape}")

    # --- Condition B: Reliability-guided imputation ---
    if not args.skip_guided:
        weights_path = data_dir / "test_reliability_weights.npy"
        if not weights_path.exists():
            logger.error(f"Reliability weights not found: {weights_path}")
            return

        rel_weights = np.load(weights_path)  # [N_test, N_ROI]
        mean_weights = rel_weights.mean(axis=0)  # [N_ROI]
        roi_weights = torch.from_numpy(mean_weights).float().cuda()
        roi_weights = roi_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, N_ROI]
        logger.info(f"Reliability weights: mean={mean_weights.mean():.3f}, "
                     f"range=[{mean_weights.min():.3f}, {mean_weights.max():.3f}]")

        logger.info("=== Condition B: Reliability-guided imputation ===")
        all_samples_b = []
        for run_idx in range(args.n_runs):
            logger.info(f"  Run {run_idx + 1}/{args.n_runs}")
            samples, reals, masks = trainer.restore(
                test_dl, shape=shape,
                coef=args.coef, stepsize=args.stepsize,
                sampling_steps=args.sampling_steps,
                roi_weights=roi_weights,
            )
            samples = unnormalize_to_zero_to_one(samples)
            samples_orig = scaler.inverse_transform(
                samples.reshape(-1, n_roi)
            ).reshape(samples.shape)
            all_samples_b.append(samples_orig)

        imputed_b = np.mean(all_samples_b, axis=0).astype(np.float32)
        out_b = output_dir / "imputed_guided.npy"
        np.save(out_b, imputed_b)
        logger.info(f"Saved: {out_b} {imputed_b.shape}")

    logger.info("\nImputation complete. Run eval_signal_recovery.py next.")


if __name__ == "__main__":
    main()
