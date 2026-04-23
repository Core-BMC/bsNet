#!/usr/bin/env python3
"""Run Diffusion-TS imputation for signal recovery experiment.

Executes Condition A (naive) and Condition B (reliability-guided) imputation
using a trained Diffusion-TS model. Uses Diffusion-TS's restore() API with
patched roi_weights support (from patch_diffusion_ts_solver.py).

Prerequisites:
  - Trained Diffusion-TS checkpoint
  - Prepared data (prepare_signal_recovery_data.py)
  - Reliability weights (compute_reliability_weights.py)
  - Patched Diffusion-TS (patch_diffusion_ts_solver.py --apply)

Usage:
  cd external/Diffusion-TS
  PYTHONPATH=.:../../ python3 ../../src/scripts/run_signal_recovery.py \\
      --data-dir ../../data/ds000243/signal_recovery/harvard_oxford \\
      --checkpoint results/signal_recovery/phase1/best_model.pt \\
      --output-dir ../../results/signal_recovery

Architecture notes:
  Diffusion-TS restore() API:
    - Takes a DataLoader of (x, mask) pairs
    - x: [B, seq_len, N_ROI] — full-length tensor, observed portion filled
    - mask: [B, seq_len, N_ROI] — boolean, True where observed
    - Internally calls sample_infill → p_sample_infill → langevin_fn
    - langevin_fn performs Langevin dynamics with gradient guidance:
        infill_loss = (x_start[mask] - target[mask]) ** 2
    - Our patch adds roi_weights to modulate infill_loss per-ROI
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_imputation_dataloader(
    test_short: np.ndarray,
    seq_len: int,
    batch_size: int = 8,
) -> tuple[DataLoader, np.ndarray]:
    """Build DataLoader for Diffusion-TS restore() API.

    Diffusion-TS expects:
      x: [B, seq_len, N_ROI] — padded time series (zeros for missing)
      mask: [B, seq_len, N_ROI] — boolean True for observed timepoints

    Args:
        test_short: [N_test, short_len, N_ROI] observed short scans
        seq_len: target full sequence length
        batch_size: batch size for inference

    Returns:
        (dataloader, mask_array) — DataLoader of (x_padded, mask) pairs
    """
    n_test, short_len, n_roi = test_short.shape

    # Pad short scans to full length
    x_padded = np.zeros((n_test, seq_len, n_roi), dtype=np.float32)
    x_padded[:, :short_len, :] = test_short

    # Mask: True where observed
    mask = np.zeros((n_test, seq_len, n_roi), dtype=bool)
    mask[:, :short_len, :] = True

    x_tensor = torch.from_numpy(x_padded)
    mask_tensor = torch.from_numpy(mask)

    dataset = TensorDataset(x_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, mask


def load_trainer(checkpoint_path: str, device: str = "cuda") -> Any:
    """Load Diffusion-TS Trainer from checkpoint.

    NOTE: Must be called from within the Diffusion-TS directory.

    Args:
        checkpoint_path: path to saved model checkpoint
        device: cuda or cpu

    Returns:
        Trainer instance with loaded model
    """
    try:
        from engine.solver import Trainer
    except ImportError as e:
        raise ImportError(
            f"Cannot import Diffusion-TS Trainer: {e}\n"
            "Run from external/Diffusion-TS/ with PYTHONPATH=.:../../"
        )

    # Load checkpoint — the exact loading mechanism depends on
    # how Diffusion-TS saves checkpoints. Adjust as needed.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # If checkpoint is a Trainer state dict:
    if "trainer" in checkpoint:
        trainer = checkpoint["trainer"]
    else:
        # May need to reconstruct Trainer from config + model weights
        logger.warning(
            "Checkpoint format unclear. You may need to adjust loading logic."
        )
        raise NotImplementedError(
            "Checkpoint loading needs adjustment for this Diffusion-TS version. "
            "Check how the model was saved during training."
        )

    return trainer


def run_imputation(
    trainer: Any,
    dataloader: DataLoader,
    shape: tuple[int, int],
    roi_weights: torch.Tensor | None = None,
    n_samples: int = 10,
    sampling_steps: int = 50,
    coef: float = 0.1,
    stepsize: float = 0.1,
) -> np.ndarray:
    """Run Diffusion-TS imputation via restore() API.

    Args:
        trainer: Diffusion-TS Trainer instance
        dataloader: DataLoader of (x_padded, mask) pairs
        shape: (seq_len, n_roi)
        roi_weights: [1, 1, N_ROI] reliability weights (None for Condition A)
        n_samples: number of imputation runs to average
        sampling_steps: number of diffusion sampling steps
        coef: guidance coefficient (default from Diffusion-TS)
        stepsize: Langevin step size (default from Diffusion-TS)

    Returns:
        [N_test, seq_len, N_ROI] imputed time series (mean of n_samples runs)
    """
    all_samples = []

    for run_idx in range(n_samples):
        logger.info(f"  Imputation run {run_idx + 1}/{n_samples}")

        # Call patched restore() with roi_weights
        samples, reals, masks = trainer.restore(
            raw_dataloader=dataloader,
            shape=shape,
            coef=coef,
            stepsize=stepsize,
            sampling_steps=sampling_steps,
            roi_weights=roi_weights,  # None for Condition A, tensor for B
        )
        all_samples.append(samples)  # [N_test, seq_len, N_ROI]

    # Average across runs
    imputed = np.mean(all_samples, axis=0)
    return imputed.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Diffusion-TS signal recovery")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Signal recovery data directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained Diffusion-TS checkpoint")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for imputed time series")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of imputation samples per subject")
    parser.add_argument("--sampling-steps", type=int, default=50,
                        help="Diffusion sampling steps (50=fast, 1000=full)")
    parser.add_argument("--coef", type=float, default=0.1,
                        help="Guidance coefficient")
    parser.add_argument("--stepsize", type=float, default=0.1,
                        help="Langevin step size")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-naive", action="store_true",
                        help="Skip Condition A (naive imputation)")
    parser.add_argument("--skip-guided", action="store_true",
                        help="Skip Condition B (reliability-guided)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    test_short = np.load(data_dir / "test_short.npy")
    with open(data_dir / "split_info.json") as f:
        split_info = json.load(f)

    seq_len = split_info["seq_len"]
    n_test, short_len, n_roi = test_short.shape
    shape = (seq_len, n_roi)
    logger.info(f"Test: {n_test} subjects, {short_len} → {seq_len} TRs, {n_roi} ROIs")

    # Build DataLoader
    dataloader, mask_arr = build_imputation_dataloader(
        test_short, seq_len, batch_size=args.batch_size
    )

    # Load trained model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    trainer = load_trainer(args.checkpoint, args.device)

    # --- Condition A: Naive imputation (roi_weights=None) ---
    if not args.skip_naive:
        logger.info("=== Condition A: Naive imputation (uniform weights) ===")
        imputed_a = run_imputation(
            trainer, dataloader, shape,
            roi_weights=None,
            n_samples=args.n_samples,
            sampling_steps=args.sampling_steps,
            coef=args.coef,
            stepsize=args.stepsize,
        )
        out_a = output_dir / "imputed_naive.npy"
        np.save(out_a, imputed_a)
        logger.info(f"Saved: {out_a} {imputed_a.shape}")

    # --- Condition B: Reliability-guided imputation ---
    if not args.skip_guided:
        weights_path = data_dir / "test_reliability_weights.npy"
        if not weights_path.exists():
            logger.error(f"Reliability weights not found: {weights_path}")
            logger.error("Run compute_reliability_weights.py first.")
            return

        rel_weights = np.load(weights_path)  # [N_test, N_ROI]
        logger.info(f"Reliability weights: {rel_weights.shape}, "
                     f"mean={rel_weights.mean():.3f}, "
                     f"range=[{rel_weights.min():.3f}, {rel_weights.max():.3f}]")

        # Convert to tensor: [1, 1, N_ROI] — broadcast over batch and time
        # Note: We use the mean across test subjects for uniform weighting
        # (each subject's weights are similar since same atlas)
        mean_weights = rel_weights.mean(axis=0)  # [N_ROI]
        roi_weights_tensor = torch.from_numpy(mean_weights).float()
        roi_weights_tensor = roi_weights_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, N_ROI]
        roi_weights_tensor = roi_weights_tensor.to(args.device)

        logger.info("=== Condition B: Reliability-guided imputation ===")
        imputed_b = run_imputation(
            trainer, dataloader, shape,
            roi_weights=roi_weights_tensor,
            n_samples=args.n_samples,
            sampling_steps=args.sampling_steps,
            coef=args.coef,
            stepsize=args.stepsize,
        )
        out_b = output_dir / "imputed_guided.npy"
        np.save(out_b, imputed_b)
        logger.info(f"Saved: {out_b} {imputed_b.shape}")

    logger.info("\nDone. Run eval_signal_recovery.py for evaluation.")


if __name__ == "__main__":
    main()
