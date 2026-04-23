#!/usr/bin/env python3
"""Run Diffusion-TS imputation for signal recovery experiment.

Executes Condition A (naive) and Condition B (reliability-guided) imputation
using a trained Diffusion-TS model. This script wraps Diffusion-TS inference
with BS-NET reliability weighting.

Prerequisites:
  - Trained Diffusion-TS checkpoint (from setup_diffusion_ts.sh → train)
  - Prepared data (prepare_signal_recovery_data.py)
  - Reliability weights (compute_reliability_weights.py)

Usage:
  cd external/Diffusion-TS
  PYTHONPATH=.:../../ python3 ../../src/scripts/run_signal_recovery.py \\
      --data-dir ../../data/ds000243/signal_recovery/harvard_oxford \\
      --checkpoint results/signal_recovery/phase1/best_model.pt \\
      --config Config/diffusion_ts_fmri_phase1.yaml \\
      --output-dir ../../results/signal_recovery
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_diffusion_ts_model(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
) -> Any:
    """Load trained Diffusion-TS model from checkpoint.

    NOTE: This function must be called from within the Diffusion-TS directory
    so that its modules are importable.

    Args:
        config_path: path to YAML config
        checkpoint_path: path to model checkpoint (.pt)
        device: cuda or cpu

    Returns:
        (model, config) tuple
    """
    # Diffusion-TS imports (must be in PYTHONPATH)
    try:
        from Utils.io_utils import load_yaml_config
        from Models.interpretable_diffusion.model_utils import (
            get_model,
        )
    except ImportError as e:
        raise ImportError(
            f"Cannot import Diffusion-TS modules: {e}\n"
            "Ensure you're running from external/Diffusion-TS/ "
            "with PYTHONPATH=.:../../"
        )

    config = load_yaml_config(config_path)
    model = get_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def impute_naive(
    model: Any,
    test_short: np.ndarray,
    seq_len: int,
    n_samples: int = 10,
    device: str = "cuda",
) -> np.ndarray:
    """Condition A: Naive imputation (all ROIs equally weighted).

    Args:
        model: trained Diffusion-TS model
        test_short: [N_test, short_len, N_ROI] observed short scans
        seq_len: target sequence length
        n_samples: number of imputation samples (averaged)
        device: cuda or cpu

    Returns:
        [N_test, seq_len, N_ROI] imputed time series (mean of samples)
    """
    n_test, short_len, n_roi = test_short.shape
    all_imputed = []

    for i in range(n_test):
        if (i + 1) % 5 == 0 or i == 0:
            logger.info(f"  Naive imputation [{i+1}/{n_test}]")

        observed = torch.from_numpy(test_short[i]).float().to(device)  # [short_len, n_roi]

        # Create observation mask: 1 = observed, 0 = to generate
        mask = torch.zeros(seq_len, n_roi, device=device)
        mask[:short_len] = 1.0

        # Pad observed to full length
        x_obs = torch.zeros(seq_len, n_roi, device=device)
        x_obs[:short_len] = observed

        # Generate n_samples and average
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                # Diffusion-TS restore() with uniform weights (all 1s)
                weights = torch.ones(1, 1, n_roi, device=device)
                x_imputed = _guided_imputation(
                    model, x_obs.unsqueeze(0), mask.unsqueeze(0), weights
                )
            samples.append(x_imputed.squeeze(0).cpu().numpy())

        imputed_mean = np.mean(samples, axis=0)  # [seq_len, n_roi]
        all_imputed.append(imputed_mean)

    return np.stack(all_imputed).astype(np.float32)


def impute_reliability_guided(
    model: Any,
    test_short: np.ndarray,
    reliability_weights: np.ndarray,
    seq_len: int,
    n_samples: int = 10,
    device: str = "cuda",
) -> np.ndarray:
    """Condition B: Reliability-guided imputation.

    Reliability weights from BS-NET modulate the reconstruction guidance:
    high-reliability ROIs get stronger observation constraints.

    Args:
        model: trained Diffusion-TS model
        test_short: [N_test, short_len, N_ROI]
        reliability_weights: [N_test, N_ROI] per-ROI weights in [0.1, 1.0]
        seq_len: target sequence length
        n_samples: number of imputation samples
        device: cuda or cpu

    Returns:
        [N_test, seq_len, N_ROI] imputed time series
    """
    n_test, short_len, n_roi = test_short.shape
    all_imputed = []

    for i in range(n_test):
        if (i + 1) % 5 == 0 or i == 0:
            logger.info(f"  Reliability-guided imputation [{i+1}/{n_test}]")

        observed = torch.from_numpy(test_short[i]).float().to(device)
        w_rho = torch.from_numpy(reliability_weights[i]).float().to(device)  # [n_roi]

        mask = torch.zeros(seq_len, n_roi, device=device)
        mask[:short_len] = 1.0

        x_obs = torch.zeros(seq_len, n_roi, device=device)
        x_obs[:short_len] = observed

        # Reshape weights for broadcasting: [1, 1, n_roi]
        weights = w_rho.unsqueeze(0).unsqueeze(0)

        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                x_imputed = _guided_imputation(
                    model, x_obs.unsqueeze(0), mask.unsqueeze(0), weights
                )
            samples.append(x_imputed.squeeze(0).cpu().numpy())

        imputed_mean = np.mean(samples, axis=0)
        all_imputed.append(imputed_mean)

    return np.stack(all_imputed).astype(np.float32)


def _guided_imputation(
    model: Any,
    x_obs: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    guidance_weight: float = 1.0,
) -> torch.Tensor:
    """Reconstruction-guided imputation (Algorithm 1 from Diffusion-TS paper).

    This is the core modification point: the observation consistency loss
    is weighted by BS-NET reliability.

    Args:
        model: Diffusion-TS model with forward/reverse diffusion
        x_obs: [B, seq_len, n_roi] padded observations
        mask: [B, seq_len, n_roi] observation mask (1=observed)
        weights: [B, 1, n_roi] reliability weights per ROI
        guidance_weight: guidance strength multiplier

    Returns:
        [B, seq_len, n_roi] imputed time series

    NOTE: This function needs adaptation to the actual Diffusion-TS API.
          The pseudocode below follows Algorithm 1 from the paper.
          Actual implementation depends on model.restore() signature.
    """
    # ================================================================
    # ADAPTATION POINT: Replace this with actual Diffusion-TS API call
    # ================================================================
    #
    # The key modification vs standard Diffusion-TS imputation:
    #
    # Standard (Condition A):
    #   loss_obs = F.mse_loss(x_hat_obs, x_obs)
    #
    # Reliability-weighted (Condition B):
    #   loss_obs = (weights * (x_hat_obs - x_obs) ** 2).mean()
    #
    # Both use the same model — only the guidance loss changes.
    #
    # Option 1: If model has restore() method
    try:
        # Try Diffusion-TS's built-in imputation with custom weights
        x_imputed = model.restore(
            x_obs,
            mask,
            guidance_weight=guidance_weight,
            roi_weights=weights,  # Custom kwarg — requires patch
        )
        return x_imputed
    except (TypeError, AttributeError):
        pass

    # Option 2: Manual reverse diffusion with guidance
    # This implements Algorithm 1 step by step
    device = x_obs.device
    batch_size, seq_len, n_roi = x_obs.shape
    timesteps = model.timesteps if hasattr(model, "timesteps") else 1000

    # Start from pure noise
    x_t = torch.randn_like(x_obs)

    # Reverse diffusion with guidance
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Predict x_0 from x_t
        x_0_pred = model(x_t, t_tensor)  # model predicts x_0 directly

        # Compute guidance gradient (weighted observation loss)
        x_0_pred_detached = x_0_pred.detach().requires_grad_(True)
        loss_obs = (
            weights * mask * (x_0_pred_detached - x_obs) ** 2
        ).sum() * guidance_weight

        if loss_obs.requires_grad:
            grad = torch.autograd.grad(loss_obs, x_0_pred_detached)[0]
            x_0_pred = x_0_pred - grad

        # Replace observed portion
        x_0_pred = mask * x_obs + (1 - mask) * x_0_pred

        # Compute x_{t-1} from x_0_pred (DDPM posterior)
        if t > 0:
            alpha_t = model.alphas_cumprod[t] if hasattr(model, "alphas_cumprod") else 1.0
            alpha_prev = model.alphas_cumprod[t - 1] if hasattr(model, "alphas_cumprod") else 1.0
            beta_t = 1 - alpha_t / alpha_prev

            noise = torch.randn_like(x_t)
            x_t = (
                torch.sqrt(alpha_prev) * x_0_pred
                + torch.sqrt(1 - alpha_prev) * noise
            )
        else:
            x_t = x_0_pred

    return x_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Diffusion-TS signal recovery")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Signal recovery data directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained Diffusion-TS checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Diffusion-TS config YAML")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for imputed time series")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of imputation samples per subject")
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
    seq_len_info = None
    try:
        import json
        with open(data_dir / "split_info.json") as f:
            seq_len_info = json.load(f)
    except FileNotFoundError:
        pass

    seq_len = seq_len_info["seq_len"] if seq_len_info else 180
    n_test, short_len, n_roi = test_short.shape
    logger.info(f"Test data: {n_test} subjects, {short_len} TRs observed → {seq_len} TRs target, {n_roi} ROIs")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, config = load_diffusion_ts_model(args.config, args.checkpoint, args.device)

    # --- Condition A: Naive ---
    if not args.skip_naive:
        logger.info("=== Condition A: Naive imputation ===")
        imputed_a = impute_naive(
            model, test_short, seq_len,
            n_samples=args.n_samples, device=args.device,
        )
        out_path_a = output_dir / "imputed_naive.npy"
        np.save(out_path_a, imputed_a)
        logger.info(f"Saved: {out_path_a} {imputed_a.shape}")

    # --- Condition B: Reliability-guided ---
    if not args.skip_guided:
        weights_path = data_dir / "test_reliability_weights.npy"
        if not weights_path.exists():
            logger.error(f"Reliability weights not found: {weights_path}")
            logger.error("Run compute_reliability_weights.py first.")
            return

        reliability_weights = np.load(weights_path)
        logger.info(f"Reliability weights: {reliability_weights.shape}, "
                     f"mean={reliability_weights.mean():.3f}")

        logger.info("=== Condition B: Reliability-guided imputation ===")
        imputed_b = impute_reliability_guided(
            model, test_short, reliability_weights, seq_len,
            n_samples=args.n_samples, device=args.device,
        )
        out_path_b = output_dir / "imputed_guided.npy"
        np.save(out_path_b, imputed_b)
        logger.info(f"Saved: {out_path_b} {imputed_b.shape}")

    logger.info("\nDone. Run eval_signal_recovery.py for evaluation.")


if __name__ == "__main__":
    main()
