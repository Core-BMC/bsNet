"""
Component Necessity Analysis for BS-NET.

Leave-one-out experiment that measures the degradation caused by removing
one component at a time from the full BS-NET pipeline. Conditions tested:
- L_full: Full pipeline (baseline)
- L_no_sb: Remove Spearman-Brown scaling
- L_no_lw: Remove Ledoit-Wolf shrinkage
- L_no_boot: Remove bootstrap resampling
- L_no_prior: Remove Bayesian prior
- L_no_atten: Remove attenuation correction entirely
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.core.bootstrap import (
    block_bootstrap_indices,
    correct_attenuation,
    estimate_optimal_block_length,
    fisher_z,
    fisher_z_inv,
    spearman_brown,
)
from src.core.pipeline import compute_split_half_reliability
from src.core.simulate import generate_synthetic_timeseries
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)


class ConditionResult(NamedTuple):
    """Result for a single condition and seed."""

    condition: str
    seed: int
    rho_hat_T: float


def run_full_pipeline(
    short_obs: np.ndarray,
    fc_ground_truth: np.ndarray,
    block_size: int,
    n_bootstraps: int = 50,
) -> float:
    """
    Full BS-NET pipeline (L_full).

    Applies all components: bootstrap resampling, Ledoit-Wolf shrinkage,
    split-half reliability, Bayesian prior, attenuation correction,
    and Spearman-Brown scaling.

    Args:
        short_obs: Short observation time series (n_samples, n_rois).
        fc_ground_truth: Ground truth FC vector (upper triangle).
        block_size: Block size for bootstrap resampling.
        n_bootstraps: Number of bootstrap iterations.

    Returns:
        float: Median of bootstrap distribution (predicted rho).
    """
    t_samples = short_obs.shape[0]
    rho_hat_b = []
    empirical_prior = (0.25, 0.05)  # Default prior

    for _ in range(n_bootstraps):
        # Block bootstrap resampling
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]

        # Compute observed correlation with ground truth
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs_t = np.corrcoef(fc_ground_truth, fc_obs_t)[0, 1]

        # Compute split-half reliability
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=True)

        # Attenuation correction with Bayesian prior
        k = 7.5  # Default k_factor (900 / 120)
        rho_est_T = correct_attenuation(
            r_obs_t,
            0.98,  # reliability_coeff
            r_split_t,
            k=k,
            empirical_prior=empirical_prior,
        )

        # Fisher-z transformation
        rho_hat_b.append(fisher_z(rho_est_T))

    rho_hat_b = np.array(rho_hat_b)
    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


def run_no_sb_pipeline(
    short_obs: np.ndarray,
    fc_ground_truth: np.ndarray,
    block_size: int,
    n_bootstraps: int = 50,
) -> float:
    """
    Remove Spearman-Brown scaling (L_no_sb).

    Everything stays the same except k=1.0 in attenuation correction.

    Args:
        short_obs: Short observation time series (n_samples, n_rois).
        fc_ground_truth: Ground truth FC vector (upper triangle).
        block_size: Block size for bootstrap resampling.
        n_bootstraps: Number of bootstrap iterations.

    Returns:
        float: Median of bootstrap distribution (predicted rho).
    """
    t_samples = short_obs.shape[0]
    rho_hat_b = []
    empirical_prior = (0.25, 0.05)

    for _ in range(n_bootstraps):
        # Block bootstrap resampling
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]

        # Compute observed correlation with ground truth
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs_t = np.corrcoef(fc_ground_truth, fc_obs_t)[0, 1]

        # Compute split-half reliability
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=True)

        # Attenuation correction with k=1.0 (no SB scaling)
        rho_est_T = correct_attenuation(
            r_obs_t,
            0.98,
            r_split_t,
            k=1.0,  # No scaling
            empirical_prior=empirical_prior,
        )

        rho_hat_b.append(fisher_z(rho_est_T))

    rho_hat_b = np.array(rho_hat_b)
    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


def run_no_lw_pipeline(
    short_obs: np.ndarray,
    fc_ground_truth: np.ndarray,
    block_size: int,
    n_bootstraps: int = 50,
) -> float:
    """
    Remove Ledoit-Wolf shrinkage (L_no_lw).

    Use plain np.corrcoef instead of shrinkage estimation.

    Args:
        short_obs: Short observation time series (n_samples, n_rois).
        fc_ground_truth: Ground truth FC vector (upper triangle).
        block_size: Block size for bootstrap resampling.
        n_bootstraps: Number of bootstrap iterations.

    Returns:
        float: Median of bootstrap distribution (predicted rho).
    """
    t_samples = short_obs.shape[0]
    rho_hat_b = []
    empirical_prior = (0.25, 0.05)

    for _ in range(n_bootstraps):
        # Block bootstrap resampling
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]

        # Compute observed correlation without shrinkage
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=False)
        r_obs_t = np.corrcoef(fc_ground_truth, fc_obs_t)[0, 1]

        # Compute split-half reliability without shrinkage
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=False)

        # Attenuation correction with Bayesian prior
        k = 7.5
        rho_est_T = correct_attenuation(
            r_obs_t,
            0.98,
            r_split_t,
            k=k,
            empirical_prior=empirical_prior,
        )

        rho_hat_b.append(fisher_z(rho_est_T))

    rho_hat_b = np.array(rho_hat_b)
    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


def run_no_boot_pipeline(
    short_obs: np.ndarray,
    fc_ground_truth: np.ndarray,
) -> float:
    """
    Remove bootstrap resampling (L_no_boot).

    Use single sample (no resampling), apply all corrections once.

    Args:
        short_obs: Short observation time series (n_samples, n_rois).
        fc_ground_truth: Ground truth FC vector (upper triangle).

    Returns:
        float: Single prediction (no distribution).
    """
    empirical_prior = (0.25, 0.05)

    # Single sample (no bootstrap)
    fc_obs_t = get_fc_matrix(short_obs, vectorized=True, use_shrinkage=True)
    r_obs_t = np.corrcoef(fc_ground_truth, fc_obs_t)[0, 1]

    # Compute split-half reliability
    r_split_t = compute_split_half_reliability(short_obs, use_shrinkage=True)

    # Attenuation correction with Bayesian prior
    k = 7.5
    rho_est_T = correct_attenuation(
        r_obs_t,
        0.98,
        r_split_t,
        k=k,
        empirical_prior=empirical_prior,
    )

    # Fisher-z and inverse
    z = fisher_z(rho_est_T)
    return float(fisher_z_inv(z))


def run_no_prior_pipeline(
    short_obs: np.ndarray,
    fc_ground_truth: np.ndarray,
    block_size: int,
    n_bootstraps: int = 50,
) -> float:
    """
    Remove Bayesian prior (L_no_prior).

    Call correct_attenuation with empirical_prior=None.

    Args:
        short_obs: Short observation time series (n_samples, n_rois).
        fc_ground_truth: Ground truth FC vector (upper triangle).
        block_size: Block size for bootstrap resampling.
        n_bootstraps: Number of bootstrap iterations.

    Returns:
        float: Median of bootstrap distribution (predicted rho).
    """
    t_samples = short_obs.shape[0]
    rho_hat_b = []

    for _ in range(n_bootstraps):
        # Block bootstrap resampling
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]

        # Compute observed correlation with ground truth
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs_t = np.corrcoef(fc_ground_truth, fc_obs_t)[0, 1]

        # Compute split-half reliability
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=True)

        # Attenuation correction without Bayesian prior
        k = 7.5
        rho_est_T = correct_attenuation(
            r_obs_t,
            0.98,
            r_split_t,
            k=k,
            empirical_prior=None,  # No prior
        )

        rho_hat_b.append(fisher_z(rho_est_T))

    rho_hat_b = np.array(rho_hat_b)
    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


def run_no_atten_pipeline(
    short_obs: np.ndarray,
    fc_ground_truth: np.ndarray,
    block_size: int,
    n_bootstraps: int = 50,
) -> float:
    """
    Remove attenuation correction entirely (L_no_atten).

    After bootstrap, compute raw correlation between FC vectors,
    then apply Spearman-Brown to scale the raw correlation.

    Args:
        short_obs: Short observation time series (n_samples, n_rois).
        fc_ground_truth: Ground truth FC vector (upper triangle).
        block_size: Block size for bootstrap resampling.
        n_bootstraps: Number of bootstrap iterations.

    Returns:
        float: Median of bootstrap distribution (predicted rho).
    """
    t_samples = short_obs.shape[0]
    rho_hat_b = []
    k = 7.5

    for _ in range(n_bootstraps):
        # Block bootstrap resampling
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]

        # Compute raw correlation between FC vectors
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_raw = np.corrcoef(fc_ground_truth, fc_obs_t)[0, 1]

        # Apply Spearman-Brown to scale the raw correlation
        r_scaled = spearman_brown(r_raw, k)

        rho_hat_b.append(fisher_z(r_scaled))

    rho_hat_b = np.array(rho_hat_b)
    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


def main() -> None:
    """
    Execute component necessity analysis across all conditions.

    Generates synthetic data, runs all 6 conditions, and saves results to CSV.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parameters
    seeds = [42, 123, 777, 2026, 9999]
    n_rois = 50
    short_duration = 120
    target_duration = 900
    noise_level = 0.25
    ar1 = 0.6
    n_bootstraps = 50

    # Create output directory
    output_dir = Path("artifacts/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Run analysis for each seed
    for seed in seeds:
        logger.info(f"Processing seed {seed}")
        np.random.seed(seed)

        # Generate synthetic data
        # Full 900 samples for ground truth
        full_obs, full_signal = generate_synthetic_timeseries(
            target_duration, n_rois, noise_level=noise_level, ar1=ar1
        )
        # Short 120 samples for observation
        short_obs, _ = generate_synthetic_timeseries(
            short_duration, n_rois, noise_level=noise_level, ar1=ar1
        )

        # Compute ground truth FC from noise-free full signal
        # generate_synthetic_timeseries returns (n_rois, n_samples)
        # get_fc_matrix expects (n_samples, n_rois)
        fc_ground_truth = get_fc_matrix(
            full_signal.T, vectorized=True, use_shrinkage=False
        )

        # Transpose short_obs if needed (should be n_samples, n_rois)
        if short_obs.shape[0] != short_duration:
            short_obs = short_obs.T

        # Estimate block size (same across all conditions)
        block_size = estimate_optimal_block_length(short_obs)
        logger.debug(f"Estimated block size: {block_size}")

        # Run all conditions
        logger.info("Running L_full (full pipeline)")
        rho_full = run_full_pipeline(
            short_obs, fc_ground_truth, block_size, n_bootstraps
        )
        results.append(ConditionResult("L_full", seed, rho_full))

        logger.info("Running L_no_sb (no Spearman-Brown)")
        rho_no_sb = run_no_sb_pipeline(
            short_obs, fc_ground_truth, block_size, n_bootstraps
        )
        results.append(ConditionResult("L_no_sb", seed, rho_no_sb))

        logger.info("Running L_no_lw (no Ledoit-Wolf)")
        rho_no_lw = run_no_lw_pipeline(
            short_obs, fc_ground_truth, block_size, n_bootstraps
        )
        results.append(ConditionResult("L_no_lw", seed, rho_no_lw))

        logger.info("Running L_no_boot (no bootstrap)")
        rho_no_boot = run_no_boot_pipeline(short_obs, fc_ground_truth)
        results.append(ConditionResult("L_no_boot", seed, rho_no_boot))

        logger.info("Running L_no_prior (no Bayesian prior)")
        rho_no_prior = run_no_prior_pipeline(
            short_obs, fc_ground_truth, block_size, n_bootstraps
        )
        results.append(ConditionResult("L_no_prior", seed, rho_no_prior))

        logger.info("Running L_no_atten (no attenuation correction)")
        rho_no_atten = run_no_atten_pipeline(
            short_obs, fc_ground_truth, block_size, n_bootstraps
        )
        results.append(ConditionResult("L_no_atten", seed, rho_no_atten))

    # Save results to CSV
    output_file = output_dir / "component_necessity.csv"
    with open(output_file, "w") as f:
        f.write("condition,seed,rho_hat_T,delta_from_full\n")

        # Compute full values per seed for delta calculation
        full_by_seed = {}
        for result in results:
            if result.condition == "L_full":
                full_by_seed[result.seed] = result.rho_hat_T

        # Write rows with delta
        for result in results:
            delta = result.rho_hat_T - full_by_seed[result.seed]
            f.write(
                f"{result.condition},{result.seed},"
                f"{result.rho_hat_T:.6f},{delta:.6f}\n"
            )

    logger.info(f"Results saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("Component Necessity Analysis Summary")
    print("=" * 80)

    conditions = [
        "L_full",
        "L_no_sb",
        "L_no_lw",
        "L_no_boot",
        "L_no_prior",
        "L_no_atten",
    ]
    print(f"{'Condition':<15} {'Mean ρ':<12} {'Std':<10} {'Δ from Full':<12}")
    print("-" * 80)

    for condition in conditions:
        cond_results = [r.rho_hat_T for r in results if r.condition == condition]
        mean_rho = np.mean(cond_results)
        std_rho = np.std(cond_results)

        full_results = [r.rho_hat_T for r in results if r.condition == "L_full"]
        mean_full = np.mean(full_results)
        delta = mean_rho - mean_full

        print(
            f"{condition:<15} {mean_rho:.4f} ± {std_rho:.4f}   {delta:+.4f}"
        )

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
