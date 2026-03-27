"""
Sensitivity analysis script for BS-NET hardcoded parameters.

Performs a two-phase parameter sweep to identify how key parameters
(reliability_coeff, observation_var, prior_mean, prior_var) affect
the bootstrap prediction accuracy (predicted_rho) and reliability.

Phase 1: Sweep reliability_coeff × observation_var (prior fixed)
Phase 2: Sweep prior_mean × prior_var (reliability and obs_var fixed)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.core.bootstrap import (
    block_bootstrap_indices,
    estimate_optimal_block_length,
    fisher_z,
    fisher_z_inv,
)
from src.core.simulate import generate_synthetic_timeseries
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)


class SensitivityResult(NamedTuple):
    """Result container for a single sensitivity analysis iteration."""

    seed: int
    param1_name: str
    param1_val: float
    param2_name: str
    param2_val: float
    rho_hat_T: float
    mae: float
    pass_: bool


def compute_split_half_reliability(time_series: np.ndarray) -> float:
    """
    Compute split-half reliability from time series data.

    Splits the time series into two halves, computes functional connectivity
    matrices for each half, and returns their correlation.

    Args:
        time_series: Time series data with shape (n_samples, n_rois).

    Returns:
        float: Correlation between split halves, clipped to [0.001, 0.999].
    """
    n_split = time_series.shape[0] // 2

    fc_split1 = get_fc_matrix(
        time_series[:n_split, :], vectorized=True, use_shrinkage=False
    )
    fc_split2 = get_fc_matrix(
        time_series[n_split:, :], vectorized=True, use_shrinkage=False
    )

    r_split = np.corrcoef(fc_split1, fc_split2)[0, 1]
    r_split = np.clip(r_split, 0.001, 0.999)

    return r_split


def run_simplified_bootstrap_prediction(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    reliability_coeff: float,
    empirical_prior: tuple[float, float],
    observation_var: float,
    k_factor: float,
    n_bootstraps: int = 30,
) -> float:
    """
    Simplified bootstrap prediction loop for sensitivity analysis.

    Mirrors the pipeline.py logic but accepts custom parameter values
    without modifying the config globally.

    Args:
        short_obs: Short observation time series (n_samples, n_rois).
        fc_reference: Reference functional connectivity vector.
        reliability_coeff: Assumed reliability coefficient.
        empirical_prior: Tuple of (mean, variance) for Bayesian prior.
        observation_var: Observation variance for Bayesian weighting.
        k_factor: Scaling factor for Spearman-Brown correction.
        n_bootstraps: Number of bootstrap iterations.

    Returns:
        float: Predicted correlation (median of bootstrap distribution).
    """
    t_samples = short_obs.shape[0]

    # Estimate optimal block length
    block_size = estimate_optimal_block_length(short_obs)

    rho_hat_b = []

    for _b in range(n_bootstraps):
        # Block bootstrap resampling
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]

        # Compute observed correlation
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=False)
        r_obs_t = np.corrcoef(fc_reference, fc_obs_t)[0, 1]

        # Compute split-half reliability
        r_split_t = compute_split_half_reliability(ts_b)

        # Custom attenuation correction with modified observation_var
        prior_mean, prior_var = empirical_prior
        weight = prior_var / (prior_var + observation_var)
        r_real_t_adjusted = weight * r_split_t + (1 - weight) * prior_mean

        # Apply attenuation correction with custom reliability
        min_rel = 0.05
        r_hat_t = max(reliability_coeff, min_rel)
        r_real_t = max(r_real_t_adjusted, min_rel)

        from src.core.bootstrap import spearman_brown

        r_true_t = r_obs_t / np.sqrt(r_hat_t * r_real_t)

        r_hat_T = spearman_brown(r_hat_t, k_factor)
        r_real_T = spearman_brown(r_real_t, k_factor)

        rho_est_T = r_true_t * np.sqrt(r_hat_T * r_real_T)
        rho_est_T = np.clip(rho_est_T, -1.0, 1.0)

        # Fisher-z transformation
        rho_hat_b.append(fisher_z(rho_est_T))

    rho_hat_b = np.array(rho_hat_b)

    # Point estimate as median
    rho_hat_T_z = np.nanmedian(rho_hat_b)
    predicted_rho = fisher_z_inv(rho_hat_T_z)

    return float(predicted_rho)


def run_phase1(
    output_path: Path,
    n_rois: int = 50,
    short_samples: int = 120,
    target_samples: int = 900,
    noise_level: float = 0.25,
    ar1: float = 0.6,
    n_bootstraps: int = 30,
    seeds: list[int] | None = None,
) -> list[SensitivityResult]:
    """
    Phase 1: Sweep reliability_coeff × observation_var.

    Keeps empirical_prior fixed at (0.25, 0.05).

    Args:
        output_path: Path to save CSV results.
        n_rois: Number of regions of interest.
        short_samples: Number of samples in short observation.
        target_samples: Number of samples in target observation.
        noise_level: Noise level for synthetic data.
        ar1: AR(1) coefficient for synthetic data.
        n_bootstraps: Number of bootstrap iterations per combo.
        seeds: Random seeds for reproducibility.

    Returns:
        List of SensitivityResult tuples.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    reliability_coeffs = [0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
    observation_vars = [0.05, 0.10, 0.15, 0.20, 0.30]
    prior_mean, prior_var = 0.25, 0.05
    k_factor = target_samples / short_samples

    results = []
    total_combos = len(reliability_coeffs) * len(observation_vars) * len(seeds)
    combo_count = 0

    print("\n" + "=" * 70)
    print("PHASE 1: Sweep reliability_coeff × observation_var")
    print("=" * 70)
    print(f"Fixed prior: mean={prior_mean}, var={prior_var}")
    print(f"Total combinations: {total_combos}")
    print()

    for seed_val in seeds:
        np.random.seed(seed_val)

        for rel_coeff in reliability_coeffs:
            for obs_var in observation_vars:
                combo_count += 1
                print(
                    f"[{combo_count:3d}/{total_combos}] "
                    f"seed={seed_val}, rel={rel_coeff:.2f}, "
                    f"obs_var={obs_var:.2f}...",
                    end=" ",
                    flush=True,
                )

                # Generate synthetic data
                long_obs, long_signal = generate_synthetic_timeseries(
                    target_samples,
                    n_rois,
                    noise_level=noise_level,
                    ar1=ar1,
                )
                long_obs = long_obs.T  # (target_samples, n_rois)
                long_signal = long_signal.T

                # Ground truth FC from noise-free signal
                fc_true_T = get_fc_matrix(long_signal, vectorized=True)

                # Short observation
                short_obs = long_obs[:short_samples, :]

                # Run simplified bootstrap prediction
                predicted_rho = run_simplified_bootstrap_prediction(
                    short_obs,
                    fc_true_T,
                    reliability_coeff=rel_coeff,
                    empirical_prior=(prior_mean, prior_var),
                    observation_var=obs_var,
                    k_factor=k_factor,
                    n_bootstraps=n_bootstraps,
                )

                # Compute metrics
                true_rho = np.corrcoef(fc_true_T, fc_true_T)[0, 1]  # Should be 1.0
                mae = abs(predicted_rho - true_rho)
                pass_result = predicted_rho >= 0.80

                result = SensitivityResult(
                    seed=seed_val,
                    param1_name="reliability_coeff",
                    param1_val=rel_coeff,
                    param2_name="observation_var",
                    param2_val=obs_var,
                    rho_hat_T=predicted_rho,
                    mae=mae,
                    pass_=pass_result,
                )
                results.append(result)

                status = "✓ PASS" if pass_result else "✗ FAIL"
                print(f"ρ̂T={predicted_rho:.4f}, mae={mae:.4f} {status}")

    return results


def run_phase2(
    output_path: Path,
    n_rois: int = 50,
    short_samples: int = 120,
    target_samples: int = 900,
    noise_level: float = 0.25,
    ar1: float = 0.6,
    n_bootstraps: int = 30,
    seeds: list[int] | None = None,
) -> list[SensitivityResult]:
    """
    Phase 2: Sweep prior_mean × prior_var.

    Keeps reliability_coeff fixed at 0.98 and observation_var fixed at 0.15.

    Args:
        output_path: Path to save CSV results.
        n_rois: Number of regions of interest.
        short_samples: Number of samples in short observation.
        target_samples: Number of samples in target observation.
        noise_level: Noise level for synthetic data.
        ar1: AR(1) coefficient for synthetic data.
        n_bootstraps: Number of bootstrap iterations per combo.
        seeds: Random seeds for reproducibility.

    Returns:
        List of SensitivityResult tuples.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    prior_means = [0.15, 0.20, 0.25, 0.30, 0.40]
    prior_vars = [0.01, 0.03, 0.05, 0.10, 0.15]
    reliability_coeff = 0.98
    observation_var = 0.15
    k_factor = target_samples / short_samples

    results = []
    total_combos = len(prior_means) * len(prior_vars) * len(seeds)
    combo_count = 0

    print("\n" + "=" * 70)
    print("PHASE 2: Sweep prior_mean × prior_var")
    print("=" * 70)
    print(f"Fixed reliability_coeff={reliability_coeff}, observation_var={observation_var}")
    print(f"Total combinations: {total_combos}")
    print()

    for seed_val in seeds:
        np.random.seed(seed_val)

        for prior_mean in prior_means:
            for prior_var in prior_vars:
                combo_count += 1
                print(
                    f"[{combo_count:3d}/{total_combos}] "
                    f"seed={seed_val}, prior_mean={prior_mean:.2f}, "
                    f"prior_var={prior_var:.2f}...",
                    end=" ",
                    flush=True,
                )

                # Generate synthetic data
                long_obs, long_signal = generate_synthetic_timeseries(
                    target_samples,
                    n_rois,
                    noise_level=noise_level,
                    ar1=ar1,
                )
                long_obs = long_obs.T
                long_signal = long_signal.T

                # Ground truth FC from noise-free signal
                fc_true_T = get_fc_matrix(long_signal, vectorized=True)

                # Short observation
                short_obs = long_obs[:short_samples, :]

                # Run simplified bootstrap prediction
                predicted_rho = run_simplified_bootstrap_prediction(
                    short_obs,
                    fc_true_T,
                    reliability_coeff=reliability_coeff,
                    empirical_prior=(prior_mean, prior_var),
                    observation_var=observation_var,
                    k_factor=k_factor,
                    n_bootstraps=n_bootstraps,
                )

                # Compute metrics
                true_rho = np.corrcoef(fc_true_T, fc_true_T)[0, 1]
                mae = abs(predicted_rho - true_rho)
                pass_result = predicted_rho >= 0.80

                result = SensitivityResult(
                    seed=seed_val,
                    param1_name="prior_mean",
                    param1_val=prior_mean,
                    param2_name="prior_var",
                    param2_val=prior_var,
                    rho_hat_T=predicted_rho,
                    mae=mae,
                    pass_=pass_result,
                )
                results.append(result)

                status = "✓ PASS" if pass_result else "✗ FAIL"
                print(f"ρ̂T={predicted_rho:.4f}, mae={mae:.4f} {status}")

    return results


def save_results_to_csv(
    results: list[SensitivityResult], output_path: Path
) -> None:
    """
    Save sensitivity analysis results to CSV file.

    Args:
        results: List of SensitivityResult tuples.
        output_path: Path to write CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "param1_name",
                "param1_val",
                "param2_name",
                "param2_val",
                "rho_hat_T",
                "mae",
                "pass",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.seed,
                    result.param1_name,
                    result.param1_val,
                    result.param2_name,
                    result.param2_val,
                    result.rho_hat_T,
                    result.mae,
                    result.pass_,
                ]
            )

    print(f"\nResults saved to: {output_path}")


def print_summary_statistics(
    results: list[SensitivityResult], phase_name: str
) -> None:
    """
    Print summary statistics for sensitivity analysis results.

    Args:
        results: List of SensitivityResult tuples.
        phase_name: Name of the phase (for output).
    """
    if not results:
        print(f"\nNo results for {phase_name}")
        return

    rho_hat_Ts = [r.rho_hat_T for r in results]
    maes = [r.mae for r in results]
    pass_count = sum(1 for r in results if r.pass_)
    total_count = len(results)

    print(f"\n{phase_name} Summary Statistics:")
    print(f"  Total combinations: {total_count}")
    print(f"  Pass rate (ρ̂T >= 0.80): {pass_count}/{total_count} ({100*pass_count/total_count:.1f}%)")
    print(f"  Predicted ρ̂T - Mean: {np.mean(rho_hat_Ts):.4f}, "
          f"Std: {np.std(rho_hat_Ts):.4f}")
    print(f"  MAE - Mean: {np.mean(maes):.4f}, Std: {np.std(maes):.4f}")
    print(f"  MAE - Min: {np.min(maes):.4f}, Max: {np.max(maes):.4f}")


def main() -> None:
    """Run the full two-phase sensitivity analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    artifacts_dir = Path("artifacts/reports")
    output_path_phase1 = artifacts_dir / "sensitivity_phase1.csv"
    output_path_phase2 = artifacts_dir / "sensitivity_phase2.csv"

    seeds = [42, 123, 456]

    # Phase 1: reliability_coeff × observation_var
    results_phase1 = run_phase1(output_path_phase1, seeds=seeds)
    save_results_to_csv(results_phase1, output_path_phase1)
    print_summary_statistics(results_phase1, "PHASE 1 (reliability_coeff × observation_var)")

    # Phase 2: prior_mean × prior_var
    results_phase2 = run_phase2(output_path_phase2, seeds=seeds)
    save_results_to_csv(results_phase2, output_path_phase2)
    print_summary_statistics(results_phase2, "PHASE 2 (prior_mean × prior_var)")

    print("\n" + "=" * 70)
    print("Sensitivity analysis complete!")
    print(f"Phase 1 results: {output_path_phase1}")
    print(f"Phase 2 results: {output_path_phase2}")
    print("=" * 70)


if __name__ == "__main__":
    main()
