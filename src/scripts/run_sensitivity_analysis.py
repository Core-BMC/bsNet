"""
Sensitivity analysis script for BS-NET hardcoded parameters (Redesigned).

Tests hyperparameter robustness by sweeping key parameters through the
**actual bootstrap pipeline path** (not an oracle shortcut).

Design:
  1. Generate synthetic data (900 samples noisy + 120 samples short)
  2. Reference FC = Pearson FC from full 900-sample noisy observation
  3. For each parameter combo, run a custom bootstrap prediction loop
  4. Evaluate: ρ̂T stability across parameter ranges → low CV = robust

Phase 1: Sweep reliability_coeff × observation_var (prior fixed)
Phase 2: Sweep prior_mean × prior_var (reliability and obs_var fixed)

Previous version flaw: reference FC was noise-free signal FC, and
metric was compared against self-correlation (trivially 1.0).
"""

from __future__ import annotations

import argparse
import csv
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
)
from src.data.data_loader import get_fc_matrix, load_timeseries_data

logger = logging.getLogger(__name__)


class SensitivityResult(NamedTuple):
    """Result container for a single sensitivity analysis iteration."""

    seed: int
    param1_name: str
    param1_val: float
    param2_name: str
    param2_val: float
    rho_hat_T: float
    r_fc: float
    pass_: bool


def compute_split_half_reliability(
    time_series: np.ndarray, use_shrinkage: bool = True
) -> float:
    """Compute split-half reliability from time series data.

    Args:
        time_series: Time series data with shape (n_samples, n_rois).
        use_shrinkage: Whether to use Ledoit-Wolf shrinkage.

    Returns:
        float: Correlation between split halves, clipped to [0.001, 0.999].
    """
    n_split = time_series.shape[0] // 2
    fc_split1 = get_fc_matrix(
        time_series[:n_split, :], vectorized=True, use_shrinkage=use_shrinkage
    )
    fc_split2 = get_fc_matrix(
        time_series[n_split:, :], vectorized=True, use_shrinkage=use_shrinkage
    )
    r_split = np.corrcoef(fc_split1, fc_split2)[0, 1]
    return float(np.clip(r_split, 0.001, 0.999))


def run_pipeline_with_params(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    reliability_coeff: float,
    empirical_prior: tuple[float, float],
    observation_var: float,
    k_factor: float,
    n_bootstraps: int = 30,
    correction_method: str = "fisher_z",
) -> tuple[float, np.ndarray]:
    """Run bootstrap prediction with custom parameters through actual pipeline path.

    This mirrors the real pipeline logic (bootstrap → split-half → SB →
    Bayesian prior → attenuation correction) but with swappable parameters.

    Args:
        short_obs: Short observation time series (n_samples, n_rois).
        fc_reference: Reference FC vector from full-length noisy observation.
        reliability_coeff: Within-session scanner reliability to test.
        empirical_prior: (mean, var) tuple for Bayesian prior.
        observation_var: Observation variance for Bayesian weighting.
        k_factor: Spearman-Brown scaling factor (target/short).
        n_bootstraps: Number of bootstrap iterations.
        correction_method: Attenuation correction method. One of: "original",
            "fisher_z", "partial", "soft_clamp". Defaults to "fisher_z".

    Returns:
        Tuple of (rho_hat_T, fc_predicted_vector) where fc_predicted_vector
        is the median FC from bootstrap samples.
    """
    t_samples = short_obs.shape[0]
    block_size = estimate_optimal_block_length(short_obs)

    rho_hat_b = []
    fc_bootstrap_samples = []

    for _b in range(n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]

        # FC from bootstrap sample (with shrinkage, matching real pipeline)
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        fc_bootstrap_samples.append(fc_obs_t)

        # Observed correlation between reference and bootstrap FC
        r_obs_t = np.corrcoef(fc_reference, fc_obs_t)[0, 1]

        # Split-half reliability (with shrinkage, matching real pipeline)
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=True)

        # Bayesian prior stabilization of split-half estimate
        prior_mean, prior_var = empirical_prior
        weight = prior_var / (prior_var + observation_var)
        r_real_t = weight * r_split_t + (1 - weight) * prior_mean

        # Attenuation correction via correct_attenuation()
        # Note: Prior is already applied above, so pass empirical_prior=None
        min_rel = 0.05
        r_hat_t = max(reliability_coeff, min_rel)
        r_real_t = max(r_real_t, min_rel)

        rho_est_T = correct_attenuation(
            r_obs_t, r_hat_t, r_real_t,
            k=k_factor,
            empirical_prior=None,
            method=correction_method,
        )

        rho_hat_b.append(fisher_z(rho_est_T))

    rho_hat_b = np.array(rho_hat_b)
    rho_hat_T = float(fisher_z_inv(np.nanmedian(rho_hat_b)))

    # Median FC vector from bootstrap samples → for rFC computation
    fc_pred = np.median(np.array(fc_bootstrap_samples), axis=0)

    return rho_hat_T, fc_pred


def run_phase1(
    output_path: Path,
    n_rois: int = 50,
    short_samples: int = 120,
    target_samples: int = 900,
    noise_level: float = 0.25,
    ar1: float = 0.6,
    n_bootstraps: int = 30,
    seeds: list[int] | None = None,
    input_npy: str | None = None,
    correction_method: str = "fisher_z",
) -> list[SensitivityResult]:
    """Phase 1: Sweep reliability_coeff x observation_var.

    Reference FC is computed from full observation (real or synthetic).
    For real data: uses full-length time series with use_shrinkage=True.
    For synthetic: uses full-length synthetic data with use_shrinkage=False.

    Args:
        output_path: Path to save CSV results.
        n_rois: Number of regions of interest.
        short_samples: Number of samples in short observation.
        target_samples: Number of samples in target observation.
        noise_level: Noise level for synthetic data.
        ar1: AR(1) coefficient for synthetic data.
        n_bootstraps: Number of bootstrap iterations per combo.
        seeds: Random seeds for reproducibility.
        input_npy: Path to .npy file with real time series. If provided, uses real data.
        correction_method: Attenuation correction method. Defaults to "fisher_z".

    Returns:
        List of SensitivityResult tuples.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    reliability_coeffs = [0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
    observation_vars = [0.05, 0.10, 0.15, 0.20, 0.30]
    prior_mean, prior_var = 0.25, 0.05
    k_factor = target_samples / short_samples

    results: list[SensitivityResult] = []
    total_combos = len(reliability_coeffs) * len(observation_vars) * len(seeds)
    combo_count = 0

    print("\n" + "=" * 70)
    print("PHASE 1: Sweep reliability_coeff x observation_var")
    print("=" * 70)
    print(f"Fixed prior: mean={prior_mean}, var={prior_var}")
    print("Reference FC: Pearson from 900-sample noisy observation")
    print(f"Total combinations: {total_combos}")
    print()

    for seed_val in seeds:
        np.random.seed(seed_val)

        # Load or generate time series data
        if input_npy is not None:
            # Real data: load from .npy file
            ts_full, ts_short, _ = load_timeseries_data(
                input_npy=input_npy, short_samples=short_samples
            )
            long_obs = ts_full
            short_obs = ts_short
            # Real data uses shrinkage
            fc_reference = get_fc_matrix(long_obs, vectorized=True, use_shrinkage=True)
        else:
            # Synthetic data: generate
            ts_full, ts_short, _ = load_timeseries_data(
                n_samples=target_samples,
                n_rois=n_rois,
                noise_level=noise_level,
                ar1=ar1,
                short_samples=short_samples,
                seed=seed_val,
            )
            long_obs = ts_full
            short_obs = ts_short
            # Synthetic data does not use shrinkage for reference
            fc_reference = get_fc_matrix(long_obs, vectorized=True, use_shrinkage=False)

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

                rho_hat_T, fc_pred = run_pipeline_with_params(
                    short_obs,
                    fc_reference,
                    reliability_coeff=rel_coeff,
                    empirical_prior=(prior_mean, prior_var),
                    observation_var=obs_var,
                    k_factor=k_factor,
                    n_bootstraps=n_bootstraps,
                    correction_method=correction_method,
                )

                # rFC: correlation between predicted FC and reference FC
                r_fc = float(np.corrcoef(fc_reference, fc_pred)[0, 1])
                pass_result = rho_hat_T >= 0.80

                result = SensitivityResult(
                    seed=seed_val,
                    param1_name="reliability_coeff",
                    param1_val=rel_coeff,
                    param2_name="observation_var",
                    param2_val=obs_var,
                    rho_hat_T=rho_hat_T,
                    r_fc=r_fc,
                    pass_=pass_result,
                )
                results.append(result)

                status = "PASS" if pass_result else "FAIL"
                print(
                    f"rho_hat_T={rho_hat_T:.4f}, "
                    f"r_fc={r_fc:.4f} {status}"
                )

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
    input_npy: str | None = None,
    correction_method: str = "fisher_z",
) -> list[SensitivityResult]:
    """Phase 2: Sweep prior_mean x prior_var.

    Keeps reliability_coeff=0.98 and observation_var=0.15 fixed.
    Reference FC computed from full observation (real or synthetic).

    Args:
        output_path: Path to save CSV results.
        n_rois: Number of regions of interest.
        short_samples: Number of samples in short observation.
        target_samples: Number of samples in target observation.
        noise_level: Noise level for synthetic data.
        ar1: AR(1) coefficient for synthetic data.
        n_bootstraps: Number of bootstrap iterations per combo.
        seeds: Random seeds for reproducibility.
        input_npy: Path to .npy file with real time series. If provided, uses real data.
        correction_method: Attenuation correction method. Defaults to "fisher_z".

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

    results: list[SensitivityResult] = []
    total_combos = len(prior_means) * len(prior_vars) * len(seeds)
    combo_count = 0

    print("\n" + "=" * 70)
    print("PHASE 2: Sweep prior_mean x prior_var")
    print("=" * 70)
    print(f"Fixed reliability_coeff={reliability_coeff}, observation_var={observation_var}")
    print("Reference FC: Pearson from 900-sample noisy observation")
    print(f"Total combinations: {total_combos}")
    print()

    for seed_val in seeds:
        np.random.seed(seed_val)

        # Load or generate time series data
        if input_npy is not None:
            # Real data: load from .npy file
            ts_full, ts_short, _ = load_timeseries_data(
                input_npy=input_npy, short_samples=short_samples
            )
            long_obs = ts_full
            short_obs = ts_short
            # Real data uses shrinkage
            fc_reference = get_fc_matrix(long_obs, vectorized=True, use_shrinkage=True)
        else:
            # Synthetic data: generate
            ts_full, ts_short, _ = load_timeseries_data(
                n_samples=target_samples,
                n_rois=n_rois,
                noise_level=noise_level,
                ar1=ar1,
                short_samples=short_samples,
                seed=seed_val,
            )
            long_obs = ts_full
            short_obs = ts_short
            # Synthetic data does not use shrinkage for reference
            fc_reference = get_fc_matrix(long_obs, vectorized=True, use_shrinkage=False)

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

                rho_hat_T, fc_pred = run_pipeline_with_params(
                    short_obs,
                    fc_reference,
                    reliability_coeff=reliability_coeff,
                    empirical_prior=(prior_mean, prior_var),
                    observation_var=observation_var,
                    k_factor=k_factor,
                    n_bootstraps=n_bootstraps,
                    correction_method=correction_method,
                )

                r_fc = float(np.corrcoef(fc_reference, fc_pred)[0, 1])
                pass_result = rho_hat_T >= 0.80

                result = SensitivityResult(
                    seed=seed_val,
                    param1_name="prior_mean",
                    param1_val=prior_mean,
                    param2_name="prior_var",
                    param2_val=prior_var,
                    rho_hat_T=rho_hat_T,
                    r_fc=r_fc,
                    pass_=pass_result,
                )
                results.append(result)

                status = "PASS" if pass_result else "FAIL"
                print(
                    f"rho_hat_T={rho_hat_T:.4f}, "
                    f"r_fc={r_fc:.4f} {status}"
                )

    return results


def save_results_to_csv(
    results: list[SensitivityResult], output_path: Path
) -> None:
    """Save sensitivity analysis results to CSV file.

    Args:
        results: List of SensitivityResult tuples.
        output_path: Path to write CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["seed", "param1_name", "param1_val", "param2_name",
             "param2_val", "rho_hat_T", "r_fc", "pass"]
        )
        for r in results:
            writer.writerow(
                [r.seed, r.param1_name, r.param1_val, r.param2_name,
                 r.param2_val, r.rho_hat_T, r.r_fc, r.pass_]
            )

    print(f"\nResults saved to: {output_path}")


def print_summary_statistics(
    results: list[SensitivityResult], phase_name: str
) -> None:
    """Print summary statistics for sensitivity analysis results.

    Args:
        results: List of SensitivityResult tuples.
        phase_name: Name of the phase (for output).
    """
    if not results:
        print(f"\nNo results for {phase_name}")
        return

    rho_vals = [r.rho_hat_T for r in results]
    r_fc_vals = [r.r_fc for r in results]
    pass_count = sum(1 for r in results if r.pass_)
    total_count = len(results)
    cv = np.std(rho_vals) / np.mean(rho_vals) if np.mean(rho_vals) > 0 else float("inf")

    print(f"\n{'=' * 70}")
    print(f"{phase_name} Summary Statistics:")
    print(f"{'=' * 70}")
    print(f"  Total combinations: {total_count}")
    print(f"  Pass rate (rho_hat_T >= 0.80): {pass_count}/{total_count} "
          f"({100 * pass_count / total_count:.1f}%)")
    print(f"  rho_hat_T — Mean: {np.mean(rho_vals):.4f}, "
          f"SD: {np.std(rho_vals):.4f}, CV: {cv:.4f}")
    print(f"  rho_hat_T — Range: [{np.min(rho_vals):.4f}, {np.max(rho_vals):.4f}]")
    print(f"  r_fc      — Mean: {np.mean(r_fc_vals):.4f}, "
          f"SD: {np.std(r_fc_vals):.4f}")
    print(f"{'=' * 70}")


def main() -> None:
    """Run the full two-phase sensitivity analysis."""
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for BS-NET parameters"
    )
    parser.add_argument(
        "--input-npy",
        type=str,
        default=None,
        help="Path to .npy file with real time series data (shape: n_samples, n_rois). "
        "If provided, runs on real data; otherwise uses synthetic data.",
    )
    parser.add_argument(
        "--correction-method",
        type=str,
        default="fisher_z",
        choices=["original", "fisher_z", "partial", "soft_clamp"],
        help="Attenuation correction method. Default: fisher_z.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    artifacts_dir = Path("artifacts/reports")
    output_path_phase1 = artifacts_dir / "sensitivity_phase1.csv"
    output_path_phase2 = artifacts_dir / "sensitivity_phase2.csv"

    seeds = [42, 123, 456]

    # Phase 1: reliability_coeff x observation_var
    results_phase1 = run_phase1(
        output_path_phase1,
        seeds=seeds,
        input_npy=args.input_npy,
        correction_method=args.correction_method,
    )
    save_results_to_csv(results_phase1, output_path_phase1)
    print_summary_statistics(
        results_phase1, "PHASE 1 (reliability_coeff x observation_var)"
    )

    # Phase 2: prior_mean x prior_var
    results_phase2 = run_phase2(
        output_path_phase2,
        seeds=seeds,
        input_npy=args.input_npy,
        correction_method=args.correction_method,
    )
    save_results_to_csv(results_phase2, output_path_phase2)
    print_summary_statistics(results_phase2, "PHASE 2 (prior_mean x prior_var)")

    print("\nSensitivity analysis complete!")
    print(f"Phase 1 results: {output_path_phase1}")
    print(f"Phase 2 results: {output_path_phase2}")


if __name__ == "__main__":
    main()
