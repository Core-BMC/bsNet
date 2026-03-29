"""
Noise degradation analysis script for BS-NET robustness testing.

Tests BS-NET's robustness under increasingly harsh conditions across three
dimensions: noise level, ROI count, and short scan duration. Identifies the
degradation boundary where prediction performance decays.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.data.data_loader import get_fc_matrix, load_timeseries_data

logger = logging.getLogger(__name__)


@dataclass
class NoiseCondition:
    """
    Container for a single test condition.

    Attributes:
        dimension: Name of the swept dimension (noise, n_rois, or short_duration).
        parameter_name: Human-readable name of the parameter.
        parameter_value: Value of the swept parameter.
        seed: Random seed for reproducibility.
        rho_hat_T: Predicted correlation from bootstrap.
        ci_lower: Lower bound of 95% CI.
        ci_upper: Upper bound of 95% CI.
    """

    dimension: str
    parameter_name: str
    parameter_value: float
    seed: int
    rho_hat_T: float
    ci_lower: float
    ci_upper: float

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.ci_upper - self.ci_lower

    @property
    def pass_flag(self) -> int:
        """
        Heuristic pass/fail: 1 if rho > 0.5 and CI width < 0.5, else 0.

        This reflects a minimal degradation boundary.
        """
        return 1 if (self.rho_hat_T > 0.5 and self.ci_width < 0.5) else 0


def run_single_condition(
    noise_level: float,
    n_rois: int,
    short_duration: int,
    seed: int,
    input_npy: str | None = None,
    n_bootstraps: int = 30,
    correction_method: str = "fisher_z",
) -> tuple[float, float, float]:
    """
    Run a single test condition and return predicted_rho, ci_lower, ci_upper.

    Args:
        noise_level: Noise standard deviation.
        n_rois: Number of regions of interest.
        short_duration: Short observation duration in samples.
        seed: Random seed.
        input_npy: Path to real data .npy file. If provided, uses real data mode.
        n_bootstraps: Number of bootstrap resamples.
        correction_method: Attenuation correction method.

    Returns:
        Tuple of (predicted_rho, ci_lower, ci_upper).
    """
    np.random.seed(seed)

    # Full target length (900 samples = 15 min at 1 TR/s)
    full_length = 900

    # Load or generate data
    if input_npy is not None:
        # Real data mode
        ts_full, ts_short, ts_signal = load_timeseries_data(
            input_npy=input_npy,
            short_samples=short_duration,
        )
        # For real data, use ts_full as signal
        signal_full_t = ts_full  # (n_samples, n_rois)
        use_shrinkage_ref = True
    else:
        # Synthetic data mode
        ts_full, ts_short, ts_signal = load_timeseries_data(
            input_npy=None,
            n_samples=full_length,
            n_rois=n_rois,
            noise_level=noise_level,
            ar1=0.6,
            short_samples=short_duration,
            seed=seed,
        )
        signal_full_t = ts_signal  # (n_samples, n_rois)
        use_shrinkage_ref = False

    # Compute reference FC from noise-free full signal, vectorized
    fc_reference = get_fc_matrix(
        signal_full_t, vectorized=True, use_shrinkage=use_shrinkage_ref
    )

    # Extract short observation
    observed_short = ts_short  # (short_duration, n_rois)

    # Create config with this condition's parameters
    config = BSNetConfig(
        n_rois=observed_short.shape[1],
        noise_level=noise_level,
        short_duration_sec=short_duration,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # Run bootstrap prediction
    result = run_bootstrap_prediction(
        observed_short, fc_reference, config,
        correction_method=correction_method,
    )

    return result.predicted_rho, result.ci_lower, result.ci_upper


def sweep_noise_level(
    noise_levels: list[float],
    n_rois: int = 50,
    short_duration: int = 120,
    seeds: list[int] | None = None,
    input_npy: str | None = None,
    n_bootstraps: int = 100,
    correction_method: str = "fisher_z",
) -> list[NoiseCondition]:
    """
    Sweep across noise levels with fixed ROI count and short duration.

    Args:
        noise_levels: List of noise standard deviations to test.
        n_rois: Fixed number of ROIs.
        short_duration: Fixed short duration in samples.
        seeds: List of random seeds.
        input_npy: Path to real data .npy file (optional).
        n_bootstraps: Number of bootstrap resamples.
        correction_method: Attenuation correction method.

    Returns:
        List of NoiseCondition results.
    """
    if seeds is None:
        seeds = [42, 123, 777]

    results = []
    for noise in noise_levels:
        for seed in seeds:
            logger.info(f"Running noise={noise}, n_rois={n_rois}, short={short_duration}, seed={seed}")
            try:
                rho, ci_lower, ci_upper = run_single_condition(
                    noise_level=noise,
                    n_rois=n_rois,
                    short_duration=short_duration,
                    seed=seed,
                    input_npy=input_npy,
                    n_bootstraps=n_bootstraps,
                    correction_method=correction_method,
                )
                condition = NoiseCondition(
                    dimension="noise_level",
                    parameter_name="noise",
                    parameter_value=noise,
                    seed=seed,
                    rho_hat_T=rho,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                )
                results.append(condition)
                logger.info(f"  Result: rho={rho:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
            except Exception as e:
                logger.error(f"  Failed: {e}")

    return results


def sweep_n_rois(
    n_rois_list: list[int],
    noise_level: float = 0.25,
    short_duration: int = 120,
    seeds: list[int] | None = None,
    input_npy: str | None = None,
    n_bootstraps: int = 100,
    correction_method: str = "fisher_z",
) -> list[NoiseCondition]:
    """
    Sweep across ROI counts with fixed noise and short duration.

    Args:
        n_rois_list: List of ROI counts to test.
        noise_level: Fixed noise level.
        short_duration: Fixed short duration in samples.
        seeds: List of random seeds.
        input_npy: Path to real data .npy file (optional).
        n_bootstraps: Number of bootstrap resamples.
        correction_method: Attenuation correction method.

    Returns:
        List of NoiseCondition results.
    """
    if seeds is None:
        seeds = [42, 123, 777]

    results = []
    for n_rois in n_rois_list:
        for seed in seeds:
            logger.info(f"Running noise={noise_level}, n_rois={n_rois}, short={short_duration}, seed={seed}")
            try:
                rho, ci_lower, ci_upper = run_single_condition(
                    noise_level=noise_level,
                    n_rois=n_rois,
                    short_duration=short_duration,
                    seed=seed,
                    input_npy=input_npy,
                    n_bootstraps=n_bootstraps,
                    correction_method=correction_method,
                )
                condition = NoiseCondition(
                    dimension="n_rois",
                    parameter_name="n_rois",
                    parameter_value=n_rois,
                    seed=seed,
                    rho_hat_T=rho,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                )
                results.append(condition)
                logger.info(f"  Result: rho={rho:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
            except Exception as e:
                logger.error(f"  Failed: {e}")

    return results


def sweep_short_duration(
    short_durations: list[int],
    n_rois: int = 50,
    noise_level: float = 0.25,
    seeds: list[int] | None = None,
    input_npy: str | None = None,
    n_bootstraps: int = 100,
    correction_method: str = "fisher_z",
) -> list[NoiseCondition]:
    """
    Sweep across short scan durations with fixed ROI count and noise.

    Args:
        short_durations: List of short durations (in samples) to test.
        n_rois: Fixed number of ROIs.
        noise_level: Fixed noise level.
        seeds: List of random seeds.
        input_npy: Path to real data .npy file (optional).
        n_bootstraps: Number of bootstrap resamples.
        correction_method: Attenuation correction method.

    Returns:
        List of NoiseCondition results.
    """
    if seeds is None:
        seeds = [42, 123, 777]

    results = []
    for short_duration in short_durations:
        for seed in seeds:
            logger.info(f"Running noise={noise_level}, n_rois={n_rois}, short={short_duration}, seed={seed}")
            try:
                rho, ci_lower, ci_upper = run_single_condition(
                    noise_level=noise_level,
                    n_rois=n_rois,
                    short_duration=short_duration,
                    seed=seed,
                    input_npy=input_npy,
                    n_bootstraps=n_bootstraps,
                    correction_method=correction_method,
                )
                condition = NoiseCondition(
                    dimension="short_duration",
                    parameter_name="short_duration",
                    parameter_value=short_duration,
                    seed=seed,
                    rho_hat_T=rho,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                )
                results.append(condition)
                logger.info(f"  Result: rho={rho:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
            except Exception as e:
                logger.error(f"  Failed: {e}")

    return results


def save_results_csv(all_results: list[NoiseCondition], output_path: Path) -> None:
    """
    Save all results to a CSV file.

    Args:
        all_results: List of NoiseCondition results.
        output_path: Path to write the CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dimension",
                "parameter_name",
                "parameter_value",
                "seed",
                "rho_hat_T",
                "ci_lower",
                "ci_upper",
                "pass",
            ]
        )
        for condition in all_results:
            writer.writerow(
                [
                    condition.dimension,
                    condition.parameter_name,
                    condition.parameter_value,
                    condition.seed,
                    f"{condition.rho_hat_T:.6f}",
                    f"{condition.ci_lower:.6f}",
                    f"{condition.ci_upper:.6f}",
                    condition.pass_flag,
                ]
            )

    logger.info(f"Results saved to {output_path}")


def print_summary_table(results: list[NoiseCondition], dimension: str) -> None:
    """
    Print a summary table for a single dimension.

    Args:
        results: List of NoiseCondition results for this dimension.
        dimension: Name of the dimension (noise_level, n_rois, or short_duration).
    """
    # Group by parameter value
    grouped = {}
    for condition in results:
        if condition.parameter_value not in grouped:
            grouped[condition.parameter_value] = []
        grouped[condition.parameter_value].append(condition)

    # Compute summary statistics
    print(f"\n{'='*80}")
    print(f"Summary: {dimension.upper()}")
    print(f"{'='*80}")
    print(f"{'Parameter':<15} {'Mean Rho':<15} {'Std Rho':<15} {'Pass Rate':<15} {'Mean CI Width':<15}")
    print(f"{'-'*80}")

    for param_value in sorted(grouped.keys()):
        conditions = grouped[param_value]
        rhos = np.array([c.rho_hat_T for c in conditions])
        passes = np.array([c.pass_flag for c in conditions])
        ci_widths = np.array([c.ci_width for c in conditions])

        mean_rho = np.mean(rhos)
        std_rho = np.std(rhos)
        pass_rate = np.mean(passes)
        mean_ci_width = np.mean(ci_widths)

        print(
            f"{param_value:<15.2f} "
            f"{mean_rho:.4f}±{std_rho:.4f}    "
            f"{pass_rate:.2%}           "
            f"{mean_ci_width:.4f}"
        )


def main() -> None:
    """
    Execute the full noise degradation analysis.

    Runs three independent sweeps (noise, n_rois, short_duration) and produces
    summary tables and a CSV output.
    """
    parser = argparse.ArgumentParser(
        description="BS-NET noise degradation analysis for robustness testing."
    )
    parser.add_argument(
        "--input-npy",
        type=str,
        default=None,
        help="Path to real data .npy file (n_samples, n_rois). If provided, uses real data mode.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=100,
        help="Number of bootstrap resamples (default: 100).",
    )
    parser.add_argument(
        "--correction-method",
        type=str,
        default="fisher_z",
        choices=["original", "fisher_z", "partial", "soft_clamp"],
        help="Attenuation correction method (default: fisher_z).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting BS-NET noise degradation analysis...")
    if args.input_npy:
        logger.info(f"Using real data from: {args.input_npy}")
    else:
        logger.info("Using synthetic data generation.")
    logger.info(f"Bootstrap resamples: {args.n_bootstraps}")
    logger.info(f"Correction method: {args.correction_method}")

    # Define sweep parameters
    noise_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    n_rois_list = [20, 50, 100, 200]
    short_durations = [30, 60, 90, 120, 180, 240]
    seeds = [42, 123, 777]

    # Run sweeps with real or synthetic data
    logger.info("Dimension 1: Noise level sweep...")
    results_noise = sweep_noise_level(
        noise_levels,
        n_rois=50,
        short_duration=120,
        seeds=seeds,
        input_npy=args.input_npy,
        n_bootstraps=args.n_bootstraps,
        correction_method=args.correction_method,
    )

    logger.info("Dimension 2: ROI count sweep...")
    results_n_rois = sweep_n_rois(
        n_rois_list,
        noise_level=0.25,
        short_duration=120,
        seeds=seeds,
        input_npy=args.input_npy,
        n_bootstraps=args.n_bootstraps,
        correction_method=args.correction_method,
    )

    logger.info("Dimension 3: Short duration sweep...")
    results_short = sweep_short_duration(
        short_durations,
        n_rois=50,
        noise_level=0.25,
        seeds=seeds,
        input_npy=args.input_npy,
        n_bootstraps=args.n_bootstraps,
        correction_method=args.correction_method,
    )

    # Combine all results
    all_results = results_noise + results_n_rois + results_short

    # Save to CSV
    output_csv = Path("artifacts/reports/noise_degradation.csv")
    save_results_csv(all_results, output_csv)

    # Print summary tables
    print_summary_table(results_noise, "noise_level")
    print_summary_table(results_n_rois, "n_rois")
    print_summary_table(results_short, "short_duration")

    logger.info("Noise degradation analysis complete.")


if __name__ == "__main__":
    main()
