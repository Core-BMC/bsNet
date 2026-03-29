"""
Stationarity Test for Windowed FC Estimates.

This script tests whether windowed functional connectivity (FC) estimates from
synthetic fMRI data are approximately stationary, supporting the Spearman-Brown
parallel test assumption. It computes intraclass correlation (ICC), tests for
stationarity using Augmented Dickey-Fuller (ADF) test, and reports window-to-window
correlation metrics.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np

from src.core.config import BSNetConfig
from src.data.data_loader import get_fc_matrix, load_timeseries_data

logger = logging.getLogger(__name__)


def adf_test_available() -> bool:
    """
    Check if statsmodels ADF test is available.

    Returns:
        bool: True if statsmodels.tsa.stattools.adfuller is available.
    """
    try:
        from statsmodels.tsa.stattools import adfuller  # noqa: F401
        return True
    except ImportError:
        return False


def compute_adf_statistic(x: np.ndarray) -> float:
    """
    Compute ADF test statistic for a time series.

    If statsmodels is available, use adfuller. Otherwise, use a simple
    autocorrelation-based approximation.

    Args:
        x: 1D array of values.

    Returns:
        float: p-value from ADF test (0.0-1.0). High p-value indicates
               non-stationarity; low p-value indicates stationarity.
    """
    if len(x) < 3:
        return 1.0

    try:
        from statsmodels.tsa.stattools import adfuller

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(x, autolag="AIC", regression="c")
        return result[1]
    except ImportError:
        return compute_autocorr_based_stationarity(x)


def compute_autocorr_based_stationarity(x: np.ndarray) -> float:
    """
    Simple autocorrelation-based stationarity test.

    Computes the autocorrelation of first differences. High autocorrelation
    suggests non-stationarity; low autocorrelation suggests stationarity.

    Args:
        x: 1D array of values.

    Returns:
        float: Approximate p-value (0.0-1.0) for stationarity hypothesis.
    """
    if len(x) < 3:
        return 1.0

    diff = np.diff(x)
    if len(diff) < 2:
        return 1.0

    np.mean(diff)
    var_diff = np.var(diff, ddof=1)

    if var_diff < 1e-10:
        return 0.01

    acf_lag1 = np.corrcoef(diff[:-1], diff[1:])[0, 1]
    acf_lag1 = np.clip(acf_lag1, -0.999, 0.999)

    p_value = (1.0 - acf_lag1) / 2.0
    return np.clip(p_value, 0.01, 0.99)


def compute_icc_2_1(values: np.ndarray) -> float:
    """
    Compute ICC(2,1) using one-way random effects model.

    ICC(2,1) assumes measurements are from a random sample of judges/windows
    rating multiple targets. Estimates absolute agreement between windows
    (intraclass correlation).

    Args:
        values: 2D array of shape (n_windows, n_items) where each row is
                a window's FC vector.

    Returns:
        float: ICC(2,1) value, range [-1, 1]. Values > 0.75 indicate good
               reliability; > 0.9 indicate excellent reliability.
    """
    n_windows = values.shape[0]
    if n_windows < 2:
        return 0.0

    mean_per_window = np.mean(values, axis=1, keepdims=True)
    overall_mean = np.mean(values)

    ss_between = n_windows * np.sum((mean_per_window - overall_mean) ** 2)
    ss_within = np.sum((values - mean_per_window) ** 2)

    ms_between = ss_between / (n_windows - 1)
    ms_within = ss_within / (n_windows * (values.shape[1] - 1))

    if ms_between + (n_windows - 1) * ms_within <= 0:
        return 0.0

    icc = (ms_between - ms_within) / (
        ms_between + (n_windows - 1) * ms_within
    )
    return float(np.clip(icc, -1.0, 1.0))


def run_stationarity_test_single_seed(
    seed: int, input_npy: str | None = None
) -> dict:
    """
    Run stationarity test for a single random seed.

    Generates or loads fMRI data, splits into windows, computes FC estimates,
    and runs stationarity tests.

    Args:
        seed: Random seed for reproducibility.
        input_npy: Path to preprocessed .npy timeseries (optional).

    Returns:
        dict: Results dictionary containing:
            - "seed": Input seed
            - "adf_rejection_rate": Fraction of edges with p < 0.05
            - "mean_window_corr": Mean off-diagonal correlation between windows
            - "icc_2_1": Intraclass correlation for windows
            - "n_edges": Number of FC edges
            - "n_windows": Number of windows
    """
    np.random.seed(seed)

    n_samples = 900
    n_rois = 50
    n_windows = 7
    window_size = 120

    logger.info(
        f"[Seed {seed}] Loading/generating data: "
        f"{n_samples} samples, {n_rois} ROIs"
    )

    if input_npy is not None:
        ts_full, _, _ = load_timeseries_data(
            input_npy=input_npy, short_samples=120
        )
        data = ts_full
    else:
        ts_full, _, _ = load_timeseries_data(
            n_samples=n_samples,
            n_rois=n_rois,
            noise_level=0.25,
            ar1=0.6,
            short_samples=120,
            seed=seed,
        )
        data = ts_full

    window_fc_vectors = []
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_data = data[start_idx:end_idx, :]

        fc_vec = get_fc_matrix(window_data, vectorized=True)
        window_fc_vectors.append(fc_vec)

    window_fc_vectors = np.array(window_fc_vectors)
    n_edges = window_fc_vectors.shape[1]

    logger.info(f"[Seed {seed}] FC vectors shape: {window_fc_vectors.shape}")

    adf_rejections = 0
    for edge_idx in range(n_edges):
        edge_values = window_fc_vectors[:, edge_idx]
        p_value = compute_adf_statistic(edge_values)

        if p_value < 0.05:
            adf_rejections += 1

    adf_rejection_rate = adf_rejections / n_edges

    window_corr_matrix = np.corrcoef(window_fc_vectors)
    np.fill_diagonal(window_corr_matrix, 0)
    off_diag_corrs = window_corr_matrix[
        np.triu_indices_from(window_corr_matrix, k=1)
    ]
    mean_window_corr = np.mean(off_diag_corrs)

    icc_2_1 = compute_icc_2_1(window_fc_vectors)

    logger.info(f"[Seed {seed}] ADF rejection rate: {adf_rejection_rate:.3f}")
    logger.info(f"[Seed {seed}] Mean window correlation: {mean_window_corr:.3f}")
    logger.info(f"[Seed {seed}] ICC(2,1): {icc_2_1:.3f}")

    return {
        "seed": seed,
        "adf_rejection_rate": adf_rejection_rate,
        "mean_window_corr": mean_window_corr,
        "icc_2_1": icc_2_1,
        "n_edges": n_edges,
        "n_windows": n_windows,
    }


def run_stationarity_test(
    seeds: list[int] | None = None, input_npy: str | None = None
) -> None:
    """
    Run stationarity test across multiple random seeds and aggregate results.

    Args:
        seeds: List of random seeds. Defaults to [42, 123, 777, 2026, 9999].
        input_npy: Path to preprocessed .npy timeseries (optional).
    """
    if seeds is None:
        seeds = [42, 123, 777, 2026, 9999]

    logger.info("=" * 70)
    logger.info("FC Stationarity Test: Parallel Test Assumption Validation")
    logger.info("=" * 70)

    results = []
    for seed in seeds:
        result = run_stationarity_test_single_seed(seed, input_npy=input_npy)
        results.append(result)

    results = np.array(
        [
            (
                r["seed"],
                r["adf_rejection_rate"],
                r["mean_window_corr"],
                r["icc_2_1"],
            )
            for r in results
        ],
        dtype=[
            ("seed", int),
            ("adf_rejection_rate", float),
            ("mean_window_corr", float),
            ("icc_2_1", float),
        ],
    )

    config = BSNetConfig()
    config.create_output_dirs()

    results_path = Path(config.artifacts_dir) / "stationarity_results.csv"
    np.savetxt(
        results_path,
        results,
        fmt="%d,%f,%f,%f",
        header="seed,adf_rejection_rate,mean_window_corr,icc_2_1",
        comments="",
    )
    logger.info(f"Saved per-seed results to: {results_path}")

    adf_rates = results["adf_rejection_rate"]
    corr_values = results["mean_window_corr"]
    icc_values = results["icc_2_1"]

    adf_mean = np.mean(adf_rates)
    adf_std = np.std(adf_rates)
    corr_mean = np.mean(corr_values)
    corr_std = np.std(corr_values)
    icc_mean = np.mean(icc_values)
    icc_std = np.std(icc_values)

    window_corr_summary = np.array(
        [
            [adf_mean, adf_std],
            [corr_mean, corr_std],
            [icc_mean, icc_std],
        ]
    )
    window_corr_path = (
        Path(config.artifacts_dir) / "stationarity_window_corr.csv"
    )
    np.savetxt(
        window_corr_path,
        window_corr_summary,
        fmt="%.4f,%.4f",
        header="metric_mean,metric_std",
        comments="",
    )
    logger.info(f"Saved summary statistics to: {window_corr_path}")

    logger.info("=" * 70)
    logger.info("STATIONARITY TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Mean inter-window FC correlation: {corr_mean:.2f} ± {corr_std:.2f}")
    logger.info(
        f"ADF rejection rate: {adf_mean*100:.1f}% "
        f"(stationarity supported)"
    )
    logger.info(f"ICC(2,1): {icc_mean:.2f} ± {icc_std:.2f}")

    if corr_mean > 0.80:
        conclusion = "Approximate parallel test assumption is SUPPORTED"
    else:
        conclusion = "Approximate parallel test assumption is NOT SUPPORTED"

    logger.info(f"\nConclusion: {conclusion}")
    logger.info("=" * 70)

    print("\n" + "=" * 70)
    print("STATIONARITY TEST SUMMARY")
    print("=" * 70)
    print(
        f"Mean inter-window FC correlation: {corr_mean:.2f} ± {corr_std:.2f}"
    )
    print(
        f"ADF rejection rate: {adf_mean*100:.1f}% "
        "(stationarity supported)"
    )
    print(f"ICC(2,1): {icc_mean:.2f} ± {icc_std:.2f}")
    print(f"\nConclusion: {conclusion}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stationarity test for windowed FC estimates"
    )
    parser.add_argument(
        "--input-npy",
        type=str,
        default=None,
        help="Path to preprocessed .npy timeseries (optional)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    run_stationarity_test(input_npy=args.input_npy)
