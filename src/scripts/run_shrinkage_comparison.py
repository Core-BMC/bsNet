"""
Shrinkage covariance estimator comparison for FC estimation from short scans.

Compares Ledoit-Wolf vs OAS vs Pearson correlation for functional connectivity
(FC) estimation from short fMRI scans across different numbers of ROIs and
random seeds. Evaluates ground truth FC from long scans (900 samples) and
compares estimation quality using correlation, mean absolute error, Frobenius
norm, and condition number metrics.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from src.core.config import BSNetConfig
from src.core.simulate import generate_synthetic_timeseries

try:
    from sklearn.covariance import OAS, LedoitWolf

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


def ledoit_wolf_shrinkage(x: np.ndarray) -> np.ndarray:
    """
    Apply Ledoit-Wolf shrinkage to sample covariance matrix.

    Implements the Ledoit-Wolf formula for optimal shrinkage intensity when
    sklearn is unavailable.

    Args:
        x: Data matrix of shape (n_samples, n_features).

    Returns:
        Shrunk covariance matrix of shape (n_features, n_features).
    """
    n, p = x.shape
    S = np.cov(x.T)

    # Shrinkage target: mu*I where mu = trace(S)/p
    mu = np.trace(S) / p
    I_mat = np.eye(p)

    # Ledoit-Wolf optimal shrinkage intensity
    # alpha = ((1-2/p)*trace(S²) + trace(S)²) / ((n+1-2/p)*(trace(S²)-trace(S)²/p))
    S_sq = S @ S
    trace_S = np.trace(S)
    trace_S_sq = np.trace(S_sq)

    numerator = (1 - 2 / p) * trace_S_sq + trace_S**2
    denominator = (n + 1 - 2 / p) * (trace_S_sq - trace_S**2 / p)

    alpha = numerator / (denominator + 1e-10)
    alpha = np.clip(alpha, 0, 1)

    S_shrunk = (1 - alpha) * S + alpha * mu * I_mat

    return S_shrunk


def oas_shrinkage(x: np.ndarray) -> np.ndarray:
    """
    Apply Oracle Approximating Shrinkage (OAS) to sample covariance matrix.

    Implements the OAS formula for optimal shrinkage intensity when sklearn
    is unavailable.

    Args:
        x: Data matrix of shape (n_samples, n_features).

    Returns:
        Shrunk covariance matrix of shape (n_features, n_features).
    """
    n, p = x.shape
    S = np.cov(x.T)

    # Shrinkage target: mu*I where mu = trace(S)/p
    mu = np.trace(S) / p
    I_mat = np.eye(p)

    # OAS optimal shrinkage intensity
    S_sq = S @ S
    trace_S = np.trace(S)
    trace_S_sq = np.trace(S_sq)

    numerator = (1 - 2 / p) * trace_S_sq + trace_S**2
    denominator = (n + 1 - 2 / p) * (trace_S_sq - trace_S**2 / p)

    alpha_oas = numerator / (denominator + 1e-10)
    alpha_oas = np.clip(alpha_oas, 0, 1)

    S_shrunk = (1 - alpha_oas) * S + alpha_oas * mu * I_mat

    return S_shrunk


def cov_to_corr(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix.

    Args:
        cov_matrix: Covariance matrix of shape (p, p).

    Returns:
        Correlation matrix of shape (p, p).
    """
    d = np.sqrt(np.diag(cov_matrix))
    corr = cov_matrix / np.outer(d, d)
    return corr


def compute_fc_metrics(
    fc_true: np.ndarray, fc_estimated: np.ndarray
) -> dict[str, float]:
    """
    Compute metrics comparing estimated FC to ground truth.

    Args:
        fc_true: Ground truth FC matrix (p, p).
        fc_estimated: Estimated FC matrix (p, p).

    Returns:
        Dictionary with keys: rho, mae, frobenius, condition_number.
    """
    # Extract upper triangles (vectorize)
    triu_idx = np.triu_indices(fc_true.shape[0], k=1)
    fc_true_vec = fc_true[triu_idx]
    fc_est_vec = fc_estimated[triu_idx]

    # Pearson correlation between vectorized upper triangles
    rho = np.corrcoef(fc_true_vec, fc_est_vec)[0, 1]

    # Mean absolute error
    mae = np.mean(np.abs(fc_true_vec - fc_est_vec))

    # Frobenius norm
    frobenius = np.linalg.norm(fc_true - fc_estimated, "fro")

    # Condition number (for estimated matrix)
    cond_num = np.linalg.cond(fc_estimated)

    return {
        "rho": float(rho),
        "mae": float(mae),
        "frobenius": float(frobenius),
        "condition_number": float(cond_num),
    }


def estimate_fc_pearson(x: np.ndarray) -> np.ndarray:
    """
    Estimate FC using Pearson correlation.

    Args:
        x: Data matrix of shape (n_samples, n_rois).

    Returns:
        Correlation matrix of shape (n_rois, n_rois).
    """
    return np.corrcoef(x.T)


def estimate_fc_ledoit_wolf(x: np.ndarray) -> np.ndarray:
    """
    Estimate FC using Ledoit-Wolf shrinkage covariance.

    Args:
        x: Data matrix of shape (n_samples, n_rois).

    Returns:
        Correlation matrix derived from shrunk covariance.
    """
    if HAS_SKLEARN:
        lw = LedoitWolf()
        cov_shrunk = lw.fit(x).covariance_
    else:
        cov_shrunk = ledoit_wolf_shrinkage(x)

    return cov_to_corr(cov_shrunk)


def estimate_fc_oas(x: np.ndarray) -> np.ndarray:
    """
    Estimate FC using Oracle Approximating Shrinkage (OAS).

    Args:
        x: Data matrix of shape (n_samples, n_rois).

    Returns:
        Correlation matrix derived from OAS-shrunk covariance.
    """
    if HAS_SKLEARN:
        oas = OAS()
        cov_shrunk = oas.fit(x).covariance_
    else:
        cov_shrunk = oas_shrinkage(x)

    return cov_to_corr(cov_shrunk)


def run_shrinkage_comparison() -> None:
    """
    Run comprehensive shrinkage method comparison across ROI counts and seeds.

    For each (n_rois, seed) pair:
      1. Generate synthetic data (900 samples for ground truth, 120 for short)
      2. Compute ground truth FC from 900 samples using Pearson
      3. Compute short-scan FC using 3 methods: Pearson, Ledoit-Wolf, OAS
      4. Compare each method's FC to ground truth using rho, MAE, Frobenius,
         and condition number
      5. Save results to artifacts/reports/shrinkage_comparison.csv
      6. Print formatted comparison table grouped by n_rois
    """
    logger.info("Starting shrinkage comparison analysis")

    n_rois_list = [20, 50, 100, 200]
    seeds = [42, 123, 777]

    # Ground truth uses 900 samples (900/120 = 7.5x longer than short scan)
    n_samples_long = 900
    n_samples_short = 120

    config = BSNetConfig()
    config.create_output_dirs()

    results = []

    for n_rois in n_rois_list:
        logger.info(f"Processing n_rois={n_rois}")
        print(f"\n{'='*70}")
        print(f"n_rois = {n_rois}")
        print(f"{'='*70}")
        print(
            f"{'Method':<15} {'rho':>10} {'MAE':>10} {'Frobenius':>12}"
            f" {'Cond#':>12}"
        )
        print("-" * 70)

        for seed in seeds:
            np.random.seed(seed)

            # Generate long data (ground truth)
            long_data, _ = generate_synthetic_timeseries(
                n_samples_long, n_rois, noise_level=0.25, ar1=0.6
            )
            long_data = long_data.T  # Shape: (n_samples, n_rois)

            # Compute ground truth FC from long data using Pearson
            fc_true = estimate_fc_pearson(long_data)

            # Generate short data
            short_data, _ = generate_synthetic_timeseries(
                n_samples_short, n_rois, noise_level=0.25, ar1=0.6
            )
            short_data = short_data.T  # Shape: (n_samples, n_rois)

            # Estimate FC using three methods
            fc_pearson = estimate_fc_pearson(short_data)
            fc_lw = estimate_fc_ledoit_wolf(short_data)
            fc_oas = estimate_fc_oas(short_data)

            # Compute metrics for each method
            metrics_pearson = compute_fc_metrics(fc_true, fc_pearson)
            metrics_lw = compute_fc_metrics(fc_true, fc_lw)
            metrics_oas = compute_fc_metrics(fc_true, fc_oas)

            # Store results
            for method, metrics in [
                ("Pearson", metrics_pearson),
                ("Ledoit-Wolf", metrics_lw),
                ("OAS", metrics_oas),
            ]:
                results.append(
                    {
                        "n_rois": n_rois,
                        "seed": seed,
                        "method": method,
                        "rho": metrics["rho"],
                        "mae": metrics["mae"],
                        "frobenius": metrics["frobenius"],
                        "condition_number": metrics["condition_number"],
                    }
                )

            # Print metrics for this seed
            print(f"Seed {seed}:")
            for method, metrics in [
                ("Pearson", metrics_pearson),
                ("Ledoit-Wolf", metrics_lw),
                ("OAS", metrics_oas),
            ]:
                print(
                    f"  {method:<12} {metrics['rho']:>10.4f} "
                    f"{metrics['mae']:>10.4f} {metrics['frobenius']:>12.4f} "
                    f"{metrics['condition_number']:>12.2e}"
                )

    # Save results to CSV
    output_path = Path(config.artifacts_dir) / "shrinkage_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n_rois", "seed", "method", "rho", "mae", "frobenius",
                        "condition_number"],
        )
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to {output_path}")
    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_shrinkage_comparison()
