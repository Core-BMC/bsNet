"""
Ablation study for BS-NET showing incremental contribution of each component.

This script systematically removes components from the full BS-NET pipeline
to quantify the contribution of each major correction step:

    - L0 (Raw): Baseline raw correlation
    - L1 (SB only): Spearman-Brown correction
    - L2 (SB + LW): Add Ledoit-Wolf shrinkage
    - L3 (SB + LW + Bootstrap): Add block bootstrap resampling
    - L4 (SB + LW + Boot + Prior): Add Bayesian empirical prior
    - L5 (BS-NET full): Full attenuation correction pipeline

Results are saved to artifacts/reports/ablation_results.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.bootstrap import (
    block_bootstrap_indices,
    correct_attenuation,
    estimate_optimal_block_length,
    spearman_brown,
)
from src.core.config import BSNetConfig
from src.core.simulate import generate_synthetic_timeseries
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)


def compute_split_half_reliability(
    time_series: np.ndarray, use_shrinkage: bool = False
) -> float:
    """
    Compute split-half reliability from time series data.

    Args:
        time_series: Time series data with shape (n_samples, n_rois).
        use_shrinkage: Whether to use Ledoit-Wolf shrinkage for covariance.

    Returns:
        float: Correlation between split halves.
    """
    n_split = time_series.shape[0] // 2

    fc_split1 = get_fc_matrix(
        time_series[:n_split, :], vectorized=True, use_shrinkage=use_shrinkage
    )
    fc_split2 = get_fc_matrix(
        time_series[n_split:, :], vectorized=True, use_shrinkage=use_shrinkage
    )

    r_split = np.corrcoef(fc_split1, fc_split2)[0, 1]
    r_split = np.clip(r_split, 0.001, 0.999)

    return r_split


def ablation_l0_raw(
    short_obs: np.ndarray,
    full_obs: np.ndarray,
    config: BSNetConfig,
) -> float:
    """
    L0 (Raw): Baseline correlation between 2-min and full FC matrices.

    Args:
        short_obs: Short observation time series (n_samples, n_rois).
        full_obs: Full duration time series (n_samples, n_rois).
        config: Configuration object.

    Returns:
        float: Predicted correlation coefficient.
    """
    fc_short = get_fc_matrix(short_obs, vectorized=True, use_shrinkage=False)
    fc_full = get_fc_matrix(full_obs, vectorized=True, use_shrinkage=False)

    rho = np.corrcoef(fc_short, fc_full)[0, 1]
    rho = np.clip(rho, -1.0, 1.0)

    return rho


def ablation_l1_sb_only(
    short_obs: np.ndarray,
    full_obs: np.ndarray,
    config: BSNetConfig,
) -> float:
    """
    L1 (SB only): Apply Spearman-Brown correction to raw correlation.

    Args:
        short_obs: Short observation time series.
        full_obs: Full duration time series.
        config: Configuration object.

    Returns:
        float: SB-corrected correlation coefficient.
    """
    fc_short = get_fc_matrix(short_obs, vectorized=True, use_shrinkage=False)
    fc_full = get_fc_matrix(full_obs, vectorized=True, use_shrinkage=False)

    rho = np.corrcoef(fc_short, fc_full)[0, 1]

    # Compute split-half reliability and apply SB
    r_split = compute_split_half_reliability(short_obs, use_shrinkage=False)
    r_sb = spearman_brown(r_split, config.k_factor)

    # Scale raw correlation by sqrt of SB-corrected reliability
    rho_scaled = rho * np.sqrt(r_sb)
    rho_scaled = np.clip(rho_scaled, -1.0, 1.0)

    return rho_scaled


def ablation_l2_sb_lw(
    short_obs: np.ndarray,
    full_obs: np.ndarray,
    config: BSNetConfig,
) -> float:
    """
    L2 (SB + LW): Apply SB correction with Ledoit-Wolf shrinkage for FC.

    Args:
        short_obs: Short observation time series.
        full_obs: Full duration time series.
        config: Configuration object.

    Returns:
        float: SB + LW corrected correlation coefficient.
    """
    fc_short = get_fc_matrix(short_obs, vectorized=True, use_shrinkage=True)
    fc_full = get_fc_matrix(full_obs, vectorized=True, use_shrinkage=True)

    rho = np.corrcoef(fc_short, fc_full)[0, 1]

    # Compute split-half reliability with shrinkage and apply SB
    r_split = compute_split_half_reliability(short_obs, use_shrinkage=True)
    r_sb = spearman_brown(r_split, config.k_factor)

    # Scale raw correlation
    rho_scaled = rho * np.sqrt(r_sb)
    rho_scaled = np.clip(rho_scaled, -1.0, 1.0)

    return rho_scaled


def ablation_l3_bootstrap(
    short_obs: np.ndarray,
    full_obs: np.ndarray,
    config: BSNetConfig,
    n_bootstrap: int = 30,
) -> float:
    """
    L3 (SB + LW + Bootstrap): Apply block bootstrap resampling.

    Args:
        short_obs: Short observation time series.
        full_obs: Full duration time series.
        config: Configuration object.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        float: Median of bootstrap-corrected correlation estimates.
    """
    fc_full = get_fc_matrix(full_obs, vectorized=True, use_shrinkage=True)
    t_samples = short_obs.shape[0]
    block_size = estimate_optimal_block_length(short_obs)

    rho_estimates = []

    for _ in range(n_bootstrap):
        # Block bootstrap resampling
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]

        fc_short = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        rho = np.corrcoef(fc_short, fc_full)[0, 1]

        # Apply SB correction
        r_split = compute_split_half_reliability(ts_b, use_shrinkage=True)
        r_sb = spearman_brown(r_split, config.k_factor)
        rho_scaled = rho * np.sqrt(r_sb)

        rho_estimates.append(np.clip(rho_scaled, -1.0, 1.0))

    return np.median(rho_estimates)


def ablation_l4_prior(
    short_obs: np.ndarray,
    full_obs: np.ndarray,
    config: BSNetConfig,
    n_bootstrap: int = 30,
) -> float:
    """
    L4 (SB + LW + Bootstrap + Prior): Add Bayesian empirical prior.

    Args:
        short_obs: Short observation time series.
        full_obs: Full duration time series.
        config: Configuration object.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        float: Median of bootstrap estimates with Bayesian prior.
    """
    fc_full = get_fc_matrix(full_obs, vectorized=True, use_shrinkage=True)
    t_samples = short_obs.shape[0]
    block_size = estimate_optimal_block_length(short_obs)
    empirical_prior = config.empirical_prior

    rho_estimates = []

    for _ in range(n_bootstrap):
        # Block bootstrap resampling
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]

        fc_short = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        rho = np.corrcoef(fc_short, fc_full)[0, 1]

        # Compute split-half reliability
        r_split = compute_split_half_reliability(ts_b, use_shrinkage=True)

        # Apply Bayesian prior shrinkage to split-half reliability
        if empirical_prior is not None:
            prior_mean, prior_var = empirical_prior
            observation_var = 0.15
            weight = prior_var / (prior_var + observation_var)
            r_split = weight * r_split + (1 - weight) * prior_mean

        r_split = np.clip(r_split, 0.001, 0.999)

        # Apply SB correction
        r_sb = spearman_brown(r_split, config.k_factor)
        rho_scaled = rho * np.sqrt(r_sb)

        rho_estimates.append(np.clip(rho_scaled, -1.0, 1.0))

    return np.median(rho_estimates)


def ablation_l5_full(
    short_obs: np.ndarray,
    full_obs: np.ndarray,
    config: BSNetConfig,
    n_bootstrap: int = 30,
) -> float:
    """
    L5 (BS-NET full): Full attenuation correction pipeline.

    Args:
        short_obs: Short observation time series.
        full_obs: Full duration time series.
        config: Configuration object.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        float: Median of fully corrected estimates.
    """
    fc_full = get_fc_matrix(full_obs, vectorized=True, use_shrinkage=True)
    t_samples = short_obs.shape[0]
    block_size = estimate_optimal_block_length(short_obs)
    empirical_prior = config.empirical_prior

    rho_estimates = []

    for _ in range(n_bootstrap):
        # Block bootstrap resampling
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]

        # Compute observed correlation
        fc_short = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs_t = np.corrcoef(fc_short, fc_full)[0, 1]

        # Compute split-half reliability
        r_split = compute_split_half_reliability(ts_b, use_shrinkage=True)

        # Full attenuation correction
        rho_est = correct_attenuation(
            r_obs_t,
            config.reliability_coeff,
            r_split,
            k=config.k_factor,
            empirical_prior=empirical_prior,
        )

        rho_estimates.append(np.clip(rho_est, -1.0, 1.0))

    return np.median(rho_estimates)


def run_single_trial(
    seed: int,
    n_rois: int,
    short_samples: int,
    full_samples: int,
    noise_level: float,
    ar1: float,
    config: BSNetConfig,
) -> dict:
    """
    Run a single ablation trial across all levels.

    Args:
        seed: Random seed for reproducibility.
        n_rois: Number of regions of interest.
        short_samples: Number of samples for short observation.
        full_samples: Number of samples for full observation.
        noise_level: Standard deviation of added Gaussian noise.
        ar1: AR(1) autocorrelation coefficient.
        config: Configuration object.

    Returns:
        dict: Results dictionary with level, method, seed, predicted_rho, mae, pass.
    """
    np.random.seed(seed)

    # Generate synthetic data - both short and full use the same random generator state
    # simulate returns (n_rois, n_samples), but get_fc_matrix expects (n_samples, n_rois)
    short_obs, short_true = generate_synthetic_timeseries(
        short_samples, n_rois, noise_level=noise_level, ar1=ar1
    )
    full_obs, full_true = generate_synthetic_timeseries(
        full_samples, n_rois, noise_level=noise_level, ar1=ar1
    )

    # Transpose to (n_samples, n_rois) for get_fc_matrix
    short_obs = short_obs.T
    short_true = short_true.T
    full_obs = full_obs.T
    full_true = full_true.T

    # Reference FC: correlation between true signal FC matrices (noiseless)
    fc_short_true = get_fc_matrix(short_true, vectorized=True, use_shrinkage=True)
    fc_full_true = get_fc_matrix(full_true, vectorized=True, use_shrinkage=True)
    fc_reference = np.corrcoef(fc_short_true, fc_full_true)[0, 1]
    fc_reference = np.clip(fc_reference, -1.0, 1.0)

    # Define ablation levels
    levels = [
        ("L0", "Raw", ablation_l0_raw),
        ("L1", "SB only", ablation_l1_sb_only),
        ("L2", "SB + LW", ablation_l2_sb_lw),
        ("L3", "SB + LW + Bootstrap", ablation_l3_bootstrap),
        ("L4", "SB + LW + Boot + Prior", ablation_l4_prior),
        ("L5", "BS-NET full", ablation_l5_full),
    ]

    results = []

    for level, method_name, func in levels:
        try:
            predicted_rho = func(short_obs, full_obs, config)
            mae = abs(predicted_rho - fc_reference)
            pass_flag = 1 if predicted_rho >= 0.80 else 0

            results.append(
                {
                    "level": level,
                    "method_name": method_name,
                    "seed": seed,
                    "rho_hat_T": predicted_rho,
                    "mae": mae,
                    "pass": pass_flag,
                    "fc_reference": fc_reference,
                }
            )
        except Exception as e:
            logger.error(
                f"Error in {level} ({method_name}) with seed {seed}: {e}",
                exc_info=True,
            )
            results.append(
                {
                    "level": level,
                    "method_name": method_name,
                    "seed": seed,
                    "rho_hat_T": np.nan,
                    "mae": np.nan,
                    "pass": 0,
                    "fc_reference": fc_reference,
                }
            )

    return results


def main() -> None:
    """
    Execute ablation study and save results to CSV.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Configuration
    config = BSNetConfig(
        n_rois=50,
        short_duration_sec=120,
        target_duration_min=15,
        noise_level=0.25,
        ar1=0.6,
    )

    config.create_output_dirs()

    seeds = [42, 123, 777, 2026, 9999]
    short_samples = config.short_samples
    full_samples = config.target_samples

    logger.info("=" * 70)
    logger.info("BS-NET ABLATION STUDY")
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info(f"  n_rois: {config.n_rois}")
    logger.info(f"  short_samples: {short_samples}")
    logger.info(f"  full_samples: {full_samples}")
    logger.info(f"  noise_level: {config.noise_level}")
    logger.info(f"  ar1: {config.ar1}")
    logger.info(f"  k_factor (scaling): {config.k_factor:.2f}")
    logger.info(f"  seeds: {seeds}")
    logger.info("=" * 70)

    all_results = []

    for seed in seeds:
        logger.info(f"\nProcessing seed {seed}...")
        trial_results = run_single_trial(
            seed=seed,
            n_rois=config.n_rois,
            short_samples=short_samples,
            full_samples=full_samples,
            noise_level=config.noise_level,
            ar1=config.ar1,
            config=config,
        )
        all_results.extend(trial_results)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save to CSV
    output_path = Path(config.artifacts_dir) / "ablation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    # Compute summary statistics per level
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 70)

    summary_stats = []

    for level in ["L0", "L1", "L2", "L3", "L4", "L5"]:
        level_data = df[df["level"] == level]
        method_name = level_data["method_name"].iloc[0]

        rho_mean = level_data["rho_hat_T"].mean()
        rho_std = level_data["rho_hat_T"].std()
        mae_mean = level_data["mae"].mean()
        mae_std = level_data["mae"].std()
        pass_rate = level_data["pass"].sum() / len(level_data)

        summary_stats.append(
            {
                "Level": level,
                "Method": method_name,
                "Rho Mean": rho_mean,
                "Rho Std": rho_std,
                "MAE Mean": mae_mean,
                "MAE Std": mae_std,
                "Pass Rate": pass_rate,
            }
        )

    summary_df = pd.DataFrame(summary_stats)

    # Print formatted table
    logger.info("\n")
    logger.info(
        f"{'Level':<10} {'Method':<25} {'Rho':<15} {'MAE':<15} {'Pass %':<10}"
    )
    logger.info("-" * 75)

    for _, row in summary_df.iterrows():
        rho_str = f"{row['Rho Mean']:.4f} ± {row['Rho Std']:.4f}"
        mae_str = f"{row['MAE Mean']:.4f} ± {row['MAE Std']:.4f}"
        pass_pct = f"{row['Pass Rate']*100:.1f}%"

        logger.info(
            f"{row['Level']:<10} {row['Method']:<25} {rho_str:<15} "
            f"{mae_str:<15} {pass_pct:<10}"
        )

    # Incremental gains
    logger.info("\n" + "=" * 70)
    logger.info("INCREMENTAL GAINS")
    logger.info("=" * 70)

    base_rho = summary_df.loc[summary_df["Level"] == "L0", "Rho Mean"].values[0]
    base_mae = summary_df.loc[summary_df["Level"] == "L0", "MAE Mean"].values[0]

    for _, row in summary_df.iterrows():
        if row["Level"] == "L0":
            logger.info(f"{row['Level']}: Baseline")
        else:
            rho_gain = row["Rho Mean"] - base_rho
            mae_reduction = base_mae - row["MAE Mean"]
            logger.info(
                f"{row['Level']}: Rho Δ = {rho_gain:+.4f}, MAE Δ = {mae_reduction:+.4f}"
            )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
