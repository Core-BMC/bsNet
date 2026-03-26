"""
Bootstrap prediction pipeline module for BS-NET.

Extracts and unifies the duplicated bootstrap prediction loop that appears
across multiple analysis scripts into a single, reusable function.
"""

import logging
from typing import NamedTuple

import numpy as np

from src.core.bootstrap import (
    block_bootstrap_indices,
    correct_attenuation,
    estimate_optimal_block_length,
    fisher_z,
    fisher_z_inv,
)
from src.core.config import BSNetConfig
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)


class BootstrapResult(NamedTuple):
    """
    Result container for bootstrap prediction analysis.

    Attributes:
        predicted_rho: Point estimate of correlation (median of bootstrap distribution).
        ci_lower: Lower bound of 95% confidence interval.
        ci_upper: Upper bound of 95% confidence interval.
        z_scores: Raw Fisher-z transformed bootstrap distribution.
    """

    predicted_rho: float
    ci_lower: float
    ci_upper: float
    z_scores: np.ndarray


def compute_split_half_reliability(
    time_series: np.ndarray, use_shrinkage: bool = True
) -> float:
    """
    Compute split-half reliability from time series data.

    Splits the time series into two halves, computes functional connectivity
    matrices for each half using shrinkage covariance estimation, and returns
    their correlation.

    Args:
        time_series: Time series data with shape (n_samples, n_rois).
        use_shrinkage: Whether to use Ledoit-Wolf shrinkage for covariance.

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
    r_split = np.clip(r_split, 0.001, 0.999)

    logger.debug(f"Split-half reliability: {r_split:.4f}")

    return r_split


def run_bootstrap_prediction(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    config: BSNetConfig | None = None,
) -> BootstrapResult:
    """
    Execute bootstrap prediction pipeline for reliability estimation.

    Runs the full bootstrap loop to estimate the true correlation and
    confidence interval by repeatedly sampling from short observations
    and applying attenuation correction.

    Args:
        short_obs: Short observation time series with shape (n_samples, n_rois).
        fc_reference: Reference functional connectivity vector (upper triangle).
        config: Configuration object. If None, uses BSNetConfig() defaults.

    Returns:
        BootstrapResult: Named tuple containing predicted_rho, confidence
            interval bounds, and the full Fisher-z bootstrap distribution.

    Raises:
        ValueError: If short_obs or fc_reference have invalid shapes.
        TypeError: If inputs are not numpy arrays.
    """
    if config is None:
        config = BSNetConfig()

    if not isinstance(short_obs, np.ndarray):
        raise TypeError(f"short_obs must be numpy array, got {type(short_obs)}")
    if not isinstance(fc_reference, np.ndarray):
        raise TypeError(f"fc_reference must be numpy array, got {type(fc_reference)}")

    if short_obs.ndim != 2:
        raise ValueError(
            f"short_obs must be 2D (n_samples, n_rois), got shape {short_obs.shape}"
        )

    t_samples = short_obs.shape[0]
    logger.info(
        f"Starting bootstrap prediction: t_samples={t_samples}, "
        f"target_samples={config.target_samples}, "
        f"n_bootstraps={config.n_bootstraps}"
    )

    # Estimate optimal block length for dependent sampling
    block_size = estimate_optimal_block_length(short_obs)
    logger.debug(f"Estimated optimal block length: {block_size} TRs")

    rho_hat_b = []
    empirical_prior = config.empirical_prior

    for _b in range(config.n_bootstraps):
        # Block bootstrap resampling
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]

        # Compute observed correlation between reference and bootstrap sample
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs_t = np.corrcoef(fc_reference, fc_obs_t)[0, 1]

        # Compute split-half reliability
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=True)

        # Attenuation correction with Bayesian prior
        rho_est_T = correct_attenuation(
            r_obs_t,
            config.reliability_coeff,
            r_split_t,
            k=config.k_factor,
            empirical_prior=empirical_prior,
        )

        # Fisher-z transformation for distribution normalization
        rho_hat_b.append(fisher_z(rho_est_T))

    rho_hat_b = np.array(rho_hat_b)

    # Compute point estimate as median of bootstrap distribution
    rho_hat_T_z = np.nanmedian(rho_hat_b)
    predicted_rho = fisher_z_inv(rho_hat_T_z)

    # Compute 95% confidence interval
    z_lower, z_upper = np.percentile(rho_hat_b, [2.5, 97.5])
    ci_lower = fisher_z_inv(z_lower)
    ci_upper = fisher_z_inv(z_upper)

    logger.info(
        f"Bootstrap complete: predicted_rho={predicted_rho:.4f}, "
        f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
    )

    return BootstrapResult(
        predicted_rho=float(predicted_rho),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        z_scores=rho_hat_b,
    )
