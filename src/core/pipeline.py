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
    """Result container for bootstrap extrapolation analysis.

    Attributes:
        rho_hat_T: Extrapolated reliability — point estimate of the
            attenuation-corrected, Spearman-Brown projected reliability
            at target duration T (median of bootstrap distribution).
        ci_lower: Lower bound of 95% confidence interval.
        ci_upper: Upper bound of 95% confidence interval.
        z_scores: Raw Fisher-z transformed bootstrap distribution.
    """

    rho_hat_T: float
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
    correction_method: str = "original",
    fisher_z_fc: bool = False,
    partial_corr: bool = False,
) -> BootstrapResult:
    """
    Execute bootstrap prediction pipeline for reliability estimation.

    Runs the full bootstrap loop to estimate the true correlation and
    confidence interval by repeatedly sampling from short observations
    and applying attenuation correction.

    Args:
        short_obs: Short observation time series with shape (n_samples, n_rois).
        fc_reference: Reference functional connectivity vector (upper triangle).
            Should be in Fisher z-space if fisher_z_fc=True, or in partial
            correlation space if partial_corr=True.
        config: Configuration object. If None, uses BSNetConfig() defaults.
        correction_method: Attenuation correction method (default "original").
        fisher_z_fc: If True, compute bootstrap FC in Fisher z-space (arctanh)
            to match a z-transformed fc_reference. Ensures both vectors are
            in the same space before computing r_obs_t. Default False for
            backward compatibility.
        partial_corr: If True, compute bootstrap FC using partial correlation
            (LW precision matrix) instead of Pearson r. fc_reference must also
            be computed with partial_corr=True. Default False for backward compat.

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

        # Compute observed correlation between reference and bootstrap sample.
        # fisher_z_fc / partial_corr must match how fc_reference was computed
        # to ensure both vectors are in the same space.
        fc_obs_t = get_fc_matrix(
            ts_b,
            vectorized=True,
            use_shrinkage=True,
            fisher_z=fisher_z_fc,
            partial_corr=partial_corr,
        )
        r_obs_t = np.corrcoef(fc_reference, fc_obs_t)[0, 1]

        # Compute split-half reliability
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=True)

        # Attenuation correction with Bayesian prior
        # reliability_coeff: within-session scanner reliability (Friedman 2008)
        rho_est_T = correct_attenuation(
            r_obs_t,
            config.reliability_coeff,
            r_split_t,
            k=config.k_factor,
            empirical_prior=empirical_prior,
            method=correction_method,
        )

        # Fisher-z transformation for distribution normalization
        rho_hat_b.append(fisher_z(rho_est_T))

    rho_hat_b = np.array(rho_hat_b)

    # Compute point estimate as median of bootstrap distribution
    rho_hat_T_z = np.nanmedian(rho_hat_b)
    rho_hat_T = fisher_z_inv(rho_hat_T_z)

    # Compute 95% confidence interval
    z_lower, z_upper = np.percentile(rho_hat_b, [2.5, 97.5])
    ci_lower = fisher_z_inv(z_lower)
    ci_upper = fisher_z_inv(z_upper)

    logger.info(
        f"Bootstrap complete: rho_hat_T={rho_hat_T:.4f}, "
        f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
    )

    return BootstrapResult(
        rho_hat_T=float(rho_hat_T),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        z_scores=rho_hat_b,
    )


def run_sliding_window_prediction(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    config: BSNetConfig | None = None,
    correction_method: str = "fisher_z",
    fisher_z_fc: bool = False,
    partial_corr: bool = False,
    window_sec: float | None = None,
    step_sec: float | None = None,
) -> BootstrapResult:
    """Sliding-window + bootstrap combined BS-NET pipeline.

    Adds an outer sliding-window loop on top of the full bootstrap pipeline:

        short_obs (e.g. 120 s)
          └─ sliding windows  (e.g. 60 s, step 15 s)
               └─ each window → run_bootstrap_prediction()
                    (bootstrap + SB prophecy + Bayesian prior + attenuation)
                    → rho_hat_T_i  (per-window reliability estimate)
          └─ median over windows → final rho_hat_T

    Rationale:
        - Tests whether shorter sub-windows (~60 s) of the original scan
          can still yield reliable predictions via the full BS-NET pipeline.
        - Averaging across overlapping windows reduces variance from any
          single window's temporal sampling.
        - Each window runs the complete bootstrap loop independently, so
          SB k-factor is calibrated to the window duration, not the full
          short scan.

    Window design:
        window_sec  default: short_duration_sec / 2  (e.g. 60 s for 120 s scan)
        step_sec    default: window_sec / 4           (75% overlap)
        k_window    = target_samples / window_vols    (SB scales from window)

    Args:
        short_obs: Short observation time series, shape (n_samples, n_rois).
        fc_reference: Reference FC vector (upper triangle). Must match
            the fisher_z_fc / partial_corr space as fc_reference in
            run_bootstrap_prediction.
        config: BSNetConfig for base parameters (tr, target_duration_min,
            n_bootstraps, reliability_coeff, empirical_prior, seed).
            short_duration_sec is overridden per window. If None, uses defaults.
        correction_method: Attenuation correction method (default "fisher_z").
        fisher_z_fc: If True, compute window FC in Fisher z-space.
        partial_corr: If True, use partial correlation for window FC.
        window_sec: Window duration in seconds.
            Default: config.short_duration_sec / 2.
        step_sec: Step (stride) between windows in seconds.
            Default: window_sec / 4 (75% overlap).

    Returns:
        BootstrapResult:
            rho_hat_T — median over per-window bootstrap estimates.
            ci_lower / ci_upper — percentile CI over per-window estimates.
            z_scores — Fisher-z transformed per-window medians.

    Raises:
        ValueError: If no valid windows can be constructed or window_sec
            exceeds the scan length.
    """
    if config is None:
        config = BSNetConfig()

    if not isinstance(short_obs, np.ndarray):
        raise TypeError(f"short_obs must be numpy array, got {type(short_obs)}")
    if short_obs.ndim != 2:
        raise ValueError(
            f"short_obs must be 2D (n_samples, n_rois), got {short_obs.shape}"
        )

    n_samples = short_obs.shape[0]
    tr        = config.tr

    # ── Window / step sizes ───────────────────────────────────────────────────
    if window_sec is None:
        window_sec = config.short_duration_sec / 2.0
    if step_sec is None:
        step_sec = window_sec / 4.0

    window_vols = max(int(window_sec / tr), 4)   # minimum 4 TRs for stability
    step_vols   = max(int(step_sec   / tr), 1)

    if window_vols > n_samples:
        raise ValueError(
            f"window_vols={window_vols} ({window_sec}s) exceeds "
            f"n_samples={n_samples}. Reduce window_sec."
        )

    # Per-window config: short_duration_sec → window_sec so SB k is correct
    window_config = BSNetConfig(
        n_rois=config.n_rois,
        tr=tr,
        short_duration_sec=int(window_sec),   # k = target / window
        target_duration_min=config.target_duration_min,
        noise_level=config.noise_level,
        ar1=config.ar1,
        n_networks=config.n_networks,
        n_bootstraps=config.n_bootstraps,
        reliability_coeff=config.reliability_coeff,
        empirical_prior=config.empirical_prior,
        seed=config.seed,
        fc_density=config.fc_density,
    )

    logger.info(
        f"Sliding-window+bootstrap: n_samples={n_samples}, "
        f"window={window_vols}TRs ({window_sec}s), step={step_vols}TRs, "
        f"k_window={window_config.k_factor:.2f}"
    )

    # ── Outer sliding-window loop ─────────────────────────────────────────────
    window_rho_z: list[float] = []
    start     = 0
    n_windows = 0
    n_failed  = 0

    while start + window_vols <= n_samples:
        ts_w = short_obs[start : start + window_vols, :]

        try:
            # Full bootstrap pipeline on this window
            win_result = run_bootstrap_prediction(
                ts_w,
                fc_reference,
                config=window_config,
                correction_method=correction_method,
                fisher_z_fc=fisher_z_fc,
                partial_corr=partial_corr,
            )
            # Accumulate in Fisher-z space for numerically stable aggregation
            window_rho_z.append(fisher_z(win_result.rho_hat_T))
            n_windows += 1
        except Exception as exc:                           # noqa: BLE001
            logger.warning(f"Window start={start} failed: {exc}")
            n_failed += 1

        start += step_vols

    if len(window_rho_z) == 0:
        raise ValueError(
            f"No valid windows produced (n_failed={n_failed}). "
            "Check window_sec vs scan length or n_bootstraps."
        )

    logger.info(
        f"Sliding-window+bootstrap complete: "
        f"{n_windows} windows (failed={n_failed})"
    )

    # ── Aggregate over windows in Fisher-z space ──────────────────────────────
    z_arr = np.array(window_rho_z)

    rho_hat_T_z = float(np.nanmedian(z_arr))
    rho_hat_T   = fisher_z_inv(rho_hat_T_z)

    # CI from window-to-window variability (not within-window bootstrap CI)
    if len(z_arr) >= 4:
        z_lo, z_hi = np.percentile(z_arr, [2.5, 97.5])
    else:
        # Too few windows for meaningful percentile CI — use ±1 SD
        z_lo = rho_hat_T_z - float(np.std(z_arr))
        z_hi = rho_hat_T_z + float(np.std(z_arr))

    ci_lower = fisher_z_inv(z_lo)
    ci_upper = fisher_z_inv(z_hi)

    logger.info(
        f"rho_hat_T={rho_hat_T:.4f}, "
        f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}], "
        f"n_windows={n_windows}"
    )

    return BootstrapResult(
        rho_hat_T=float(rho_hat_T),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        z_scores=z_arr,
    )
