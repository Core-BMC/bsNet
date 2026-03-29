"""
Bootstrap resampling and attenuation correction for BS-NET.

Implements block bootstrap, Spearman-Brown prophecy extrapolation,
and multiple attenuation correction methods including Fisher z-space
correction to address the overcorrection / ceiling effect problem.

References:
    - Shou et al. (2014): Reliability correction for FC. DOI: 10.1016/j.neuroimage.2015.10.011
    - Teeuw et al. (2021): Reliability modelling of resting-state FC. DOI: 10.1016/j.neuroimage.2021.117842
    - Spearman (1904), Brown (1910): Correction for attenuation / prophecy formula.
    - Zimmerman (2007): Attenuation with biased reliability estimates. DOI: 10.1177/0013164406299132
"""

import numpy as np


# ============================================================================
# Fisher z transforms
# ============================================================================
def fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_inv(z):
    return np.tanh(z)


# ============================================================================
# Spearman-Brown prophecy formula
# ============================================================================
def spearman_brown(r_t, k):
    r_t = np.clip(r_t, 0.001, 0.999)
    return (k * r_t) / (1 + (k - 1) * r_t)


# ============================================================================
# Correction methods (enum-like)
# ============================================================================
CORRECTION_METHODS = ("original", "fisher_z", "partial", "soft_clamp")


# ============================================================================
# Original attenuation correction (baseline)
# ============================================================================
def correct_attenuation(r_obs_t, r_hat_t, r_real_t, k, empirical_prior=None,
                        method="original"):
    """Correct attenuation in observed correlation and extrapolate to target duration.

    Supports multiple correction methods to address the overcorrection / ceiling
    effect that occurs when r_obs is large relative to estimated reliability.

    Args:
        r_obs_t: Observed correlation between reference FC and bootstrap sample FC.
        r_hat_t: Within-session scanner measurement reliability (default 0.98).
            Represents fMRI scanner precision per Friedman et al. (2008),
            DOI: 10.1016/j.neuroimage.2008.02.005.
        r_real_t: Split-half reliability of the bootstrap sample.
        k: Spearman-Brown scaling factor (target_duration / short_duration).
        empirical_prior: Optional (mean, var) tuple for Bayesian stabilization
            of split-half reliability estimates.
        method: Correction method. One of:
            - "original": Standard CTT disattenuation + hard clip to [-1, 1].
                (Spearman 1904). Known to produce ceiling effect when r_obs
                is high relative to sqrt(r_hat * r_real).
            - "fisher_z": Fisher z-space disattenuation (Shou et al. 2014,
                Teeuw et al. 2021). Performs all operations in arctanh space
                where the correction is additive and naturally bounded via
                tanh back-transform. Avoids hard clipping.
            - "partial": Partial correction for attenuation (Zimmerman 2007).
                Applies a damped correction factor (alpha=0.5) to avoid
                overcorrecting when reliability estimates are noisy.
            - "soft_clamp": Soft sigmoid clamping. Uses tanh compression
                to smoothly map overcorrected values onto (0, 1) instead of
                hard clipping. Preserves rank ordering of subjects.

    Returns:
        float: Attenuation-corrected, Spearman-Brown extrapolated reliability ρ̂T.
    """
    if method not in CORRECTION_METHODS:
        raise ValueError(f"Unknown method '{method}'. Use one of {CORRECTION_METHODS}")

    # Dispatch to method-specific implementation
    if method == "original":
        return _correct_original(r_obs_t, r_hat_t, r_real_t, k, empirical_prior)
    elif method == "fisher_z":
        return _correct_fisher_z(r_obs_t, r_hat_t, r_real_t, k, empirical_prior)
    elif method == "partial":
        return _correct_partial(r_obs_t, r_hat_t, r_real_t, k, empirical_prior)
    elif method == "soft_clamp":
        return _correct_soft_clamp(r_obs_t, r_hat_t, r_real_t, k, empirical_prior)
    return None  # unreachable


def _apply_bayesian_prior(r_real_t, empirical_prior):
    """Apply Bayesian shrinkage to split-half reliability estimate.

    Args:
        r_real_t: Raw split-half reliability.
        empirical_prior: (mean, var) tuple or None.

    Returns:
        Shrunk reliability estimate.
    """
    if empirical_prior is not None:
        prior_mean, prior_var = empirical_prior
        observation_var = 0.15
        weight = prior_var / (prior_var + observation_var)
        r_real_t = weight * r_real_t + (1 - weight) * prior_mean
    return r_real_t


# ----------------------------------------------------------------------------
# Method 1: Original (baseline) — hard clip at ±1
# ----------------------------------------------------------------------------
def _correct_original(r_obs_t, r_hat_t, r_real_t, k, empirical_prior):
    """Standard CTT correction for attenuation (Spearman 1904)."""
    min_rel = 0.05

    r_real_t = _apply_bayesian_prior(r_real_t, empirical_prior)
    r_hat_t = max(r_hat_t, min_rel)
    r_real_t = max(r_real_t, min_rel)

    r_true_t = r_obs_t / np.sqrt(r_hat_t * r_real_t)

    r_hat_T = spearman_brown(r_hat_t, k)
    r_real_T = spearman_brown(r_real_t, k)

    rho_hat_T = r_true_t * np.sqrt(r_hat_T * r_real_T)
    return np.clip(rho_hat_T, -1.0, 1.0)


# ----------------------------------------------------------------------------
# Method 2: Fisher z-space correction (Shou 2014, Teeuw 2021)
# ----------------------------------------------------------------------------
def _correct_fisher_z(r_obs_t, r_hat_t, r_real_t, k, empirical_prior):
    """Attenuation correction performed entirely in Fisher z-space.

    Instead of dividing correlations (which can exceed 1.0), we work in
    Fisher z (arctanh) space where the correction is approximately additive.
    Back-transforming via tanh naturally bounds results to (-1, 1).

    Derivation:
        In correlation space: r_true = r_obs / sqrt(rel_x * rel_y)
        In z-space (log-odds of correlation):
            z_true ≈ z_obs - 0.5 * log(rel_x * rel_y)
                    = z_obs - 0.5 * (log(rel_x) + log(rel_y))

        For SB extrapolation:
            z_true_T ≈ z_true + 0.5 * (log(rel_x_T * rel_y_T) - log(rel_x * rel_y))
            This simplifies to: z_true_T = z_obs + SB_correction_in_z_space.

    Reference:
        Shou et al. (2014), NeuroImage 118, pp.126-141.
        Teeuw et al. (2021), NeuroImage 231, 117842.
    """
    min_rel = 0.05

    r_real_t = _apply_bayesian_prior(r_real_t, empirical_prior)
    r_hat_t = max(r_hat_t, min_rel)
    r_real_t = max(r_real_t, min_rel)

    # Work in z-space
    z_obs = fisher_z(r_obs_t)

    # Disattenuation in z-space (additive correction)
    # z_true = z_obs - 0.5 * ln(rel_x * rel_y)
    # Since ln(rel) < 0 for rel < 1, this ADDS to z_obs (correction upward)
    rel_product_t = r_hat_t * r_real_t
    z_correction_t = -0.5 * np.log(rel_product_t)

    # SB-extrapolated reliabilities at target duration
    r_hat_T = spearman_brown(r_hat_t, k)
    r_real_T = spearman_brown(r_real_t, k)
    rel_product_T = r_hat_T * r_real_T

    # Re-attenuation at target duration (subtract correction for target reliability)
    z_correction_T = -0.5 * np.log(rel_product_T)

    # z_true_T = z_obs + (disattenuation_at_t) - (re-attenuation_at_T)
    # = z_obs + z_correction_t - z_correction_T
    # = z_obs - 0.5 * ln(rel_t) + 0.5 * ln(rel_T)
    # = z_obs + 0.5 * ln(rel_T / rel_t)
    z_true_T = z_obs + z_correction_t - z_correction_T

    # Back-transform: tanh naturally bounds to (-1, 1)
    return float(fisher_z_inv(z_true_T))


# ----------------------------------------------------------------------------
# Method 3: Partial correction (Zimmerman 2007)
# ----------------------------------------------------------------------------
def _correct_partial(r_obs_t, r_hat_t, r_real_t, k, empirical_prior,
                     alpha=0.5):
    """Partial correction for attenuation with damping factor.

    Instead of full disattenuation (which overcorrects when reliability
    estimates are noisy), applies a fraction alpha of the correction.

    r_true_partial = r_obs / (rel_product ** (alpha/2))

    When alpha=1.0, this equals full correction. When alpha=0.0, no correction.
    alpha=0.5 is the geometric mean of corrected and uncorrected.

    Reference:
        Zimmerman (2007), Educational and Psychological Measurement, 67(6), 920-939.
        McCrae et al. (2011): partial correction recommendation for personality research.
    """
    min_rel = 0.05

    r_real_t = _apply_bayesian_prior(r_real_t, empirical_prior)
    r_hat_t = max(r_hat_t, min_rel)
    r_real_t = max(r_real_t, min_rel)

    # Partial disattenuation
    rel_product_t = r_hat_t * r_real_t
    r_true_t = r_obs_t / (rel_product_t ** (alpha / 2.0))

    # SB extrapolation of reliabilities
    r_hat_T = spearman_brown(r_hat_t, k)
    r_real_T = spearman_brown(r_real_t, k)
    rel_product_T = r_hat_T * r_real_T

    # Partial re-attenuation at target
    rho_hat_T = r_true_t * (rel_product_T ** (alpha / 2.0))
    return float(np.clip(rho_hat_T, -1.0, 1.0))


# ----------------------------------------------------------------------------
# Method 4: Soft sigmoid clamp
# ----------------------------------------------------------------------------
def _correct_soft_clamp(r_obs_t, r_hat_t, r_real_t, k, empirical_prior):
    """Soft clamping via tanh compression instead of hard clip.

    Computes the unclamped attenuation correction, then applies tanh
    compression so that values smoothly approach 1.0 without hard
    discontinuity. Preserves rank ordering of subjects.

    Equivalent to: rho_hat_T = tanh(raw_correction) when raw > 0.
    For raw < 0: -tanh(|raw|).
    """
    min_rel = 0.05

    r_real_t = _apply_bayesian_prior(r_real_t, empirical_prior)
    r_hat_t = max(r_hat_t, min_rel)
    r_real_t = max(r_real_t, min_rel)

    r_true_t = r_obs_t / np.sqrt(r_hat_t * r_real_t)

    r_hat_T = spearman_brown(r_hat_t, k)
    r_real_T = spearman_brown(r_real_t, k)

    # Raw (unclamped) correction
    rho_raw = r_true_t * np.sqrt(r_hat_T * r_real_T)

    # Soft clamp: use tanh which naturally maps R → (-1, 1)
    # tanh(x) ≈ x for small x, and → 1 smoothly for large x
    return float(np.tanh(rho_raw))

def estimate_optimal_block_length(time_series):
    """
    Simplified dynamic block length estimation inspired by Politis & White (2004).
    Calculates average AR(1) coefficient across valid ROIs to appropriately scale the block size.
    """
    n_samples, n_rois = time_series.shape
    if n_samples < 5:
        return n_samples

    ar1_sum = 0
    valid_rois = 0
    for i in range(n_rois):
        std_prev = np.std(time_series[:-1, i])
        std_next = np.std(time_series[1:, i])
        if std_prev > 1e-5 and std_next > 1e-5:
            c = np.corrcoef(time_series[:-1, i], time_series[1:, i])[0, 1]
            if not np.isnan(c):
                ar1_sum += c
                valid_rois += 1

    ar1 = ar1_sum / valid_rois if valid_rois > 0 else 0
    ar1 = max(0.01, min(0.99, ar1))

    # Heuristic block length based on AR(1) decay
    b = int(np.ceil(-2 / np.log(ar1))) * 2
    b = max(5, min(b, max(5, n_samples // 4)))

    return int(b)

def block_bootstrap_indices(n_samples, block_size, n_blocks):
    if block_size > n_samples:
        block_size = n_samples
    if block_size <= 0:
        block_size = 1
    start_indices = np.random.randint(0, n_samples - block_size + 1, size=n_blocks)
    indices = np.concatenate([np.arange(start, start + block_size) for start in start_indices])
    return indices
