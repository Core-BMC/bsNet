
import numpy as np


def fisher_z(r):
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_inv(z):
    return np.tanh(z)

def spearman_brown(r_t, k):
    r_t = np.clip(r_t, 0.001, 0.999)
    return (k * r_t) / (1 + (k - 1) * r_t)

def correct_attenuation(r_obs_t, r_hat_t, r_real_t, k, empirical_prior=None):
    """Correct attenuation in observed correlation and extrapolate to target duration.

    Args:
        r_obs_t: Observed correlation between reference FC and bootstrap sample FC.
        r_hat_t: Within-session scanner measurement reliability (default 0.98).
            Represents fMRI scanner precision per Friedman et al. (2008),
            DOI: 10.1016/j.neuroimage.2008.02.005.
        r_real_t: Split-half reliability of the bootstrap sample.
        k: Spearman-Brown scaling factor (target_duration / short_duration).
        empirical_prior: Optional (mean, var) tuple for Bayesian stabilization
            of split-half reliability estimates.

    Returns:
        float: Attenuation-corrected, Spearman-Brown extrapolated reliability ρ̂T,
            clipped to [-1.0, 1.0].
    """
    min_rel = 0.05

    # Bayesian Prior Update (Stabilizing split-half calculation bias)
    if empirical_prior is not None:
        prior_mean, prior_var = empirical_prior
        # Assuming observation is noisy, giving it high observation_var
        observation_var = 0.15
        weight = prior_var / (prior_var + observation_var)
        r_real_t = weight * r_real_t + (1 - weight) * prior_mean

    r_hat_t = max(r_hat_t, min_rel)
    r_real_t = max(r_real_t, min_rel)

    r_true_t = r_obs_t / np.sqrt(r_hat_t * r_real_t)

    r_hat_T = spearman_brown(r_hat_t, k)
    r_real_T = spearman_brown(r_real_t, k)

    rho_hat_T = r_true_t * np.sqrt(r_hat_T * r_real_T)
    return np.clip(rho_hat_T, -1.0, 1.0)

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
