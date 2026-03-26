import numpy as np


def generate_synthetic_timeseries(n_samples, n_rois, noise_level=1.0, ar1=0.5):
    """
    Generate synthetic fMRI-like time series with temporal autocorrelation
    and spatial covariance structure.
    """
    # Base true signal component
    signal = np.random.randn(n_rois, n_samples)

    # Add temporal autocorrelation (AR1 process) to mimic slow drift/BOLD
    for t in range(1, n_samples):
        signal[:, t] = ar1 * signal[:, t-1] + (1 - ar1) * np.random.randn(n_rois)

    # Introduce 7 true spatial correlation blocks (mimicking Yeo 7-networks)
    mixing_matrix = np.eye(n_rois)
    block_size = n_rois // 7
    for i in range(7):
        start = i * block_size
        end = (i + 1) * block_size if i < 6 else n_rois
        mixing_matrix[start:end, start:end] += 0.8 * np.random.rand(end-start, end-start)
    mixing_matrix += 0.1 * np.random.randn(n_rois, n_rois) # noise
    signal = mixing_matrix @ signal

    # Add noise
    noise = noise_level * np.random.randn(n_rois, n_samples)
    observed = signal + noise

    return observed, signal
