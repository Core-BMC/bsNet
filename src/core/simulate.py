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
        
    # Introduce some true spatial correlation (blocky structure)
    mixing_matrix = np.eye(n_rois) + 0.3 * np.random.randn(n_rois, n_rois)
    signal = mixing_matrix @ signal
    
    # Add noise
    noise = noise_level * np.random.randn(n_rois, n_samples)
    observed = signal + noise
    
    return observed, signal
