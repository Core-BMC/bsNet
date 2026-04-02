import numpy as np

try:
    import nilearn.datasets
    from nilearn.maskers import NiftiLabelsMasker
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False

try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


from pathlib import Path


def load_timeseries_data(
    input_npy: str | None = None,
    n_samples: int = 900,
    n_rois: int = 50,
    noise_level: float = 0.25,
    ar1: float = 0.6,
    short_samples: int = 120,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or generate timeseries data for defense experiments.

    Supports two modes:
    - Real data: load preprocessed .npy file (n_samples, n_rois)
    - Synthetic: generate via generate_synthetic_timeseries()

    Args:
        input_npy: Path to preprocessed .npy timeseries. If None, generate synthetic.
        n_samples: Total samples for synthetic generation (ignored if input_npy).
        n_rois: Number of ROIs for synthetic generation (ignored if input_npy).
        noise_level: Noise level for synthetic data (ignored if input_npy).
        ar1: AR(1) coefficient for synthetic data (ignored if input_npy).
        short_samples: Number of samples for short observation.
        seed: Random seed for synthetic generation (ignored if input_npy).

    Returns:
        Tuple of (ts_full, ts_short, ts_signal):
        - ts_full: Full observation (n_samples, n_rois).
        - ts_short: Short observation (short_samples, n_rois).
        - ts_signal: Noise-free signal (n_samples, n_rois).
          For real data, ts_signal = ts_full (no separate signal available).
    """
    if input_npy is not None:
        npy_path = Path(input_npy)
        if not npy_path.exists():
            raise FileNotFoundError(f"Input .npy not found: {npy_path}")
        ts_full = np.load(str(npy_path)).astype(np.float64)

        # Validate shape: expect (n_samples, n_rois)
        if ts_full.ndim != 2:
            raise ValueError(f"Expected 2D array, got {ts_full.ndim}D: {ts_full.shape}")
        if ts_full.shape[0] < ts_full.shape[1]:
            # Likely (n_rois, n_samples) — transpose
            ts_full = ts_full.T

        # Remove zero-variance ROIs
        valid = np.std(ts_full, axis=0) > 1e-8
        if np.sum(~valid) > 0:
            ts_full = ts_full[:, valid]

        if ts_full.shape[0] < short_samples + 10:
            raise ValueError(
                f"Too few timepoints ({ts_full.shape[0]}) for "
                f"short_samples={short_samples}"
            )

        ts_short = ts_full[:short_samples, :]
        ts_signal = ts_full  # No separate signal for real data

        return ts_full, ts_short, ts_signal

    # Synthetic mode
    from src.core.simulate import generate_synthetic_timeseries

    if seed is not None:
        np.random.seed(seed)

    observed, signal = generate_synthetic_timeseries(
        n_samples, n_rois, noise_level=noise_level, ar1=ar1,
    )
    # generate_synthetic_timeseries returns (n_rois, n_samples) — transpose
    ts_full = observed.T  # (n_samples, n_rois)
    ts_signal = signal.T
    ts_short = ts_full[:short_samples, :]

    return ts_full, ts_short, ts_signal


def fetch_schaefer_atlas(n_rois=400, resolution=2, yeo_networks=7):
    """
    Fetch Schaefer 2018 atlas using nilearn.
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn is required for fetch_schaefer_atlas")
    print(f"Fetching Schaefer 2018 atlas (ROIs: {n_rois}, Networks: {yeo_networks}, Resolution: {resolution}mm)...")
    atlas = nilearn.datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution
    )
    return atlas

def create_masker(atlas_img, standardize="zscore_sample", detrend=True, low_pass=None, high_pass=None, t_r=None):
    """
    Create a NiftiLabelsMasker object with standard preprocessing.
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn is required for create_masker")
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=standardize,
        detrend=detrend,
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=t_r,
        verbose=1
    )
    return masker

def get_fc_matrix(
    time_series,
    vectorized: bool = True,
    use_shrinkage: bool = False,
    fisher_z: bool = False,
    partial_corr: bool = False,
):
    """Compute FC matrix from timeseries.

    Computes Pearson r or partial correlation (optionally with Ledoit-Wolf
    shrinkage), then optionally applies Fisher z-transform (arctanh).

    Method selection:
        - partial_corr=False, use_shrinkage=False : np.corrcoef (Pearson r)
        - partial_corr=False, use_shrinkage=True  : LW regularized Pearson r
        - partial_corr=True,  use_shrinkage=True  : LW precision → partial corr
          (recommended: LW regularization makes precision matrix inversion stable)
        - partial_corr=True,  use_shrinkage=False : pseudo-inverse of np.corrcoef
          (not recommended for short timeseries — ill-conditioned)

    Partial correlation derivation:
        Given precision matrix Θ = Σ⁻¹:
            partial_corr[i,j] = -Θ[i,j] / sqrt(Θ[i,i] * Θ[j,j])
        This reflects direct pairwise connectivity after partialling out
        all other ROIs (nilearn ConnectivityMeasure kind='partial correlation').

    Args:
        time_series: Array of shape (n_samples, n_rois).
        vectorized: If True, return upper-triangle vector; else return matrix.
        use_shrinkage: Use Ledoit-Wolf covariance estimator (recommended for
            short timeseries). Falls back to np.corrcoef if sklearn unavailable.
        fisher_z: If True, apply arctanh to correlation values before
            returning (Fisher z-transform). Values are clipped to ±0.9999
            before transform to avoid ±inf. Default False for backward compat.
        partial_corr: If True, compute partial correlation via precision matrix
            inversion instead of Pearson r. Requires use_shrinkage=True for
            numerical stability with short timeseries (n_samples < n_rois).
            Default False for backward compat.

    Returns:
        FC vector or matrix. Values are in [-1, 1] (Pearson r or partial corr)
        if fisher_z=False, or arctanh-transformed (unbounded) if fisher_z=True.
    """
    if time_series.shape[0] < 3:
        res = np.zeros((time_series.shape[1], time_series.shape[1]))
        if vectorized:
            return res[np.triu_indices_from(res, k=1)]
        return res

    if use_shrinkage and HAS_SKLEARN:
        lw = LedoitWolf()
        cov_matrix = lw.fit(time_series).covariance_

        if partial_corr:
            # Precision matrix = inverse of regularized covariance
            # LW shrinkage ensures positive-definite → stable inversion
            precision = np.linalg.inv(cov_matrix)
            d = np.sqrt(np.diag(precision))
            d[d == 0] = 1e-10
            corr_matrix = -precision / np.outer(d, d)
            np.fill_diagonal(corr_matrix, 0)
        else:
            d = np.sqrt(np.diag(cov_matrix))
            d[d == 0] = 1e-10
            corr_matrix = cov_matrix / np.outer(d, d)
            corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
            np.fill_diagonal(corr_matrix, 0)
    else:
        corr_matrix = np.corrcoef(time_series.T)
        if partial_corr:
            # Use pseudo-inverse (may be unstable for short timeseries)
            precision = np.linalg.pinv(corr_matrix)
            d = np.sqrt(np.diag(precision))
            d[d == 0] = 1e-10
            corr_matrix = -precision / np.outer(d, d)
        np.fill_diagonal(corr_matrix, 0)

    if fisher_z:
        # arctanh(0) = 0 for diagonal entries; clip off-diagonal to ±0.9999
        corr_matrix = np.arctanh(np.clip(corr_matrix, -0.9999, 0.9999))

    if vectorized:
        indices = np.triu_indices_from(corr_matrix, k=1)
        return corr_matrix[indices]

    return corr_matrix
