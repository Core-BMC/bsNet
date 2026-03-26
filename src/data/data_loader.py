import nilearn.datasets
import numpy as np
from nilearn.maskers import NiftiLabelsMasker
from sklearn.covariance import LedoitWolf


def fetch_schaefer_atlas(n_rois=400, resolution=2, yeo_networks=7):
    """
    Fetch Schaefer 2018 atlas using nilearn.
    """
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

def get_fc_matrix(time_series, vectorized=True, use_shrinkage=False):
    """
    Compute Pearson correlation matrix (FC) from time series.
    Assumes time_series is shape (n_samples, n_rois).
    If use_shrinkage is True, use Ledoit-Wolf estimation for robustness on short data.
    """
    if time_series.shape[0] < 3:
        # Edge case: avoid breaking
        res = np.zeros((time_series.shape[1], time_series.shape[1]))
        if vectorized:
            return res[np.triu_indices_from(res, k=1)]
        return res

    if use_shrinkage:
        lw = LedoitWolf()
        cov_matrix = lw.fit(time_series).covariance_
        d = np.sqrt(np.diag(cov_matrix))
        d[d == 0] = 1e-10
        corr_matrix = cov_matrix / np.outer(d, d)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    else:
        corr_matrix = np.corrcoef(time_series.T)

    np.fill_diagonal(corr_matrix, 0)

    if vectorized:
        indices = np.triu_indices_from(corr_matrix, k=1)
        return corr_matrix[indices]

    return corr_matrix
