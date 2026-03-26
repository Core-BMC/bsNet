import sys
import os
import numpy as np
import nibabel as nib
from pathlib import Path

# Connect to local MoBSE repo
sys.path.append("/Users/hwon/GitHub/MoBSE")
try:
    from mobse.data.prepare import _download_openneuro_rest_bold
    from mobse.data.nuisance import build_paper_nuisance_confounds
    print("Successfully connected to local MoBSE repository.")
except ImportError as e:
    print(f"Error importing from MoBSE: {e}")
    sys.exit(1)

from data_loader import fetch_schaefer_atlas, create_masker, get_fc_matrix
from bootstrap import (
    fisher_z, fisher_z_inv, 
    correct_attenuation, 
    estimate_optimal_block_length,
    block_bootstrap_indices
)

def run_empirical_pipeline():
    print("--- Phase 2: OpenNeuro Real Data Integration ---")
    
    cache_dir = Path("data/openneuro")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch empirical data using MoBSE optimized logic
    print("Fetching ds000030 adults (HC)...")
    fetch_res = _download_openneuro_rest_bold(
        dataset_id="ds000030",
        snapshot_tag=None,
        task="rest",
        n_subjects=1, # fetch 1 subject to test pipeline
        min_age=18,
        diagnosis="CONTROL",
        strict_hc=True,
        api_url="https://openneuro.org/crn/graphql",
        cache_root=cache_dir,
        progress=None
    )
    
    bold_files = fetch_res["bold_files"]
    if not bold_files:
        print("No BOLD files downloaded.")
        return
        
    target_bold = bold_files[0]
    print(f"Process BOLD target: {target_bold}")
    
    # Extract TR
    img = nib.load(target_bold)
    tr = img.header.get_zooms()[3]
    if tr <= 0.0 or tr > 5.0:
        tr = 2.0 # fallback
        
    print(f"Extracted TR: {tr}s")
    
    # 2. Nuisance Regression
    print("Applying MoBSE CompCor/GSR Denoising...")
    nuisance = build_paper_nuisance_confounds(
        bold_path=target_bold,
        tr=tr,
        external_confounds=None, # Pure data-driven
        include_compcor=True,
        compcor_components=5,
        include_gsr=True,
        add_derivatives=True,
        add_quadratic=True
    )
    
    # 3. Atlas Masking
    print("Extracting Schaefer 100 networks...")
    atlas = fetch_schaefer_atlas(n_rois=100)
    masker = create_masker(
        atlas.maps, 
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.008,
        t_r=tr
    )
    
    time_series = masker.fit_transform(target_bold, confounds=nuisance)
    T_samples = time_series.shape[0]
    total_min = (T_samples * tr) / 60
    
    print(f"Extracted Clean Time-series: {time_series.shape} ({total_min:.1f} minutes total)")
    
    # 4. BS-NET Execution
    short_len_min = 2.0
    t_samples = int(short_len_min * 60 / tr)
    
    if t_samples >= T_samples:
        print(f"Warning: Full scan is only {total_min:.1f}m, which is too short for validation. Exiting.")
        return
        
    # Oracle Truth
    fc_true_T = get_fc_matrix(time_series, vectorized=True)
    
    # Short Slice
    short_obs = time_series[:t_samples, :]
    
    print("\n--- Running BS-NET Core Prediction ---")
    block_size = estimate_optimal_block_length(short_obs)
    print(f"Optimal Block Length: {block_size} TRs")
    
    n_bootstraps = 100
    rho_hat_b = []
    
    # Empirical Fake-Oracle Model for test (mock prediction from neural network)
    fc_pred_t = fc_true_T + 0.1 * np.random.randn(*fc_true_T.shape)
    empirical_prior = (0.25, 0.05)
    
    for b in range(n_bootstraps):
        idx = block_bootstrap_indices(t_samples, block_size, n_blocks=t_samples // block_size)
        ts_b = short_obs[idx, :]
        
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs_t = np.corrcoef(fc_pred_t, fc_obs_t)[0, 1]
        
        n_split = ts_b.shape[0] // 2
        fc_split1 = get_fc_matrix(ts_b[:n_split, :], vectorized=True, use_shrinkage=True)
        fc_split2 = get_fc_matrix(ts_b[n_split:, :], vectorized=True, use_shrinkage=True)
        r_split_t = np.corrcoef(fc_split1, fc_split2)[0, 1]
        
        r_hat_t = 0.98
        
        rho_est_T = correct_attenuation(
            r_obs_t, r_hat_t, r_split_t, 
            k=T_samples/t_samples, 
            empirical_prior=empirical_prior
        )
        rho_hat_b.append(fisher_z(rho_est_T))
        
    rho_hat_T_z = np.nanmedian(rho_hat_b)
    rho_hat_T_final = fisher_z_inv(rho_hat_T_z)
    
    z_lower, z_upper = np.percentile(rho_hat_b, [2.5, 97.5])
    ci_lower = fisher_z_inv(z_lower)
    ci_upper = fisher_z_inv(z_upper)
    
    actual_rho_T = np.corrcoef(fc_pred_t, fc_true_T)[0, 1]
    
    print(f"========== [Final Results (Empirical Data)] ==========")
    print(f"Subject Scan Length: {total_min:.1f}m (TR: {tr}s)")
    print(f"True Oracle \u03C1*(T):      {actual_rho_T:.4f}")
    if rho_hat_T_final > 0.8:
        print(f"Predicted \u03C1*(T):       {rho_hat_T_final:.4f}  \u2705 (>80% Target Goal Reached)")
    else:
        print(f"Predicted \u03C1*(T):       {rho_hat_T_final:.4f}  \u26A0\uFE0F (Below 80% target)")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print("======================================================")

if __name__ == "__main__":
    np.random.seed(42)
    run_empirical_pipeline()
