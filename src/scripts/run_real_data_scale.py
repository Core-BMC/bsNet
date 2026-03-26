import sys
import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from datetime import datetime

# Connect to MoBSE
sys.path.append("/Users/hwon/GitHub/MoBSE")
try:
    from mobse.data.prepare import _download_openneuro_rest_bold_multi
    from mobse.data.nuisance import build_paper_nuisance_confounds
except ImportError as e:
    print(f"Error importing from MoBSE: {e}")
    sys.exit(1)

from src.data.data_loader import fetch_schaefer_atlas, create_masker, get_fc_matrix
from src.core.bootstrap import (
    fisher_z, fisher_z_inv, 
    correct_attenuation, 
    estimate_optimal_block_length,
    block_bootstrap_indices
)

class DummyProgress:
    def update(self, stage, current=None, total=None, message=""):
        if "openneuro_index" not in stage or ("tree_calls" in message and int(message.split("=")[-1]) % 100 == 0):
            print(f"[{stage}] {message}")

def run_scale_up_pipeline():
    print("--- Phase 3: Scale-up OpenNeuro Validation (n=100) ---")
    
    cache_dir = Path("data/openneuro")
    out_dir = Path("artifacts/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_ids = ["ds000030", "ds000243", "ds002790", "ds002785"]
    
    print(f"Targeting {len(dataset_ids)} Datasets: {dataset_ids}")
    print("Fetching up to 100 Adult (HC) subjects ...")
    
    diagnosis_terms = "normal, healthy control, control, ctrl, nor, hc"
    
    fetch_res = _download_openneuro_rest_bold_multi(
        dataset_ids=dataset_ids,
        snapshot_tag=None,
        task="rest,restingstate", 
        n_subjects=100,
        min_age=18,
        diagnosis=diagnosis_terms,
        strict_hc=False, 
        api_url="https://openneuro.org/crn/graphql",
        cache_root=cache_dir,
        progress=DummyProgress()
    )
    
    records = fetch_res["records"]
    collected = len(records)
    collected = len(records)
    print(f"\nTotal Collected Subjects: {collected} / 100")
    if collected == 0:
        print("No subjects collected. Exiting.")
        return

    # Load shared Schaefer 100-ROI Atlas
    print("Pre-fetching Schaefer 100 Atlas...")
    atlas = fetch_schaefer_atlas(n_rois=100)
    
    results = []
    
    # Process each subject
    for i, rec in enumerate(records, start=1):
        target_bold = rec["bold_file"]
        sub_key = rec["subject_key"]
        print(f"\n[{i}/{collected}] Processing {sub_key} ...")
        
        try:
            img = nib.load(target_bold)
            tr = img.header.get_zooms()[3]
            if tr <= 0.0 or tr > 5.0: tr = 2.0
            
            # Denoise
            nuisance = build_paper_nuisance_confounds(
                bold_path=target_bold, tr=tr, 
                include_compcor=True, compcor_components=5,
                include_gsr=True, add_derivatives=True, add_quadratic=True
            )
            
            masker = create_masker(
                atlas.maps, standardize="zscore_sample", detrend=True,
                low_pass=0.1, high_pass=0.008, t_r=tr
            )
            
            time_series = masker.fit_transform(target_bold, confounds=nuisance)
            T_samples = time_series.shape[0]
            total_min = (T_samples * tr) / 60.0
            
            if total_min < 4.0:
                print(f" => Skip: Scan too short ({total_min:.1f}m)")
                continue

            short_len_min = 2.0
            t_samples = int(short_len_min * 60 / tr)
            
            # Ground truth metrics
            fc_true_T = get_fc_matrix(time_series, vectorized=True)
            short_obs = time_series[:t_samples, :]
            
            block_size = estimate_optimal_block_length(short_obs)
            fc_pred_t_mock = fc_true_T + 0.1 * np.random.randn(*fc_true_T.shape)
            
            rho_hat_b = []
            empirical_prior = (0.25, 0.05)
            
            # Bootstrap 50 iterations locally to save computation time over 100 subjects
            n_bootstraps = 50 
            for b in range(n_bootstraps):
                idx = block_bootstrap_indices(t_samples, block_size, n_blocks=t_samples // block_size)
                ts_b = short_obs[idx, :]
                
                fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
                r_obs_t = np.corrcoef(fc_pred_t_mock, fc_obs_t)[0, 1]
                
                n_split = ts_b.shape[0] // 2
                r_split_t = np.corrcoef(
                    get_fc_matrix(ts_b[:n_split, :], vectorized=True, use_shrinkage=True), 
                    get_fc_matrix(ts_b[n_split:, :], vectorized=True, use_shrinkage=True)
                )[0, 1]
                
                rho_est_T = correct_attenuation(
                    r_obs_t, 0.98, r_split_t, k=T_samples/t_samples, empirical_prior=empirical_prior
                )
                rho_hat_b.append(fisher_z(rho_est_T))
                
            pred_rho = fisher_z_inv(np.nanmedian(rho_hat_b))
            true_rho = np.corrcoef(fc_pred_t_mock, fc_true_T)[0, 1]
            
            print(f" => True Oracle: {true_rho:.4f} | Predicted: {pred_rho:.4f}")
            
            results.append({
                "subject": sub_key,
                "dataset": rec["dataset_id"],
                "scan_min": total_min,
                "tr": tr,
                "true_rho": true_rho,
                "pred_rho": pred_rho,
                "error": abs(true_rho - pred_rho),
                "is_above_80": pred_rho >= 0.8
            })
            
        except Exception as e:
            print(f" => Failed: {e}")
            
    if not results:
        print("\nNo completely processed results.")
        return
        
    df = pd.DataFrame(results)
    csv_path = out_dir / f"scale_up_100_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    
    mean_true = df["true_rho"].mean()
    mean_pred = df["pred_rho"].mean()
    mean_err = df["error"].mean()
    success_rate = (df["is_above_80"].sum() / len(df)) * 100
    
    print("\n========== [Phase 3 Scale-Up Aggregation] ==========")
    print(f"Processed / Requested: {len(df)} / 100")
    print(f"Mean True Oracle: {mean_true:.4f}")
    if mean_pred >= 0.8:
        print(f"Mean Predicted Target: {mean_pred:.4f} \u2705 (>80% Maintained!)")
    else:
        print(f"Mean Predicted Target: {mean_pred:.4f} \u26A0\uFE0F")
    print(f"Accuracy >80% Ratio:   {success_rate:.1f}%")
    print(f"Mean Absolute Error:   {mean_err:.4f}")
    print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    np.random.seed(42)
    run_scale_up_pipeline()
