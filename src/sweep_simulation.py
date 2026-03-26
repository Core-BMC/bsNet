import numpy as np
import pandas as pd
from simulate import generate_synthetic_timeseries
from data_loader import get_fc_matrix
from bootstrap import (
    fisher_z, fisher_z_inv, 
    correct_attenuation, 
    estimate_optimal_block_length,
    block_bootstrap_indices
)

def run_duration_sweep():
    # Canonical theoretical simulation parameters
    TR = 1.0
    T_minutes = 15
    T_samples = int(T_minutes * 60 / TR)
    n_rois = 50
    
    sweep_seconds = [30, 60, 90, 120, 150, 180, 210, 240]
    seeds = [42, 123, 777, 2026, 9999]
    
    print("--- BS-NET Time Duration Sweep (Multi-Seed Validation) ---")
    print(f"Full Target: {T_minutes} minutes ({T_samples} TRs, TR={TR}s)")
    print("Sweeping Durations:", sweep_seconds, "seconds")
    print("Evaluating Across Seeds:", seeds, "\n")
    
    all_results = []
    
    for seed in seeds:
        np.random.seed(seed)
        long_obs, long_signal = generate_synthetic_timeseries(n_samples=T_samples, n_rois=n_rois, noise_level=0.25, ar1=0.6)
        time_series = long_obs.T
        pure_signal = long_signal.T
        
        fc_true_T = get_fc_matrix(pure_signal, vectorized=True)
        fc_pred_t = fc_true_T + 0.1 * np.random.randn(*fc_true_T.shape)
        
        for t_sec in sweep_seconds:
            t_samples = int(t_sec / TR)
            short_obs = time_series[:t_samples, :]
            
            block_size = estimate_optimal_block_length(short_obs)
            if block_size < 2: block_size = 2
            
            n_bootstraps = 100
            rho_hat_b = []
            empirical_prior = (0.25, 0.05)
            
            for b in range(n_bootstraps):
                idx = block_bootstrap_indices(t_samples, block_size, n_blocks=max(1, t_samples // block_size))
                if len(idx) < t_samples: pass
                ts_b = short_obs[idx, :]
                
                fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
                r_obs_t = np.corrcoef(fc_pred_t, fc_obs_t)[0, 1]
                
                n_split = ts_b.shape[0] // 2
                fc_split1 = get_fc_matrix(ts_b[:n_split, :], vectorized=True, use_shrinkage=True)
                fc_split2 = get_fc_matrix(ts_b[n_split:, :], vectorized=True, use_shrinkage=True)
                r_split_t = np.corrcoef(fc_split1, fc_split2)[0, 1]
                
                if r_split_t <= 0.05: r_split_t = 0.05
                r_hat_t = 0.98
                k_factor = T_samples / t_samples
                
                rho_est_T = correct_attenuation(
                    r_obs_t, r_hat_t, r_split_t, k=k_factor, empirical_prior=empirical_prior
                )
                rho_hat_b.append(fisher_z(rho_est_T))
                
            rho_hat_T_z = np.nanmedian(rho_hat_b)
            pred_rho = fisher_z_inv(rho_hat_T_z)
            
            z_lower, z_upper = np.percentile(rho_hat_b, [2.5, 97.5])
            ci_lower = fisher_z_inv(z_lower)
            ci_upper = fisher_z_inv(z_upper)
            
            actual_rho_T = np.corrcoef(fc_pred_t, fc_true_T)[0, 1]
            error = abs(actual_rho_T - pred_rho)
            
            all_results.append({
                "Seed": seed,
                "Duration (s)": t_sec,
                "Predicted": pred_rho,
                "Error": error,
                "CI Lower": ci_lower,
                "CI Upper": ci_upper
            })
            
    # Aggregate results over seeds
    df = pd.DataFrame(all_results)
    agg_df = df.groupby("Duration (s)").mean().reset_index()
    
    print(f"{'Dur(s)':<8} | {'Pred_Mean':<10} | {'Err_Mean':<10} | {'CI Lower':<10} | {'CI Upper':<10}")
    print("-" * 65)
    for _, row in agg_df.iterrows():
        print(f"{int(row['Duration (s)']):<8} | {row['Predicted']:.4f}     | {row['Error']:.4f}     | {row['CI Lower']:.4f}     | {row['CI Upper']:.4f}")
        
    df.to_csv("artifacts/reports/duration_sweep_seeds_results.csv", index=False)
    agg_df.to_csv("artifacts/reports/duration_sweep_seeds_aggregated.csv", index=False)
    print("\nSweep Complete! Saved detailed and aggregated CSVs to artifacts/reports/")

if __name__ == "__main__":
    run_duration_sweep()
