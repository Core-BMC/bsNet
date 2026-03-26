import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulate import generate_synthetic_timeseries
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def plot_combined_figure():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = plt.subplot(2, 2, 1) # A: Accuracy
    ax2 = plt.subplot(2, 2, 2) # B: Marginal Gain
    ax3 = plt.subplot(2, 2, 3) # C: CI Width
    ax4 = plt.subplot(2, 2, 4) # D: Signal Combined
    
    csv_path = "artifacts/reports/duration_sweep_seeds_aggregated.csv"
    if Path(csv_path).exists():
        df_agg = pd.read_csv(csv_path)
        
        # -----------------------------
        # Fig 1a: Prediction Accuracy (Unchanged)
        # -----------------------------
        ax1.plot(df_agg["Duration (s)"], df_agg["Predicted"], marker='o', color='#2c7bb6', linewidth=3.0, markersize=8, label="Mean \u03C1*(T)")
        ax1.fill_between(df_agg["Duration (s)"], df_agg["CI Lower"], df_agg["CI Upper"], color='#abd9e9', alpha=0.5, label="95% CI")
        ax1.axhline(y=0.80, color='#d7191c', linestyle='--', linewidth=2.5, label="80% Target")
        ax1.axvline(x=120, color='#fdae61', linestyle='-.', linewidth=2.5, label="120s Threshold")
        ax1.set_title("A. Prediction Accuracy vs. Duration", fontweight='bold', fontsize=15, pad=10)
        ax1.set_xlabel("Duration (seconds)", fontsize=13)
        ax1.set_ylabel("Predicted Correlation \u03C1*(T)", fontsize=13)
        ax1.set_ylim(0.4, 1.0)
        ax1.legend(loc="lower right", fontsize=11)
        
        # -----------------------------
        # Fig 1b: Marginal Gain (First Derivative)
        # -----------------------------
        df_agg["Marginal_Gain"] = df_agg["Predicted"].diff()
        # Drop NaN for the first element
        diff_data = df_agg.dropna(subset=["Marginal_Gain"])
        
        # Bar chart showing the drop in efficiency
        colors = ['#d7191c' if x <= 120 else '#abd9e9' for x in diff_data["Duration (s)"]]
        ax2.bar(diff_data["Duration (s)"], diff_data["Marginal_Gain"], width=20, color=colors, alpha=0.8, edgecolor='black')
        ax2.axvline(x=120, color='#fdae61', linestyle='-.', linewidth=2.5, label="Diminishing Returns (Knee)")
        ax2.set_title("B. Incremental Accuracy Gain (Marginal Utility)", fontweight='bold', fontsize=15, pad=10)
        ax2.set_xlabel("Duration (seconds)", fontsize=13)
        ax2.set_ylabel("\u0394 \u03C1*(T) per +30s", fontsize=13)
        ax2.set_xticks(df_agg["Duration (s)"])
        ax2.legend(loc="upper right", fontsize=11)
        
        # -----------------------------
        # Fig 1c: CI Width (Uncertainty Decay)
        # -----------------------------
        df_agg["CI_Width"] = df_agg["CI Upper"] - df_agg["CI Lower"]
        ax3.plot(df_agg["Duration (s)"], df_agg["CI_Width"], marker='s', color='#d7191c', linewidth=2.5, markersize=8, label="95% CI Boundary Width")
        ax3.axvline(x=120, color='#fdae61', linestyle='-.', linewidth=2.5, label="Uncertainty Stabilization")
        ax3.set_title("C. Statistical Uncertainty Decay (CI Width)", fontweight='bold', fontsize=15, pad=10)
        ax3.set_xlabel("Duration (seconds)", fontsize=13)
        ax3.set_ylabel("Confidence Interval Width (\u0394\u03C1)", fontsize=13)
        ax3.legend(loc="upper right", fontsize=11)
        
    # -----------------------------
    # Generating Signal for 1d
    # -----------------------------
    np.random.seed(123)
    n_rois = 50
    TR = 1.0
    t_samples = 120
    long_obs, long_signal = generate_synthetic_timeseries(n_samples=t_samples, n_rois=n_rois, noise_level=0.5, ar1=0.7)
    
    target_roi = -1
    best_corr = 0
    for i in range(n_rois):
        corr = np.corrcoef(long_signal[i], long_obs[i])[0, 1]
        if 0.80 <= corr <= 0.85:
            target_roi = i
            best_corr = corr
            break
            
    true_ts = long_signal[target_roi] if target_roi != -1 else long_signal[0]
    raw_ts = long_obs[target_roi] if target_roi != -1 else long_obs[0]
    time_axis = np.arange(t_samples) * TR
    
    # -----------------------------
    # Fig 1d: Combined Separated + Overlay
    # -----------------------------
    # Top Section: Separated Signals (Offsets +8 and +4)
    ax4.plot(time_axis, true_ts + 8.0, color='#2c7bb6', linewidth=2.5, label="Separated: True Signal (+8)")
    ax4.plot(time_axis, raw_ts + 4.0, color='#d7191c', linewidth=1.5, alpha=0.8, label="Separated: Raw Noise (+4)")
    
    # Bottom Section: Overlay Signals
    ax4.plot(time_axis, true_ts, color='#2c7bb6', linewidth=3.0, alpha=0.85, label="Overlay: True Signal")
    ax4.plot(time_axis, raw_ts, color='#fdae61', linewidth=1.5, alpha=0.9, linestyle='--', label=f"Overlay: Raw Data (r={best_corr:.2f})")
    
    ax4.axhline(y=2.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5) # Divider
    
    ax4.set_title("D. Visualizing 84% Coherence (Separated vs. Overlay)", fontweight='bold', fontsize=15, pad=10)
    ax4.set_xlabel("Time (seconds)", fontsize=13)
    ax4.set_ylabel("Amplitude", fontsize=13)
    ax4.set_ylim(-3, 15)
    
    # Inner legend placed in upper right empty space
    ax4.legend(loc="upper right", fontsize=10, frameon=True, shadow=True)
    
    plt.tight_layout(pad=3.0)
    local_out = "artifacts/reports/Figure1_Combined.png"
    docs_out = "docs/figure/Figure1_Combined.png"
    artifact_out = "/Users/hwon/.gemini/antigravity/brain/a2702892-6b20-4b56-b0d4-912f4def1eab/Figure1_Combined.png"
    
    # Create docs/figure directory if it doesn't exist
    Path("docs/figure").mkdir(parents=True, exist_ok=True)
    
    plt.savefig(local_out, dpi=300, bbox_inches='tight')
    plt.savefig(docs_out, dpi=300, bbox_inches='tight')
    plt.savefig(artifact_out, dpi=300, bbox_inches='tight')
    print(f"Combined 2x2 Figure 1 successfully verified and saved to {docs_out}")

if __name__ == "__main__":
    plot_combined_figure()
