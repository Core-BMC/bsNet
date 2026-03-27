"""Generate Figure 2: Validation metrics for BS-NET prediction accuracy (10-seed).

Creates a 2x2 visualization of prediction accuracy, distribution overlap,
error distribution, and clinical pass rates.

Design: 10 seeds x 20 subjects x 50 ROIs
  - Each subject: generate synthetic long TS → reference FC → mock FC →
    short obs → run_bootstrap_prediction → compare true_rho vs pred_rho
  - Cross-seed variability shown in all panels
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.core.simulate import generate_synthetic_timeseries
from src.data.data_loader import get_fc_matrix
from src.visualization.style import PALETTE, apply_bsnet_theme, save_figure

warnings.filterwarnings("ignore")

# Reproducible 10-seed set (shared across all figures)
SEEDS: list[int] = [42, 123, 777, 2026, 9999, 314, 628, 1414, 2718, 3141]


def run_single_seed(
    seed: int,
    n_subjects: int = 20,
    n_rois: int = 50,
    t_samples_long: int = 450,
    t_samples_short: int = 120,
    noise_level: float = 0.25,
    ar1: float = 0.6,
) -> list[dict]:
    """Run validation for a single seed.

    Mimics run_real_data_scale.py logic:
    1. Generate long TS → reference FC
    2. Mock FC = reference FC + noise
    3. Short obs from long TS
    4. run_bootstrap_prediction(short_obs, mock_fc, config) → rho_hat_T
    5. true_rho = corr(mock_fc, reference_fc)

    Args:
        seed: Random seed.
        n_subjects: Number of subjects per seed.
        n_rois: Number of ROIs.
        t_samples_long: Full scan length.
        t_samples_short: Short scan length.
        noise_level: Noise level for synthetic TS generation.
        ar1: AR(1) coefficient.

    Returns:
        List of result dicts with true_rho, pred_rho, error, is_above_80.
    """
    np.random.seed(seed)
    results: list[dict] = []

    target_min = (t_samples_long * 1.0) / 60.0
    short_min = (t_samples_short * 1.0) / 60.0

    config = BSNetConfig(
        n_rois=n_rois,
        tr=1.0,
        short_duration_sec=short_min * 60,
        target_duration_min=target_min,
        n_bootstraps=50,
    )

    for sub in range(n_subjects):
        # Generate full time series
        long_obs, _ = generate_synthetic_timeseries(
            t_samples_long, n_rois, noise_level=noise_level, ar1=ar1,
        )
        time_series = long_obs.T  # (samples, rois)

        # Reference FC from full scan
        fc_true_T = get_fc_matrix(time_series, vectorized=True, use_shrinkage=True)

        # Mock predicted FC (reference + noise, as in run_real_data_scale.py)
        fc_pred_t_mock = fc_true_T + 0.1 * np.random.randn(*fc_true_T.shape)

        # Short observation
        short_obs = time_series[:t_samples_short, :]

        # Run BS-NET pipeline
        result = run_bootstrap_prediction(short_obs, fc_pred_t_mock, config)

        # true_rho: how close mock FC is to reference FC
        true_rho = float(np.corrcoef(fc_pred_t_mock, fc_true_T)[0, 1])
        pred_rho = float(result.rho_hat_T)

        results.append(
            {
                "seed": seed,
                "subject": f"seed{seed}_sub{sub:03d}",
                "true_rho": true_rho,
                "pred_rho": pred_rho,
                "error": abs(true_rho - pred_rho),
                "is_above_80": pred_rho >= 0.80,
            }
        )

    return results


def main() -> None:
    """Generate and save Figure 2: Prediction validation metrics.

    Runs 10 seeds x 20 subjects synthetic validation, then creates
    four-panel visualization with cross-seed variability.
    """
    print("--- Figure 2: Validation (10-seed synthetic design) ---")
    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for i, seed in enumerate(SEEDS):
        print(f"  Seed {i + 1}/{len(SEEDS)} (seed={seed})...")
        seed_results = run_single_seed(seed, n_subjects=20, n_rois=50)
        all_results.extend(seed_results)

    df = pd.DataFrame(all_results)
    csv_path = reports_dir / "validation_10seed_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(df)} rows)")

    # Cross-seed summary
    total_subjects = len(df)
    mean_true = df["true_rho"].mean()
    mean_pred = df["pred_rho"].mean()
    mean_error = df["error"].mean()

    pass_80 = int(df["is_above_80"].sum())
    pass_80_rate = (pass_80 / total_subjects) * 100
    pass_90 = int((df["pred_rho"] >= 0.90).sum())
    pass_90_rate = (pass_90 / total_subjects) * 100
    failed = total_subjects - pass_80
    failed_rate = (failed / total_subjects) * 100

    # Per-seed pass rates for error bars in Panel D
    seed_stats = df.groupby("seed").agg(
        pass_80_rate=("is_above_80", lambda x: x.mean() * 100),
        pass_90_rate=("pred_rho", lambda x: (x >= 0.90).mean() * 100),
        fail_rate=("is_above_80", lambda x: (1 - x.mean()) * 100),
    ).reset_index()

    t_stat, p_val = stats.ttest_rel(df["true_rho"], df["pred_rho"])
    slope, intercept, r_value, _, _ = stats.linregress(
        df["true_rho"], df["pred_rho"],
    )
    r_squared_val = float(r_value ** 2)

    print("\n--- QUANTITATIVE ANALYSIS (10-seed aggregate) ---")
    print(f"Total Subjects: {total_subjects} ({len(SEEDS)} seeds x 20)")
    print(f"Mean Reference FC: {mean_true:.4f}")
    print(f"Mean Predicted FC: {mean_pred:.4f}")
    print(f"Mean Absolute Error: {mean_error:.4f}")
    print(f"Pass Rate (>= 0.80): {pass_80_rate:.1f}%")
    print(f"Excellent (>= 0.90): {pass_90_rate:.1f}%")
    print(f"Failed (< 0.80): {failed_rate:.1f}%")
    print(f"Paired T-test: t={t_stat:.4f}, p={p_val:.4e}")
    print(f"R^2: {r_squared_val:.4f}")

    # --- Plotting ---
    apply_bsnet_theme()
    fig = plt.figure(figsize=(16, 12))

    # Panel A: Scatter Plot (all seeds pooled, color by seed)
    ax1 = plt.subplot(2, 2, 1)
    sns.scatterplot(
        x="true_rho", y="pred_rho", data=df, alpha=0.5, ax=ax1, s=40,
        hue="seed", palette="tab10", legend=False,
    )
    xlim = (df["true_rho"].min() - 0.02, df["true_rho"].max() + 0.02)
    ax1.plot(xlim, xlim, "k--", label="Identity (y=x)")
    x_range = np.linspace(xlim[0], xlim[1], 100)
    ax1.plot(
        x_range, intercept + slope * x_range,
        color=PALETTE["highlight"], linewidth=2.0,
        label=f"Fit ($R^2={r_squared_val:.2f}$)",
    )
    ax1.set_title(
        "A. Prediction Accuracy (Reference vs. Predicted)",
        fontsize=14, fontweight="bold",
    )
    ax1.set_xlabel("Reference FC Correlation (ρ_true)", fontsize=12)
    ax1.set_ylabel("Predicted FC Correlation (ρ_pred)", fontsize=12)
    ax1.set_xlim(*xlim)
    ax1.legend(fontsize=10)

    # Panel B: Distribution Overlay
    ax2 = plt.subplot(2, 2, 2)
    sns.kdeplot(
        df["true_rho"], fill=True, label="Reference FC",
        ax=ax2, color=PALETTE["true"], alpha=0.5,
    )
    sns.kdeplot(
        df["pred_rho"], fill=True, label="Predicted FC",
        ax=ax2, color=PALETTE["raw"], alpha=0.5,
    )
    bias = mean_true - mean_pred
    df["pred_rho_shifted"] = df["pred_rho"] + bias
    sns.kdeplot(
        df["pred_rho_shifted"], fill=False, label="Predicted (Mean-Shifted)",
        ax=ax2, color=PALETTE["accent"], linestyle="--", linewidth=2.5,
    )
    ax2.axvline(
        x=0.8, color=PALETTE["highlight"], linestyle=":",
        label="Clinical Threshold (0.8)",
    )
    ax2.set_title(
        "B. Density Distribution of FC Correlation",
        fontsize=14, fontweight="bold",
    )
    ax2.set_xlabel("Pearson Correlation (ρ)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.legend(loc="upper left", fontsize=10)

    # Panel C: Error Histogram
    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(df["error"], bins=25, kde=True, ax=ax3, color=PALETTE["bsnet"])
    ax3.set_title(
        "C. Prediction Error Distribution",
        fontsize=14, fontweight="bold",
    )
    ax3.set_xlabel("Absolute Error (|ρ_true - ρ_pred|)", fontsize=12)
    ax3.set_ylabel("Subject Count", fontsize=12)

    # Panel D: Pass Rate Bar Chart with cross-seed error bars
    ax4 = plt.subplot(2, 2, 4)
    categories = ["Failed (< 0.8)", "Good (0.8–0.9)", "Excellent (>= 0.9)"]

    # Per-seed category counts (as rates)
    good_rate = seed_stats["pass_80_rate"] - seed_stats["pass_90_rate"]
    means = [
        seed_stats["fail_rate"].mean(),
        good_rate.mean(),
        seed_stats["pass_90_rate"].mean(),
    ]
    sds = [
        seed_stats["fail_rate"].std(),
        good_rate.std(),
        seed_stats["pass_90_rate"].std(),
    ]
    colors = [
        PALETTE["pass_fail"],
        PALETTE["pass_good"],
        PALETTE["pass_excellent"],
    ]
    bars = ax4.bar(
        categories, means, yerr=sds, color=colors,
        edgecolor="black", alpha=0.8, capsize=5,
        error_kw={"linewidth": 1.2},
    )
    for bar, m, s in zip(bars, means, sds):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 1,
            f"{m:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax4.set_title(
        "D. Clinical Reliability Pass Rate (mean ± SD across seeds)",
        fontsize=14, fontweight="bold",
    )
    ax4.set_ylabel("Rate (%)", fontsize=12)
    ax4.set_ylim(0, max(means) + max(sds) + 15)

    plt.tight_layout(pad=3.0)
    save_figure(fig, "Figure2_Validation.png")
    print("\nFigure 2 saved.")


if __name__ == "__main__":
    main()
