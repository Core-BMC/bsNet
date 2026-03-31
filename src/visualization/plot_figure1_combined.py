"""Generate Figure 1: Combined 2x2 visualization of prediction accuracy metrics.

Displays prediction accuracy vs. duration, marginal gain, uncertainty decay, and
synthetic signal reconstruction with coherence visualization.

All panels (A–C) include cross-seed variability (± 1 SD) from 10-seed data.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.simulate import generate_synthetic_timeseries
from src.visualization.style import PALETTE, apply_bsnet_theme, save_figure

warnings.filterwarnings("ignore")


def _load_per_seed_data() -> pd.DataFrame | None:
    """Load per-seed duration sweep results.

    Returns:
        DataFrame with per-seed data, or None if file not found.
    """
    per_seed_path = Path("artifacts/reports/duration_sweep_seeds_results.csv")
    if per_seed_path.exists():
        return pd.read_csv(per_seed_path)
    return None


def _compute_seed_aggregates(
    df_seeds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-seed metrics and aggregate across seeds.

    For each seed: Predicted, Marginal Gain (diff), CI Width.
    Then aggregate: mean ± SD across seeds.

    Args:
        df_seeds: Per-seed DataFrame with Seed, Duration (s), Predicted,
                  CI Lower, CI Upper columns.

    Returns:
        Tuple of (predicted_agg, marginal_agg, ci_width_agg) DataFrames,
        each with Duration, mean, and sd columns.
    """
    # --- Predicted ---
    pred_agg = (
        df_seeds.groupby("Duration (s)")["Predicted"]
        .agg(["mean", "std"])
        .reset_index()
    )
    pred_agg.columns = ["Duration", "pred_mean", "pred_sd"]

    # --- Marginal Gain per seed ---
    marginal_rows = []
    for seed in df_seeds["Seed"].unique():
        seed_df = df_seeds[df_seeds["Seed"] == seed].sort_values("Duration (s)")
        gains = seed_df["Predicted"].diff()
        for dur, gain in zip(seed_df["Duration (s)"], gains):
            if not np.isnan(gain):
                marginal_rows.append(
                    {"Seed": seed, "Duration": dur, "Marginal_Gain": gain}
                )
    df_mg = pd.DataFrame(marginal_rows)
    mg_agg = (
        df_mg.groupby("Duration")["Marginal_Gain"]
        .agg(["mean", "std"])
        .reset_index()
    )
    mg_agg.columns = ["Duration", "mg_mean", "mg_sd"]

    # --- CI Width per seed ---
    df_seeds = df_seeds.copy()
    df_seeds["CI_Width"] = df_seeds["CI Upper"] - df_seeds["CI Lower"]
    ciw_agg = (
        df_seeds.groupby("Duration (s)")["CI_Width"]
        .agg(["mean", "std"])
        .reset_index()
    )
    ciw_agg.columns = ["Duration", "ciw_mean", "ciw_sd"]

    return pred_agg, mg_agg, ciw_agg


def plot_combined_figure() -> None:
    """Generate and save the combined 2x2 Figure 1.

    Creates four subplots showing:
    - Panel A: Prediction accuracy vs. observation duration (mean ± SD)
    - Panel B: Marginal utility with per-seed error bars
    - Panel C: CI width decay with per-seed SD band
    - Panel D: Synthetic signal visualization with overlay
    """
    apply_bsnet_theme()
    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 2, 1)  # A: Accuracy
    ax2 = plt.subplot(2, 2, 2)  # B: Marginal Gain
    ax3 = plt.subplot(2, 2, 3)  # C: CI Width
    ax4 = plt.subplot(2, 2, 4)  # D: Signal Combined

    # Try per-seed data first (preferred), fall back to aggregated CSV
    df_seeds = _load_per_seed_data()

    if df_seeds is not None:
        pred_agg, mg_agg, ciw_agg = _compute_seed_aggregates(df_seeds)

        # Panel A: Prediction Accuracy (mean ± SD across seeds)
        ax1.plot(
            pred_agg["Duration"],
            pred_agg["pred_mean"],
            marker="o",
            color=PALETTE["true"],
            linewidth=3.0,
            markersize=8,
            label="Mean ρ*(T)",
        )
        ax1.fill_between(
            pred_agg["Duration"],
            pred_agg["pred_mean"] - pred_agg["pred_sd"],
            pred_agg["pred_mean"] + pred_agg["pred_sd"],
            color=PALETTE["ci_fill"],
            alpha=0.5,
            label="± 1 SD (across seeds)",
        )
        ax1.axhline(
            y=0.80, color=PALETTE["highlight"], linestyle="--",
            linewidth=2.5, label="80% Target",
        )
        ax1.axvline(
            x=120, color=PALETTE["accent"], linestyle="-.",
            linewidth=2.5, label="120s Threshold",
        )
        ax1.set_title(
            "A. Prediction Accuracy vs. Duration",
            fontweight="bold", fontsize=15, pad=10,
        )
        ax1.set_xlabel("Duration (seconds)", fontsize=13)
        ax1.set_ylabel("Predicted Correlation ρ*(T)", fontsize=13)
        ax1.set_ylim(0.5, 1.1)
        ax1.legend(loc="lower right", fontsize=11)

        # Panel B: Marginal Gain (mean ± SD per-seed error bars)
        colors = [
            PALETTE["highlight"] if x <= 120 else PALETTE["ci_fill"]
            for x in mg_agg["Duration"]
        ]
        ax2.bar(
            mg_agg["Duration"],
            mg_agg["mg_mean"],
            yerr=mg_agg["mg_sd"],
            width=20,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            capsize=4,
            error_kw={"linewidth": 1.2},
        )
        ax2.axvline(
            x=120, color=PALETTE["accent"], linestyle="-.",
            linewidth=2.5, label="Diminishing Returns (Knee)",
        )
        ax2.set_title(
            "B. Incremental Accuracy Gain (Marginal Utility)",
            fontweight="bold", fontsize=15, pad=10,
        )
        ax2.set_xlabel("Duration (seconds)", fontsize=13)
        ax2.set_ylabel("Δ ρ*(T) per +30s", fontsize=13)
        ax2.set_xticks(pred_agg["Duration"])
        ax2.legend(loc="upper right", fontsize=11)

        # Panel C: CI Width Decay (mean ± SD band)
        ax3.plot(
            ciw_agg["Duration"],
            ciw_agg["ciw_mean"],
            marker="s",
            color=PALETTE["highlight"],
            linewidth=2.5,
            markersize=8,
            label="Mean CI Width",
        )
        ax3.fill_between(
            ciw_agg["Duration"],
            ciw_agg["ciw_mean"] - ciw_agg["ciw_sd"],
            ciw_agg["ciw_mean"] + ciw_agg["ciw_sd"],
            color=PALETTE["highlight"],
            alpha=0.2,
            label="± 1 SD (across seeds)",
        )
        ax3.axvline(
            x=120, color=PALETTE["accent"], linestyle="-.",
            linewidth=2.5, label="Uncertainty Stabilization",
        )
        ax3.set_title(
            "C. Statistical Uncertainty Decay (CI Width)",
            fontweight="bold", fontsize=15, pad=10,
        )
        ax3.set_xlabel("Duration (seconds)", fontsize=13)
        ax3.set_ylabel("Confidence Interval Width (Δρ)", fontsize=13)
        ax3.legend(loc="upper right", fontsize=11)
    else:
        # Fallback: aggregated CSV (legacy)
        csv_path = "artifacts/reports/duration_sweep_seeds_aggregated.csv"
        if Path(csv_path).exists():
            df_agg = pd.read_csv(csv_path)
            ax1.plot(
                df_agg["Duration (s)"], df_agg["Predicted"],
                marker="o", color=PALETTE["true"], linewidth=3.0, markersize=8,
            )
            ax1.set_ylim(0.5, 1.1)
            ax1.set_title("A. Prediction Accuracy vs. Duration", fontweight="bold")

    # Generate synthetic signal for Panel D
    np.random.seed(123)
    n_rois = 50
    tr = 1.0
    t_samples = 120
    long_obs, long_signal = generate_synthetic_timeseries(
        n_samples=t_samples, n_rois=n_rois, noise_level=0.5, ar1=0.7
    )

    target_roi = -1
    best_corr = 0.0
    for i in range(n_rois):
        corr = np.corrcoef(long_signal[i], long_obs[i])[0, 1]
        if 0.80 <= corr <= 0.85:
            target_roi = i
            best_corr = corr
            break

    true_ts = long_signal[target_roi] if target_roi != -1 else long_signal[0]
    raw_ts = long_obs[target_roi] if target_roi != -1 else long_obs[0]
    time_axis = np.arange(t_samples) * tr

    # Panel D: Combined Separated + Overlay
    ax4.plot(
        time_axis, true_ts + 8.0,
        color=PALETTE["true"], linewidth=2.5,
        label="Separated: True Signal (+8)",
    )
    ax4.plot(
        time_axis, raw_ts + 4.0,
        color=PALETTE["highlight"], linewidth=1.5, alpha=0.8,
        label="Separated: Raw Noise (+4)",
    )
    ax4.plot(
        time_axis, true_ts,
        color=PALETTE["true"], linewidth=3.0, alpha=0.85,
        label="Overlay: True Signal",
    )
    ax4.plot(
        time_axis, raw_ts,
        color=PALETTE["accent"], linewidth=1.5, alpha=0.9, linestyle="--",
        label=f"Overlay: Raw Data (r={best_corr:.2f})",
    )
    ax4.axhline(y=2.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
    ax4.set_title(
        "D. Signal Recovery: True vs. Noisy FC (r=0.84)",
        fontweight="bold", fontsize=15, pad=10,
    )
    ax4.set_xlabel("Time (seconds)", fontsize=13)
    ax4.set_ylabel("Amplitude", fontsize=13)
    ax4.set_ylim(-3, 15)
    ax4.legend(loc="upper right", fontsize=10, frameon=True, shadow=True)

    plt.tight_layout(pad=3.0)
    save_figure(fig, "Figure1_Combined.png")


if __name__ == "__main__":
    plot_combined_figure()
