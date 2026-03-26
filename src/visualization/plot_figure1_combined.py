"""Generate Figure 1: Combined 2x2 visualization of prediction accuracy metrics.

Displays prediction accuracy vs. duration, marginal gain, uncertainty decay, and
synthetic signal reconstruction with coherence visualization.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.simulate import generate_synthetic_timeseries
from src.visualization.style import PALETTE, apply_bsnet_theme, save_figure

warnings.filterwarnings("ignore")


def plot_combined_figure() -> None:
    """Generate and save the combined 2x2 Figure 1.

    Creates four subplots showing:
    - Panel A: Prediction accuracy vs. observation duration
    - Panel B: Marginal utility (incremental accuracy gain)
    - Panel C: Confidence interval width (uncertainty decay)
    - Panel D: Synthetic signal visualization with overlay

    Returns:
        None
    """
    apply_bsnet_theme()
    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 2, 1)  # A: Accuracy
    ax2 = plt.subplot(2, 2, 2)  # B: Marginal Gain
    ax3 = plt.subplot(2, 2, 3)  # C: CI Width
    ax4 = plt.subplot(2, 2, 4)  # D: Signal Combined

    csv_path = "artifacts/reports/duration_sweep_seeds_aggregated.csv"
    if Path(csv_path).exists():
        df_agg = pd.read_csv(csv_path)

        # Panel A: Prediction Accuracy
        ax1.plot(
            df_agg["Duration (s)"],
            df_agg["Predicted"],
            marker="o",
            color=PALETTE["true"],
            linewidth=3.0,
            markersize=8,
            label="Mean ρ*(T)",
        )
        ax1.fill_between(
            df_agg["Duration (s)"],
            df_agg["CI Lower"],
            df_agg["CI Upper"],
            color=PALETTE["ci_fill"],
            alpha=0.5,
            label="95% CI",
        )
        ax1.axhline(
            y=0.80,
            color=PALETTE["highlight"],
            linestyle="--",
            linewidth=2.5,
            label="80% Target",
        )
        ax1.axvline(
            x=120,
            color=PALETTE["accent"],
            linestyle="-.",
            linewidth=2.5,
            label="120s Threshold",
        )
        ax1.set_title(
            "A. Prediction Accuracy vs. Duration",
            fontweight="bold",
            fontsize=15,
            pad=10,
        )
        ax1.set_xlabel("Duration (seconds)", fontsize=13)
        ax1.set_ylabel("Predicted Correlation ρ*(T)", fontsize=13)
        ax1.set_ylim(0.4, 1.0)
        ax1.legend(loc="lower right", fontsize=11)

        # Panel B: Marginal Gain
        df_agg["Marginal_Gain"] = df_agg["Predicted"].diff()
        diff_data = df_agg.dropna(subset=["Marginal_Gain"])

        colors = [
            PALETTE["highlight"] if x <= 120 else PALETTE["ci_fill"]
            for x in diff_data["Duration (s)"]
        ]
        ax2.bar(
            diff_data["Duration (s)"],
            diff_data["Marginal_Gain"],
            width=20,
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax2.axvline(
            x=120,
            color=PALETTE["accent"],
            linestyle="-.",
            linewidth=2.5,
            label="Diminishing Returns (Knee)",
        )
        ax2.set_title(
            "B. Incremental Accuracy Gain (Marginal Utility)",
            fontweight="bold",
            fontsize=15,
            pad=10,
        )
        ax2.set_xlabel("Duration (seconds)", fontsize=13)
        ax2.set_ylabel("Δ ρ*(T) per +30s", fontsize=13)
        ax2.set_xticks(df_agg["Duration (s)"])
        ax2.legend(loc="upper right", fontsize=11)

        # Panel C: CI Width (Uncertainty Decay)
        df_agg["CI_Width"] = df_agg["CI Upper"] - df_agg["CI Lower"]
        ax3.plot(
            df_agg["Duration (s)"],
            df_agg["CI_Width"],
            marker="s",
            color=PALETTE["highlight"],
            linewidth=2.5,
            markersize=8,
            label="95% CI Boundary Width",
        )
        ax3.axvline(
            x=120,
            color=PALETTE["accent"],
            linestyle="-.",
            linewidth=2.5,
            label="Uncertainty Stabilization",
        )
        ax3.set_title(
            "C. Statistical Uncertainty Decay (CI Width)",
            fontweight="bold",
            fontsize=15,
            pad=10,
        )
        ax3.set_xlabel("Duration (seconds)", fontsize=13)
        ax3.set_ylabel("Confidence Interval Width (Δρ)", fontsize=13)
        ax3.legend(loc="upper right", fontsize=11)

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
        time_axis,
        true_ts + 8.0,
        color=PALETTE["true"],
        linewidth=2.5,
        label="Separated: True Signal (+8)",
    )
    ax4.plot(
        time_axis,
        raw_ts + 4.0,
        color=PALETTE["highlight"],
        linewidth=1.5,
        alpha=0.8,
        label="Separated: Raw Noise (+4)",
    )

    ax4.plot(
        time_axis,
        true_ts,
        color=PALETTE["true"],
        linewidth=3.0,
        alpha=0.85,
        label="Overlay: True Signal",
    )
    ax4.plot(
        time_axis,
        raw_ts,
        color=PALETTE["accent"],
        linewidth=1.5,
        alpha=0.9,
        linestyle="--",
        label=f"Overlay: Raw Data (r={best_corr:.2f})",
    )

    ax4.axhline(y=2.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)

    ax4.set_title(
        "D. Visualizing 84% Coherence (Separated vs. Overlay)",
        fontweight="bold",
        fontsize=15,
        pad=10,
    )
    ax4.set_xlabel("Time (seconds)", fontsize=13)
    ax4.set_ylabel("Amplitude", fontsize=13)
    ax4.set_ylim(-3, 15)

    ax4.legend(loc="upper right", fontsize=10, frameon=True, shadow=True)

    plt.tight_layout(pad=3.0)
    save_figure(fig, "Figure1_Combined.png")


if __name__ == "__main__":
    plot_combined_figure()
