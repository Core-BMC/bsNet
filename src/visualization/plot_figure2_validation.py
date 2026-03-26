"""Generate Figure 2: Validation metrics for BS-NET prediction accuracy.

Creates a 2x2 visualization of prediction accuracy, distribution overlap,
error distribution, and clinical pass rates.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.visualization.style import PALETTE, apply_bsnet_theme, save_figure


def main() -> None:
    """Generate and save Figure 2: Prediction validation metrics.

    Loads the latest scale-up results and generates four-panel visualization
    showing prediction accuracy, distribution overlap, error distribution,
    and clinical reliability pass rates.

    Returns:
        None
    """
    reports_dir = Path("artifacts/reports")
    result_files = list(reports_dir.glob("scale_up_100_results_*.csv"))
    if not result_files:
        print("No results found.")
        return

    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading '{latest_file.name}' for Figure 2 Generation...")

    df = pd.read_csv(latest_file)

    # Quantitative Analysis
    total_subjects = len(df)
    mean_true = df["true_rho"].mean()
    mean_pred = df["pred_rho"].mean()
    mean_error = df["error"].mean()

    pass_80 = df["is_above_80"].sum()
    pass_80_rate = (pass_80 / total_subjects) * 100

    pass_90 = (df["pred_rho"] >= 0.90).sum()
    pass_90_rate = (pass_90 / total_subjects) * 100

    failed = total_subjects - pass_80
    failed_rate = (failed / total_subjects) * 100

    # Paired t-test
    t_stat, p_val = stats.ttest_rel(df["true_rho"], df["pred_rho"])

    # Linear Regression for Panel A
    slope, intercept, r_value, p_val_lin, std_err = stats.linregress(
        df["true_rho"], df["pred_rho"]
    )
    r_squared = r_value**2

    print("\n--- QUANTITATIVE ANALYSIS ---")
    print(f"Total Subjects: {total_subjects}")
    print(f"Mean True FC: {mean_true:.4f}")
    print(f"Mean Predicted FC: {mean_pred:.4f}")
    print(f"Mean Absolute Error: {mean_error:.4f}")
    print(f"Pass Rate (>= 80%): {pass_80_rate:.1f}% ({pass_80}/{total_subjects})")
    print(
        f"Excellent Pass Rate (>= 90%): {pass_90_rate:.1f}% "
        f"({pass_90}/{total_subjects})"
    )
    print(f"Failed Rate (< 80%): {failed_rate:.1f}% ({failed}/{total_subjects})")
    print(f"Paired T-test (True vs Pred): t={t_stat:.4f}, p={p_val:.4e}")

    # Handle numpy version compatibility for r_squared
    try:
        r_squared_val = (
            r_squared[0] if isinstance(r_squared, (list, np.ndarray)) else r_squared
        )
    except (IndexError, TypeError):
        r_squared_val = r_squared
    print(f"Linear Regression R^2: {float(r_squared_val):.4f}")

    # Plotting Figure 2
    fig = plt.figure(figsize=(16, 12))
    apply_bsnet_theme()

    # Panel A: Scatter Plot
    ax1 = plt.subplot(2, 2, 1)
    sns.scatterplot(
        x="true_rho",
        y="pred_rho",
        data=df,
        alpha=0.7,
        ax=ax1,
        s=60,
        color=PALETTE["bsnet"],
    )
    ax1.plot(
        [0.85, 1.00],
        [0.85, 1.00],
        "k--",
        label="Identity (y=x)",
    )
    x_range = np.linspace(0.85, 1.00, 100)
    ax1.plot(
        x_range,
        intercept + slope * x_range,
        color=PALETTE["highlight"],
        linewidth=2.0,
        label=f"Fit ($R^2={float(r_squared_val):.2f}$)",
    )
    ax1.set_title(
        "A. Prediction Accuracy (True vs. Predicted)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Ground Truth FC Correlation (ρ_true)", fontsize=12)
    ax1.set_ylabel("Predicted FC Correlation (ρ_pred)", fontsize=12)
    ax1.set_xlim(0.85, 1.00)
    ax1.set_ylim(0.6, 1.2)
    ax1.legend()

    # Panel B: Distribution Overlay
    ax2 = plt.subplot(2, 2, 2)
    sns.kdeplot(
        df["true_rho"],
        fill=True,
        label="Ground Truth (Raw)",
        ax=ax2,
        color=PALETTE["true"],
        alpha=0.5,
    )
    sns.kdeplot(
        df["pred_rho"],
        fill=True,
        label="Predicted (Raw)",
        ax=ax2,
        color=PALETTE["raw"],
        alpha=0.5,
    )

    # Calculate Mean-Shift Bias
    bias = mean_true - mean_pred
    df["pred_rho_shifted"] = df["pred_rho"] + bias

    # Plot Mean-Shifted Predicted
    sns.kdeplot(
        df["pred_rho_shifted"],
        fill=False,
        label="Predicted (Mean-Shifted)",
        ax=ax2,
        color=PALETTE["accent"],
        linestyle="--",
        linewidth=2.5,
    )

    ax2.set_title(
        "B. Density Distribution of FC Correlation",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Pearson Correlation (ρ)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.axvline(
        x=0.8,
        color=PALETTE["highlight"],
        linestyle=":",
        label="Clinical Threshold (0.8)",
    )
    ax2.legend(loc="upper left", fontsize=10)

    # Panel C: Error Histogram
    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(df["error"], bins=20, kde=True, ax=ax3, color=PALETTE["bsnet"])
    ax3.set_title(
        "C. Prediction Error Distribution",
        fontsize=14,
        fontweight="bold",
    )
    ax3.set_xlabel("Absolute Error (|ρ_true - ρ_pred|)", fontsize=12)
    ax3.set_ylabel("Subject Count", fontsize=12)

    # Panel D: Categorical Pass Rate Bar Chart
    ax4 = plt.subplot(2, 2, 4)
    categories = ["Failed (< 0.8)", "Good (0.8 - 0.9)", "Excellent (≥ 0.9)"]
    counts = [failed, pass_80 - pass_90, pass_90]
    colors = [
        PALETTE["pass_fail"],
        PALETTE["pass_good"],
        PALETTE["pass_excellent"],
    ]
    bars = ax4.bar(categories, counts, color=colors, edgecolor="black", alpha=0.8)
    for bar in bars:
        yval = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f"{yval}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax4.set_title(
        "D. Clinical Reliability Pass Rate",
        fontsize=14,
        fontweight="bold",
    )
    ax4.set_ylabel("Number of Subjects (%)", fontsize=12)
    ax4.set_ylim(0, max(counts) + 15)

    plt.tight_layout(pad=3.0)

    # Save the figure
    save_figure(fig, "Figure2_Validation.png")
    print("\nFigure saved to documentation hub and artifact reports.")


if __name__ == "__main__":
    main()
