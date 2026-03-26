"""Format Figure 4: Continuous-axis visualization for 100-ROI baseline model.

Creates continuous y-axis plots for Jaccard overlap and ARI metrics on
the smaller 100-ROI Yeo parcellation model.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.style import MODEL_PALETTE, apply_bsnet_theme, save_figure

warnings.filterwarnings("ignore")


def main() -> None:
    """Generate and save continuous-axis Figure 4 for 100-ROI model.

    Creates two visualizations with continuous y-axis:
    - Panel A: Per-network Jaccard overlap across models
    - Panel B: Global modularity continuity (ARI)

    Returns:
        None
    """
    print("--- Formatting Continuous Axis Figure 4 (Yeo 100) ---")
    out_dir = Path("artifacts/reports")
    figure_dir = Path("docs/figure")
    figure_dir.mkdir(parents=True, exist_ok=True)

    # SUBNETWORKS (Continuous Y-axis)
    df_net = pd.read_csv(out_dir / "per_network_jaccard_results_yeo100.csv")

    apply_bsnet_theme()
    fig = plt.figure(figsize=(14, 7))

    sns.violinplot(
        data=df_net,
        x="Network",
        y="Jaccard Overlap",
        hue="Model",
        palette=MODEL_PALETTE,
        split=False,
        inner="box",
        linewidth=1,
        width=0.8,
    )
    sns.swarmplot(
        data=df_net,
        x="Network",
        y="Jaccard Overlap",
        hue="Model",
        palette=["#9C1C3D", "#0A4D8D"],
        dodge=True,
        alpha=0.8,
        size=4.5,
        edgecolor="white",
        linewidth=0.5,
    )

    plt.ylim(0.40, 1.1)

    # Legend mapping inside empty space (lower right)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], loc="lower right", title="Model")

    plt.xlabel("Canonical Network")
    plt.ylabel(
        "Jaccard Similarity w/ True 15m Network",
        fontweight="bold",
    )

    plt.title(
        "Continuous Jaccard Curve (Baseline 100-ROI Model)",
        fontweight="bold",
        pad=15,
    )
    plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.3)

    plt.xticks(rotation=15, ha="right")

    save_figure(fig, "Figure4_Subnetworks_Yeo100.png")
    print("\nExtracted and formatted Continuous Jaccard Plot.")

    # OVERALL ARI
    print("\n--- Formatting Overall ARI Figure 4 (Yeo 100) ---")
    df_ari = pd.read_csv(out_dir / "subnetwork_ari_results_yeo100.csv")
    fig2 = plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=df_ari,
        x="Model",
        y="ARI",
        hue="Model",
        palette=MODEL_PALETTE,
        split=False,
        inner="box",
        linewidth=1,
        width=0.7,
    )
    sns.swarmplot(
        data=df_ari,
        x="Model",
        y="ARI",
        hue="Model",
        palette=["#9C1C3D", "#0A4D8D"],
        dodge=False,
        alpha=0.8,
        size=5,
        edgecolor="white",
        linewidth=0.5,
        legend=False,
    )

    plt.ylim(0.40, 1.1)

    plt.title(
        "Global Modularity Continuity (Baseline 100-ROI Model)",
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("Adjusted Rand Index (ARI)", fontweight="bold")
    plt.xlabel("")

    save_figure(fig2, "Figure4_Overall_ARI_Yeo100.png")
    print("Extracted and formatted Extended ARI Plot.")


if __name__ == "__main__":
    main()
