"""Format Figure 4: Broken-axis visualization of network Jaccard overlap.

Creates a broken y-axis plot for better visualization of bimodal Jaccard
distributions with high and low similarity clusters.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.style import MODEL_PALETTE, apply_bsnet_theme, save_figure

warnings.filterwarnings("ignore")


def main() -> None:
    """Generate and save broken-axis Figure 4 visualization.

    Creates a two-panel broken y-axis plot to visualize bimodal Jaccard
    similarity distributions across networks, improving visibility of both
    high-similarity and low-similarity clusters.

    Returns:
        None
    """
    print("--- Formatting Broken Axis Figure 4 ---")
    out_dir = Path("artifacts/reports")
    figure_dir = Path("docs/figure")
    figure_dir.mkdir(parents=True, exist_ok=True)

    df_net = pd.read_csv(out_dir / "per_network_jaccard_results.csv")

    # Create subplots for broken y-axis
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.subplots_adjust(hspace=0.08)

    apply_bsnet_theme()

    # Top plot (>0.6) and Bottom plot (<=0.6)
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
        ax=ax1,
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
        ax=ax1,
    )

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
        ax=ax2,
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
        ax=ax2,
    )

    # Dynamic Y limits based on data clusters
    df_top = df_net[df_net["Jaccard Overlap"] > 0.6]
    df_bot = df_net[df_net["Jaccard Overlap"] <= 0.6]

    top_min = max(0.65, df_top["Jaccard Overlap"].min() - 0.05)
    ax1.set_ylim(top_min, 1.1)

    if not df_bot.empty:
        bot_min = df_bot["Jaccard Overlap"].min() - 0.05
        bot_max = df_bot["Jaccard Overlap"].max() + 0.05
    else:
        bot_min, bot_max = 0.45, 0.55

    ax2.set_ylim(bot_min, bot_max)

    # Hide spines to make it look like a broken axis
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    # Draw diagonal slash marks
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Legend mapping inside empty space
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], labels[:2], loc="upper right", title="Model")
    if ax2.get_legend():
        ax2.get_legend().remove()

    ax1.set_xlabel("")
    ax2.set_xlabel("Canonical Network")
    ax1.set_ylabel("")
    ax2.set_ylabel("")

    fig.text(
        0.06,
        0.5,
        "Jaccard Similarity w/ True 15m Network",
        va="center",
        rotation="vertical",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_title(
        "Smoothed Network Jaccard Curve (Schaefer 400 Parcellation)",
        fontweight="bold",
        pad=15,
    )
    ax1.axhline(y=1.0, color="r", linestyle="--", alpha=0.3)

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha="right")

    save_figure(fig, "Figure4_Subnetworks_BrokenAxis.png")
    print("\nExtracted and formatted broken-axis Jaccard Plot.")

    print("\n--- Formatting Overall ARI Figure 4 ---")
    df_ari = pd.read_csv(out_dir / "subnetwork_ari_results.csv")
    fig2 = plt.figure(figsize=(8, 6))
    sns.violinplot(
        data=df_ari,
        x="Model",
        y="ARI",
        hue="Model",
        palette=MODEL_PALETTE,
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
        alpha=0.8,
        size=5,
        edgecolor="white",
        linewidth=0.5,
        legend=False,
    )
    plt.ylim(0.7, 1.1)
    plt.title(
        "Global Modularity Continuity (Schaefer 400 Parcellation)",
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("Adjusted Rand Index (ARI)")
    plt.xlabel("")

    save_figure(fig2, "Figure4_Overall_ARI_BrokenAxis.png")
    print("Extracted and formatted Overall ARI Plot.")


if __name__ == "__main__":
    main()
