"""Generate Figure 6: Cross-Dataset Generalization — ADHD-200 (redesigned).

Three-panel layout matching Fig 3/4/7 style:
  A. Scatter: r_FC vs ρ̂T per subject (ADHD=red, Control=blue markers)
  B. Violin: Raw FC vs BS-NET ρ̂T distribution comparison
  C. Violin: BS-NET ρ̂T by group (ADHD vs Control)

Style: no outline violin, alpha=0.65, dark jitter dots, red diamond mean±SD
Color: Amber (Raw) + Blue (BS-NET), Red (ADHD) + Blue (Control)

Data: data/adhd/results/adhd_multiseed_{atlas}_10seeds.csv
Output: Figure6_ADHD_Validation.png
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualization.style import (
    FONT,
    LINE,
    apply_bsnet_theme,
    save_figure,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/adhd/results")

# Fig 4/7 color schema
COLOR_RAW = "#fdae61"
COLOR_BSNET = "#4A90E2"
COLOR_ADHD = "#d7191c"
COLOR_CONTROL = "#2c7bb6"
DOT_COLOR = "#333333"
DOT_SIZE = 8.0
DOT_ALPHA = 0.40
JITTER_X = 0.06
JITTER_Y = 0.005
VIOLIN_ALPHA = 0.65


def _set_violin_alpha(ax: plt.Axes, alpha: float = VIOLIN_ALPHA) -> None:
    """Set uniform alpha on all violin body PolyCollections."""
    from matplotlib.collections import PolyCollection

    for art in ax.collections:
        if isinstance(art, PolyCollection):
            art.set_alpha(alpha)


def _jitter_scatter(
    ax: plt.Axes, vals: np.ndarray, x_pos: float,
    rng: np.random.RandomState, x_sig: float = JITTER_X,
    y_sig: float = JITTER_Y,
) -> None:
    """Add dark jittered scatter dots."""
    n = len(vals)
    if n == 0:
        return
    ax.scatter(
        x_pos + rng.normal(0, x_sig, n),
        vals + rng.normal(0, y_sig, n),
        s=DOT_SIZE, c=DOT_COLOR, alpha=DOT_ALPHA,
        edgecolors="none", zorder=4,
    )


def _diamond(ax: plt.Axes, vals: np.ndarray, x_pos: float) -> None:
    """Red diamond mean±SD marker."""
    ax.errorbar(
        x_pos, np.nanmean(vals), yerr=np.nanstd(vals),
        fmt="D", color="red", markersize=6,
        markeredgecolor="darkred", markeredgewidth=1.0,
        ecolor="darkred", elinewidth=1.2, capsize=3, capthick=1.2,
        zorder=15,
    )


def load_multiseed(atlas: str) -> dict:
    """Load ADHD multi-seed CSV."""
    csv_path = RESULTS_DIR / f"adhd_multiseed_{atlas}_10seeds.csv"
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))
    return {
        "r_fc": np.array([float(r["r_fc_raw"]) for r in rows]),
        "rho_mean": np.array([float(r["rho_mean"]) for r in rows]),
        "rho_std": np.array([float(r["rho_std"]) for r in rows]),
        "group": np.array([r["group"].lower().strip() for r in rows]),
        "n": len(rows),
    }


def plot_figure6(data: dict, atlas: str) -> plt.Figure:
    """Create Figure 6: ADHD Cross-Dataset Validation (3-panel).

    Args:
        data: Dict from load_multiseed().
        atlas: Atlas name.

    Returns:
        Matplotlib Figure.
    """
    apply_bsnet_theme()

    r_fc = data["r_fc"]
    rho = data["rho_mean"]
    group = data["group"]
    n = data["n"]
    improvement = rho - r_fc

    is_adhd = group == "adhd"
    is_ctrl = ~is_adhd

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, 6),
        gridspec_kw={"width_ratios": [1, 0.8, 0.8], "wspace": 0.30},
    )
    rng = np.random.RandomState(42)

    # ── Panel A: Scatter r_FC vs ρ̂T (group-colored) ──
    ax1.scatter(
        r_fc[is_ctrl], rho[is_ctrl], s=30, c=COLOR_CONTROL, alpha=0.7,
        edgecolors="white", linewidth=0.5, zorder=3, label="Control",
        marker="o",
    )
    ax1.scatter(
        r_fc[is_adhd], rho[is_adhd], s=30, c=COLOR_ADHD, alpha=0.7,
        edgecolors="white", linewidth=0.5, zorder=3, label="ADHD",
        marker="^",
    )
    lims = [
        min(r_fc.min(), rho.min()) - 0.05,
        max(r_fc.max(), rho.max()) + 0.05,
    ]
    ax1.plot(lims, lims, "k--", alpha=0.3, linewidth=LINE["thin"], label="identity")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_title(
        f"A. Raw FC vs BS-NET ({atlas.upper()}, N={n})",
        fontweight="bold", fontsize=FONT["title"],
    )
    ax1.set_xlabel(r"$r_{FC}$ (raw)", fontsize=FONT["axis_label"])
    ax1.set_ylabel(r"$\hat{\rho}_T$ (mean across seeds)", fontsize=FONT["axis_label"])
    ax1.legend(loc="lower right", fontsize=FONT["legend_small"])

    # ── Panel B: Violin Raw vs BS-NET ──
    df_b = pd.DataFrame({
        "value": np.concatenate([r_fc, rho]),
        "condition": ["Raw FC"] * n + ["BS-NET"] * n,
    })
    order_b = ["Raw FC", "BS-NET"]
    palette_b = [COLOR_RAW, COLOR_BSNET]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.violinplot(
            data=df_b, x="condition", y="value", order=order_b,
            palette=palette_b, inner="box", linewidth=0, width=0.75,
            cut=2, ax=ax2,
        )
    _set_violin_alpha(ax2)

    for i, cond in enumerate(order_b):
        vals = df_b.loc[df_b["condition"] == cond, "value"].values
        _jitter_scatter(ax2, vals, i, rng)
        _diamond(ax2, vals, i)

    n_imp = np.sum(improvement > 0)
    ax2.set_title(
        "B. Distribution Comparison",
        fontweight="bold", fontsize=FONT["title"],
    )
    ax2.set_ylabel("FC similarity", fontsize=FONT["axis_label"])
    ax2.set_xlabel("")
    ax2.text(
        0.5, 0.02,
        f"{n_imp/n*100:.1f}% improved ({n_imp}/{n})",
        transform=ax2.transAxes, ha="center", fontsize=FONT["legend_small"],
        fontstyle="italic", color="#555555",
    )

    # ── Panel C: Violin BS-NET ρ̂T by group ──
    df_c = pd.DataFrame({
        "rho": rho,
        "Group": np.where(is_adhd, "ADHD", "Control"),
    })
    order_c = ["ADHD", "Control"]
    palette_c = [COLOR_ADHD, COLOR_CONTROL]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.violinplot(
            data=df_c, x="Group", y="rho", order=order_c,
            palette=palette_c, inner="box", linewidth=0, width=0.75,
            cut=2, ax=ax3,
        )
    _set_violin_alpha(ax3, alpha=0.45)

    for i, grp in enumerate(order_c):
        vals = df_c.loc[df_c["Group"] == grp, "rho"].values
        _jitter_scatter(ax3, vals, i, rng, x_sig=0.08, y_sig=0.008)
        _diamond(ax3, vals, i)

    ax3.set_title(
        "C. BS-NET ρ̂T by Group",
        fontweight="bold", fontsize=FONT["title"],
    )
    ax3.set_ylabel(r"$\hat{\rho}_T$ (BS-NET)", fontsize=FONT["axis_label"])
    ax3.set_xlabel("")

    plt.tight_layout(pad=2.0)
    return fig


def main() -> None:
    """Run Figure 6 generation."""
    parser = argparse.ArgumentParser(description="Figure 6: ADHD Validation")
    parser.add_argument("--atlas", default="cc200", choices=["cc200", "cc400"])
    args = parser.parse_args()

    data = load_multiseed(args.atlas)
    r_fc = data["r_fc"]
    rho = data["rho_mean"]
    group = data["group"]
    improvement = rho - r_fc

    print(f"ADHD {args.atlas.upper()} — N={data['n']}")
    print(f"  r_FC:  {r_fc.mean():.3f} ± {r_fc.std():.3f}")
    print(f"  ρ̂T:   {rho.mean():.3f} ± {rho.std():.3f}")
    print(f"  Δ:     {improvement.mean():+.3f} ± {improvement.std():.3f}")
    for g in ["adhd", "control"]:
        mask = group == g
        print(f"  {g.upper()}: ρ̂T={rho[mask].mean():.3f}±{rho[mask].std():.3f}")

    fig = plot_figure6(data, args.atlas)
    save_figure(fig, "Figure6_ADHD_Validation.png")
    print("\nFigure 6 saved: Figure6_ADHD_Validation.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
