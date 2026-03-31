"""Generate Figure 5: ABIDE Empirical Validation (redesigned).

Four-panel layout matching Fig 3/4/7 style:
  A. Scatter: r_FC (raw) vs ρ̂T (BS-NET) per subject — identity line reference
  B. Violin: Raw FC vs BS-NET ρ̂T distribution comparison
  C. Violin: Improvement (Δ = ρ̂T − r_FC) distribution
  D. Violin: Seed σ (cross-seed stability)

Style: no outline violin, alpha=0.65, dark jitter dots, red diamond mean±SD
Color: Amber (Raw) + Blue (BS-NET) — matching Fig 4/7 schema

Data: data/abide/results/abide_multiseed_{atlas}_10seeds.csv
Output: Figure5_ABIDE_Validation.png
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

RESULTS_DIR = Path("data/abide/results")

# Fig 4/7 color schema
COLOR_RAW = "#fdae61"    # amber
COLOR_BSNET = "#4A90E2"  # blue
COLOR_IMPROVE = "#E891B2" # pink (Panel C)
COLOR_SEED = "#7BC8A4"    # green (Panel D)
DOT_COLOR = "#333333"
DOT_SIZE = 3.6
DOT_ALPHA = 0.35
JITTER_X = 0.08
JITTER_Y = 0.005
VIOLIN_ALPHA = 0.65
MAX_DOTS = 400


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
    """Add dark jittered scatter dots with optional subsampling."""
    n = len(vals)
    if n == 0:
        return
    if n > MAX_DOTS:
        idx = rng.choice(n, size=MAX_DOTS, replace=False)
        vals = vals[idx]
        n = MAX_DOTS
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
    """Load ABIDE multi-seed CSV."""
    csv_path = RESULTS_DIR / f"abide_multiseed_{atlas}_10seeds.csv"
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))
    return {
        "r_fc": np.array([float(r["r_fc_raw"]) for r in rows]),
        "rho_mean": np.array([float(r["rho_hat_T_mean"]) for r in rows]),
        "rho_std": np.array([float(r["rho_hat_T_std"]) for r in rows]),
        "n": len(rows),
    }


def plot_figure5(data: dict, atlas: str) -> plt.Figure:
    """Create Figure 5: ABIDE Validation (4-panel).

    Args:
        data: Dict from load_multiseed().
        atlas: Atlas name (cc200, cc400).

    Returns:
        Matplotlib Figure.
    """
    apply_bsnet_theme()

    r_fc = data["r_fc"]
    rho = data["rho_mean"]
    rho_std = data["rho_std"]
    improvement = rho - r_fc
    n = data["n"]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, figsize=(20, 6),
        gridspec_kw={"width_ratios": [1.2, 0.7, 0.5, 0.5], "wspace": 0.12},
        constrained_layout=True,
    )
    rng = np.random.RandomState(42)

    # ── Panel A: Scatter r_FC vs ρ̂T ──
    ax1.scatter(
        r_fc, rho, s=12, c=COLOR_BSNET, alpha=0.5,
        edgecolors="white", linewidth=0.3, zorder=3,
    )
    # Identity line
    lims = [min(r_fc.min(), rho.min()) - 0.05, max(r_fc.max(), rho.max()) + 0.05]
    ax1.plot(lims, lims, "k--", alpha=0.3, linewidth=LINE["thin"], label="identity")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_title(
        f"A. Raw FC vs BS-NET ({atlas.upper()}, N={n})",
        fontweight="bold", fontsize=FONT["title"],
    )
    ax1.set_xlabel(r"$r_{FC}$ (raw, 2 min vs full)", fontsize=FONT["axis_label"])
    ax1.set_ylabel(r"$\hat{\rho}_T$ (mean ± 1 σ across seeds)", fontsize=FONT["axis_label"])
    ax1.legend(loc="lower right", fontsize=FONT["legend_small"])

    # ── Panel B: Violin Raw vs BS-NET ──
    df_b = pd.DataFrame({
        "value": np.concatenate([r_fc, rho]),
        "condition": ["Raw FC (2m)"] * n + ["BS-NET (2m)"] * n,
    })
    order_b = ["Raw FC (2m)", "BS-NET (2m)"]
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

    n_improved = np.sum(improvement > 0)
    pct = n_improved / n * 100
    ax2.set_title(
        "B. Distribution Comparison",
        fontweight="bold", fontsize=FONT["title"],
    )
    ax2.set_ylabel(r"FC similarity", fontsize=FONT["axis_label"])
    ax2.set_xlabel("")
    # Annotation
    ax2.text(
        0.5, 0.02,
        f"{pct:.1f}% improved ({n_improved}/{n})",
        transform=ax2.transAxes, ha="center", fontsize=FONT["legend_small"],
        fontstyle="italic", color="#555555",
    )

    # ── Panel C: Δ Improvement ──
    df_imp = pd.DataFrame({"value": improvement, "metric": "Δ (ρ̂T − r_FC)"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.violinplot(
            data=df_imp, x="metric", y="value",
            palette=[COLOR_IMPROVE], inner="box", linewidth=0, width=0.6,
            cut=2, ax=ax3,
        )
    _set_violin_alpha(ax3)
    _jitter_scatter(ax3, improvement, 0, rng, y_sig=0.003)
    _diamond(ax3, improvement, 0)
    ax3.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax3.set_title(
        "C. Improvement (Δ)",
        fontweight="bold", fontsize=FONT["title"],
    )
    ax3.set_ylabel(r"Δ ($\hat{\rho}_T - r_{FC}$)", fontsize=FONT["axis_label"])
    ax3.set_xlabel("")

    # ── Panel D: Seed σ ──
    df_seed = pd.DataFrame({"value": rho_std, "metric": "Seed σ"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.violinplot(
            data=df_seed, x="metric", y="value",
            palette=[COLOR_SEED], inner="box", linewidth=0, width=0.6,
            cut=2, ax=ax4,
        )
    _set_violin_alpha(ax4)
    _jitter_scatter(ax4, rho_std, 0, rng, y_sig=0.0003)
    _diamond(ax4, rho_std, 0)
    ax4.set_title(
        "D. Seed Stability",
        fontweight="bold", fontsize=FONT["title"],
    )
    ax4.set_ylabel("Seed σ (cross-seed SD)", fontsize=FONT["axis_label"])
    ax4.set_xlabel("")

    return fig


def main() -> None:
    """Run Figure 5 generation."""
    parser = argparse.ArgumentParser(description="Figure 5: ABIDE Validation")
    parser.add_argument("--atlas", default="cc200", choices=["cc200", "cc400"])
    args = parser.parse_args()

    data = load_multiseed(args.atlas)
    r_fc = data["r_fc"]
    rho = data["rho_mean"]
    improvement = rho - r_fc

    print(f"ABIDE {args.atlas.upper()} — N={data['n']}")
    print(f"  r_FC:  {r_fc.mean():.3f} ± {r_fc.std():.3f}")
    print(f"  ρ̂T:   {rho.mean():.3f} ± {rho.std():.3f}")
    print(f"  Δ:     {improvement.mean():+.3f} ± {improvement.std():.3f}")
    print(f"  Improved: {np.sum(improvement > 0)}/{data['n']} ({np.mean(improvement > 0)*100:.1f}%)")
    print(f"  Seed σ: {data['rho_std'].mean():.4f} (mean)")

    fig = plot_figure5(data, args.atlas)
    save_figure(fig, "Figure5_ABIDE_Validation.png")
    print("\nFigure 5 saved: Figure5_ABIDE_Validation.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
