"""Figure 3: ABIDE Large-Scale Validation (Main, N=468 default).

Main figure intentionally focuses on validation performance only
(scatter/distribution/improvement/seed stability) without CONSORT.
Quality-filtered CONSORT analysis is moved to Supplementary Figure S3.

Usage:
  python -m src.visualization.plot_figure3_abide
  python -m src.visualization.plot_figure3_abide --filtered-strict
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import (
    ACCENT_COLORS,
    CONDITION_PALETTE,
    FONT,
    LINE,
    apply_bsnet_theme,
    save_figure,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/abide/results")

# Colors
COLOR_RAW = CONDITION_PALETTE["raw"]
COLOR_BSNET = CONDITION_PALETTE["bsnet"]
COLOR_IMPROVE = ACCENT_COLORS["improvement"]
COLOR_SEED = ACCENT_COLORS["seed_sigma"]

# Violin+box style (matching Fig 2 Panel D scheme)
VIOLIN_ALPHA = 0.45
BOX_ALPHA = 0.80
VIOLIN_WIDTH = 0.5
BOX_WIDTH = 0.24
OUTLIER_MARKER_SIZE = 3.0
OUTLIER_ALPHA = 0.35


def load_multiseed(atlas: str, filtered_strict: bool = False) -> dict:
    """Load ABIDE multi-seed CSV.

    Args:
        atlas: Atlas name.
        filtered_strict: If True, load strict-filtered CSV.

    Returns:
        Dict with r_fc, rho_mean, rho_std, n.
    """
    if filtered_strict:
        csv_path = RESULTS_DIR / f"abide_multiseed_{atlas}_10seeds_filtered_strict.csv"
    else:
        csv_path = RESULTS_DIR / f"abide_multiseed_{atlas}_10seeds.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    return {
        "r_fc": np.array([float(r["r_fc_raw"]) for r in rows]),
        "rho_mean": np.array([float(r["rho_hat_T_mean"]) for r in rows]),
        "rho_std": np.array([float(r["rho_hat_T_std"]) for r in rows]),
        "n": len(rows),
    }


def _violin_box_sd(
    ax: plt.Axes,
    data: list[np.ndarray],
    positions: list[float],
    colors: list[str],
) -> None:
    """Draw violin + boxplot + mean±SD + outlier dots."""
    vp = ax.violinplot(
        data,
        positions=positions,
        showmedians=False,
        showextrema=False,
        widths=VIOLIN_WIDTH,
    )
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(VIOLIN_ALPHA)
        body.set_edgecolor("none")

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(
            marker=".",
            markersize=OUTLIER_MARKER_SIZE,
            markerfacecolor="#444444",
            markeredgecolor="none",
            alpha=OUTLIER_ALPHA,
        ),
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(BOX_ALPHA)
        patch.set_edgecolor(colors[i])

def plot_figure3(data: dict, atlas: str) -> plt.Figure:
    """Create Figure 3 main panels (A–D)."""
    apply_bsnet_theme()

    r_fc = data["r_fc"]
    rho = data["rho_mean"]
    rho_std = data["rho_std"]
    improvement = rho - r_fc
    n = data["n"]

    fig, (ax_scatter, ax_comp, ax_imp, ax_seed) = plt.subplots(
        1,
        4,
        figsize=(20, 6),
        gridspec_kw={"width_ratios": [1.2, 0.7, 0.5, 0.5], "wspace": 0.18},
        constrained_layout=True,
    )

    # Panel A: Scatter r_FC vs ρ̂T
    ax_scatter.scatter(
        r_fc,
        rho,
        s=18,
        c=COLOR_BSNET,
        alpha=0.6,
        edgecolors="white",
        linewidth=0.3,
        zorder=3,
    )
    lims = [min(r_fc.min(), rho.min()) - 0.05, max(r_fc.max(), rho.max()) + 0.05]
    ax_scatter.plot(lims, lims, "k--", alpha=0.3, linewidth=LINE["thin"], label="identity")
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_title(
        f"A. Raw FC vs BS-NET ({atlas.upper()}, N={n})",
        fontweight="bold",
        fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )
    ax_scatter.set_xlabel(r"$r_{FC}$ (raw, 2 min vs full)", fontsize=FONT["axis_label"])
    ax_scatter.set_ylabel(r"$\hat{\rho}_T$ (BS-NET, mean across seeds)", fontsize=FONT["axis_label"])
    ax_scatter.legend(loc="lower right", fontsize=FONT["legend_small"])

    # Panel B: Distribution comparison (Fig 2 Panel D style)
    _violin_box_sd(
        ax_comp,
        data=[r_fc, rho],
        positions=[0, 1],
        colors=[COLOR_RAW, COLOR_BSNET],
    )

    n_improved = int(np.sum(improvement > 0))
    pct = n_improved / n * 100
    ax_comp.set_xticks([0, 1])
    ax_comp.set_xticklabels(["Raw FC\n(2 min)", "BS-NET\n(2 min)"], fontsize=FONT["legend_small"])
    ax_comp.set_title(
        "B. Distribution Comparison",
        fontweight="bold",
        fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )
    ax_comp.set_ylabel("FC similarity", fontsize=FONT["axis_label"])
    ax_comp.set_xlabel("")
    ax_comp.text(
        0.5,
        0.02,
        f"{pct:.1f}% improved ({n_improved}/{n})",
        transform=ax_comp.transAxes,
        ha="center",
        fontsize=FONT["legend_small"],
        fontstyle="italic",
        color="#555555",
    )

    # Panel C: Improvement (Fig 2 Panel D style)
    _violin_box_sd(
        ax_imp,
        data=[improvement],
        positions=[0],
        colors=[COLOR_IMPROVE],
    )
    ax_imp.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_imp.set_xticks([0])
    ax_imp.set_xticklabels(["Δ (ρ̂T − r_FC)"], fontsize=FONT["legend_small"])
    ax_imp.set_title(
        "C. Improvement (Δ)",
        fontweight="bold",
        fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )
    ax_imp.set_ylabel(r"Δ ($\hat{\rho}_T - r_{FC}$)", fontsize=FONT["axis_label"])
    ax_imp.set_xlabel("")

    # Panel D: Seed stability (Fig 2 Panel D style)
    _violin_box_sd(
        ax_seed,
        data=[rho_std],
        positions=[0],
        colors=[COLOR_SEED],
    )
    ax_seed.set_xticks([0])
    ax_seed.set_xticklabels(["Seed σ"], fontsize=FONT["legend_small"])
    ax_seed.set_title(
        "D. Seed Stability",
        fontweight="bold",
        fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )
    ax_seed.set_ylabel("Seed σ (cross-seed SD)", fontsize=FONT["axis_label"])
    ax_seed.set_xlabel("")

    return fig


def main() -> None:
    """Generate Figure 3."""
    parser = argparse.ArgumentParser(description="Figure 3: ABIDE Validation (main)")
    parser.add_argument("--atlas", default="cc200", choices=["cc200", "cc400"])
    parser.add_argument(
        "--filtered-strict",
        action="store_true",
        help="Use strict-filtered CSV (supplementary sensitivity).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data = load_multiseed(args.atlas, filtered_strict=args.filtered_strict)

    r_fc = data["r_fc"]
    rho = data["rho_mean"]
    improvement = rho - r_fc

    print(f"ABIDE {args.atlas.upper()} — N={data['n']}")
    print(f"  r_FC:  {r_fc.mean():.3f} ± {r_fc.std():.3f}")
    print(f"  ρ̂T:   {rho.mean():.3f} ± {rho.std():.3f}")
    print(f"  Δ:     {improvement.mean():+.3f} ± {improvement.std():.3f}")
    print(
        f"  Improved: {np.sum(improvement > 0)}/{data['n']} "
        f"({np.mean(improvement > 0)*100:.1f}%)"
    )
    print(f"  Seed σ: {data['rho_std'].mean():.4f} (mean)")

    fig = plot_figure3(data, args.atlas)
    save_figure(fig, "Fig3_ABIDE_Validation.png")
    print("\nFigure 3 saved: Fig3_ABIDE_Validation.png")


if __name__ == "__main__":
    main()
