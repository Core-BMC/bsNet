"""Figure 2: Component Necessity Analysis (Leave-One-Out).

Validates each BS-NET pipeline component's contribution via leave-one-out
analysis. Primary: ds000243 N=52 (Schaefer 200 cortical), fallback: ABIDE N=468.

Design:
  Panel A: IQR range bar + scatter of ρ̂T per condition (6 conditions)
  Panel B: Bar chart of −Δρ̂T (contribution) with significance markers (***)
  Style: Floating bars with subsampled scatter dots, red diamond mean±SD

Input:  artifacts/reports/component_necessity_ds000243_schaefer200_N52.csv
        (fallback: component_necessity_ABIDE_cc200_N468.csv)
Output: Fig2_ComponentNecessity.png
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

# Allow direct script execution:
#   python src/visualization/plot_figure3_component.py
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.style import (
    CONDITION_PALETTE,
    DOT_COLOR,
    FONT,
    LINE,
    PALETTE,
    apply_bsnet_theme,
    save_figure,
)

logger = logging.getLogger(__name__)

# Condition ordering and display
CONDITION_ORDER: list[str] = [
    "L_full", "L_no_sb", "L_no_lw", "L_no_boot", "L_no_prior", "L_no_atten",
]
CONDITION_LABELS: dict[str, str] = {
    "L_full": "Full\nPipeline",
    "L_no_sb": "w/o\nSpearman-\nBrown",
    "L_no_lw": "w/o\nLedoit-\nWolf",
    "L_no_boot": "w/o\nBootstrap",
    "L_no_prior": "w/o\nBayesian\nPrior",
    "L_no_atten": "w/o\nAttenuation\nCorr.",
}

# Colors: Full=blue(hero), ablated by impact severity
CONDITION_COLORS: dict[str, str] = {
    "L_full": CONDITION_PALETTE["bsnet"],       # blue — full pipeline (hero)
    "L_no_sb": PALETTE["highlight"],             # red — critical component
    "L_no_lw": CONDITION_PALETTE["reference"],   # gray — negligible
    "L_no_boot": PALETTE["pass_excellent"],      # green — inflated
    "L_no_prior": CONDITION_PALETTE["raw"],      # amber — critical
    "L_no_atten": PALETTE["original"],           # pink — modest
}

# Scatter styling
DOT_SIZE = 4.0
DOT_ALPHA = 0.10
MAX_DOTS = 350  # subsample for clarity
BAR_ALPHA = 0.70
BAR_WIDTH = 0.55
JITTER_WIDTH = 0.10  # scatter spread


def _scatter_strip(
    ax: plt.Axes,
    vals: np.ndarray,
    x_pos: float,
    rng: np.random.RandomState,
    color: str = DOT_COLOR,
) -> None:
    """Add jittered scatter strip at x_pos with group color."""
    n = len(vals)
    if n == 0:
        return
    if n > MAX_DOTS:
        idx = rng.choice(n, size=MAX_DOTS, replace=False)
        vals = vals[idx]
        n = MAX_DOTS
    x_jit = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=n)
    ax.scatter(
        x_pos + x_jit, vals,
        s=DOT_SIZE, c=color, alpha=DOT_ALPHA,
        edgecolors="none", zorder=3, rasterized=True,
    )


def _mean_diamond(ax: plt.Axes, vals: np.ndarray, x_pos: float) -> None:
    """Overlay red diamond mean±SD marker."""
    m, sd = np.nanmean(vals), np.nanstd(vals)
    ax.errorbar(
        x_pos, m, yerr=sd,
        fmt="D", color="red", markersize=5.5,
        markeredgecolor="darkred", markeredgewidth=0.8,
        ecolor="darkred", elinewidth=1.2, capsize=3.5, capthick=1.0,
        zorder=10,
    )


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load component necessity CSV."""
    df = pd.read_csv(csv_path)
    df["condition"] = pd.Categorical(
        df["condition"], categories=CONDITION_ORDER, ordered=True,
    )
    return df.dropna(subset=["condition"]).sort_values("condition")


def plot_figure2(df: pd.DataFrame) -> plt.Figure:
    """Create Figure 2: Component Necessity with range bars + scatter.

    Args:
        df: DataFrame with condition, rho_hat_T, delta_from_full.

    Returns:
        Matplotlib Figure object.
    """
    apply_bsnet_theme()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [6, 5], "wspace": 0.32},
    )
    rng = np.random.RandomState(42)

    # ── Panel A: Absolute ρ̂T by condition ──
    for i, cond in enumerate(CONDITION_ORDER):
        vals = df.loc[df["condition"] == cond, "rho_hat_T"].values
        q1, q3 = np.nanpercentile(vals, [25, 75])
        color = CONDITION_COLORS[cond]

        # Floating range bar (IQR: Q1–Q3)
        ax1.bar(
            i, q3 - q1, bottom=q1, width=BAR_WIDTH,
            color=color, alpha=BAR_ALPHA, edgecolor="white",
            linewidth=0.5, zorder=2,
        )
        # Scatter strip (group color)
        _scatter_strip(ax1, vals, i, rng, color=color)
        # Mean±SD diamond
        _mean_diamond(ax1, vals, i)

    # Reference line at full pipeline mean
    full_mean = df.loc[df["condition"] == "L_full", "rho_hat_T"].mean()
    ax1.axhline(
        full_mean, color="#4A90E2", linestyle="--",
        linewidth=LINE["thin"], alpha=0.5,
        label=f"Full = {full_mean:.3f}",
    )
    ax1.set_xticks(range(len(CONDITION_ORDER)))
    ax1.set_xticklabels(
        [CONDITION_LABELS[c] for c in CONDITION_ORDER],
        fontsize=FONT["legend_small"], ha="center",
    )
    ax1.set_title(
        "A. Extrapolated Reliability by Condition",
        fontweight="bold", fontsize=FONT["title"], pad=12,
    )
    ax1.set_ylabel(r"$\hat{\rho}_T$", fontsize=FONT["axis_label"])
    ax1.set_xlabel("")
    ax1.legend(loc="lower left", fontsize=FONT["legend_small"])

    # ── Panel B: Performance drop when removed (−Δ = contribution) ──
    ablated = [c for c in CONDITION_ORDER if c != "L_full"]

    # Compute subject-level means for paired t-tests
    from scipy import stats as sp_stats

    subj_mean = df.groupby(["subject_id", "condition"], observed=True)["rho_hat_T"].mean().reset_index()
    pivot = subj_mean.pivot(index="subject_id", columns="condition", values="rho_hat_T")
    full_vals = pivot["L_full"].values
    n_comparisons = len(ablated)

    bar_tops = []  # track bar tip positions for significance markers
    for i, cond in enumerate(ablated):
        vals = df.loc[df["condition"] == cond, "delta_from_full"].values
        drop = -np.nanmean(vals)  # positive = essential, negative = inflation
        sd = np.nanstd(vals)
        color = CONDITION_COLORS[cond]

        # Standard bar (mean drop)
        ax2.bar(
            i, drop, width=BAR_WIDTH,
            color=color, alpha=BAR_ALPHA, edgecolor="white",
            linewidth=0.5, zorder=2,
        )
        # One-sided error bar: upward for positive, downward for negative
        yerr_up = sd if drop >= 0 else 0
        yerr_down = 0 if drop >= 0 else sd
        ax2.errorbar(
            i, drop, yerr=[[yerr_down], [yerr_up]],
            fmt="none",
            ecolor=color, elinewidth=1.2, capsize=4, capthick=1.0,
            zorder=10,
        )

        # Paired t-test vs Full (Bonferroni corrected)
        abl_vals = pivot[cond].values
        _, p_raw = sp_stats.ttest_rel(full_vals, abl_vals)
        p_corr = min(p_raw * n_comparisons, 1.0)
        sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else ""

        if sig:
            # Place marker above bar+errorbar (positive) or below (negative)
            if drop >= 0:
                y_marker = drop + sd + 0.008
            else:
                y_marker = drop - sd - 0.012
            ax2.text(
                i, y_marker, sig, ha="center", va="center",
                fontsize=FONT["legend"], fontweight="bold", color="#333333",
            )

    ax2.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax2.set_xticks(range(len(ablated)))
    ax2.set_xticklabels(
        [CONDITION_LABELS[c] for c in ablated],
        fontsize=FONT["legend_small"], ha="center",
    )
    ax2.set_title(
        "B. Performance Drop When Removed",
        fontweight="bold", fontsize=FONT["title"], pad=12,
    )
    ax2.set_ylabel(
        r"$-\Delta\hat{\rho}_T$ (contribution)",
        fontsize=FONT["axis_label"],
    )
    ax2.set_xlabel("")

    plt.tight_layout(pad=2.5)
    return fig


def main() -> None:
    """Run Figure 2 generation pipeline."""
    parser = argparse.ArgumentParser(description="Figure 2: Component Necessity")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to component_necessity CSV (auto-detect if omitted)",
    )
    args = parser.parse_args()

    # Auto-detect CSV
    if args.csv:
        csv_path = Path(args.csv)
    else:
        candidates = [
            Path("artifacts/reports/component_necessity_ds000243_schaefer200_N52.csv"),
            Path("artifacts/reports/component_necessity_ABIDE_cc200_N468.csv"),
            Path("artifacts/reports/component_necessity_Synthetic_50-900-120.csv"),
            Path("artifacts/reports/component_necessity.csv"),
        ]
        csv_path = next((c for c in candidates if c.exists()), None)
        if csv_path is None:
            raise FileNotFoundError(
                f"No component necessity CSV found. Checked: {[str(c) for c in candidates]}"
            )

    print(f"Loading: {csv_path}")
    df = load_data(csv_path)
    n_subjects = df["subject_id"].nunique() if "subject_id" in df.columns else "?"
    n_seeds = df["seed"].nunique() if "seed" in df.columns else "?"
    print(f"  N={n_subjects} subjects, {n_seeds} seeds, {len(df)} rows")

    # Summary stats
    print("\n--- Component Necessity Summary ---")
    for cond in CONDITION_ORDER:
        vals = df.loc[df["condition"] == cond, "rho_hat_T"]
        delta = df.loc[df["condition"] == cond, "delta_from_full"]
        print(
            f"  {cond:<15} "
            f"ρ̂T={vals.mean():.3f}±{vals.std():.3f}  "
            f"Δ={delta.mean():+.3f}±{delta.std():.3f}"
        )

    fig = plot_figure2(df)
    save_figure(fig, "Fig2_ComponentNecessity.png")
    print("\nFigure 2 saved: Fig2_ComponentNecessity.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
