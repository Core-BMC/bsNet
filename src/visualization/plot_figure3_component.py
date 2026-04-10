"""Generate Figure 3: Component Necessity Analysis (ABIDE real-data).

Validates each BS-NET pipeline component's contribution via leave-one-out
analysis on ABIDE N=468 subjects, 10 seeds.

Design:
  Panel A: Violin + jitter of ρ̂T per condition (6 conditions)
  Panel B: Violin + jitter of Δρ̂T from full pipeline (5 ablated conditions)
  Style: Matches Figure 4/7 (no outline, uniform alpha, dark jitter dots,
         red diamond mean±SD)

Input:  artifacts/reports/component_necessity_ABIDE_cc200_N468.csv
        (fallback: component_necessity_Synthetic_50-900-120.csv)
Output: Figure3_ComponentNecessity.png
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
import seaborn as sns

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
    "L_full": "Full Pipeline",
    "L_no_sb": "w/o Spearman-Brown",
    "L_no_lw": "w/o Ledoit-Wolf",
    "L_no_boot": "w/o Bootstrap",
    "L_no_prior": "w/o Bayesian Prior",
    "L_no_atten": "w/o Attenuation Corr.",
}

# Fig 4/7 color schema: Full=blue(hero), ablated conditions by impact
CONDITION_COLORS: dict[str, str] = {
    "L_full": CONDITION_PALETTE["bsnet"],       # blue — full pipeline (hero)
    "L_no_sb": PALETTE["highlight"],             # red — critical component
    "L_no_lw": CONDITION_PALETTE["reference"],   # gray — negligible
    "L_no_boot": PALETTE["pass_excellent"],      # green — inflated
    "L_no_prior": CONDITION_PALETTE["raw"],      # amber — critical
    "L_no_atten": PALETTE["original"],           # pink — modest
}

# Δρ panel: color by direction/magnitude
DELTA_NEG_COLOR = PALETTE["highlight"]            # red for negative (performance drop)
DELTA_POS_COLOR = PALETTE["pass_excellent"]       # green for positive (inflation)
DELTA_NEUTRAL_COLOR = CONDITION_PALETTE["reference"]  # gray for negligible

# Scatter styling (matching Fig 4/7) — DOT_COLOR imported from style
DOT_SIZE = 3.6
DOT_ALPHA = 0.35
JITTER_X_SIGMA = 0.08
JITTER_Y_SIGMA_RHO = 0.012
JITTER_Y_SIGMA_DELTA = 0.005
MAX_DOTS_PER_CONDITION = 300  # subsample to avoid over-plotting


VIOLIN_ALPHA = 0.65


def _set_violin_alpha(ax: plt.Axes, alpha: float = VIOLIN_ALPHA) -> None:
    """Set uniform alpha on all violin body PolyCollections."""
    from matplotlib.collections import PolyCollection

    for art in ax.collections:
        if isinstance(art, PolyCollection):
            art.set_alpha(alpha)


def _jitter_scatter(
    ax: plt.Axes,
    vals: np.ndarray,
    x_pos: float,
    rng: np.random.RandomState,
    y_sigma: float = 0.003,
    x_sigma: float | None = None,
) -> None:
    """Add dark jittered scatter dots at a given x position.

    Subsamples to MAX_DOTS_PER_CONDITION if n is large.
    """
    n = len(vals)
    if n == 0:
        return
    if n > MAX_DOTS_PER_CONDITION:
        idx = rng.choice(n, size=MAX_DOTS_PER_CONDITION, replace=False)
        vals = vals[idx]
        n = MAX_DOTS_PER_CONDITION
    x_sig = x_sigma if x_sigma is not None else JITTER_X_SIGMA
    x_jit = rng.normal(0, x_sig, size=n)
    y_jit = rng.normal(0, y_sigma, size=n)
    ax.scatter(
        x_pos + x_jit, vals + y_jit,
        s=DOT_SIZE, c=DOT_COLOR, alpha=DOT_ALPHA,
        edgecolors="none", zorder=4,
    )


def _add_mean_sd_diamond(
    ax: plt.Axes,
    vals: np.ndarray,
    x_pos: float,
) -> None:
    """Overlay red diamond mean±SD marker."""
    m = np.nanmean(vals)
    sd = np.nanstd(vals)
    ax.errorbar(
        x_pos, m, yerr=sd,
        fmt="D", color="red", markersize=6,
        markeredgecolor="darkred", markeredgewidth=1.0,
        ecolor="darkred", elinewidth=1.2, capsize=3, capthick=1.2,
        zorder=15,
    )


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load component necessity CSV.

    Args:
        csv_path: Path to CSV file.

    Returns:
        DataFrame with condition, rho_hat_T, delta_from_full columns.
    """
    df = pd.read_csv(csv_path)
    # Ensure condition ordering
    df["condition"] = pd.Categorical(
        df["condition"], categories=CONDITION_ORDER, ordered=True,
    )
    return df.dropna(subset=["condition"]).sort_values("condition")


def plot_figure3(df: pd.DataFrame) -> plt.Figure:
    """Create Figure 3: Component Necessity (violin + jitter style).

    Args:
        df: DataFrame with condition, rho_hat_T, delta_from_full.

    Returns:
        Matplotlib Figure object.
    """
    apply_bsnet_theme()
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[6, 5], wspace=0.30)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    rng = np.random.RandomState(42)

    # ── Panel A: Absolute ρ̂T by condition ──
    palette_a = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.violinplot(
            data=df, x="condition", y="rho_hat_T",
            order=CONDITION_ORDER,
            palette=palette_a, inner="box", linewidth=0, width=0.75,
            cut=2, ax=ax1,
        )
    _set_violin_alpha(ax1)

    for i, cond in enumerate(CONDITION_ORDER):
        vals = df.loc[df["condition"] == cond, "rho_hat_T"].values
        _jitter_scatter(ax1, vals, i, rng, y_sigma=JITTER_Y_SIGMA_RHO)
        _add_mean_sd_diamond(ax1, vals, i)

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
        fontsize=FONT["legend_small"], rotation=45, ha="right",
    )
    ax1.set_title(
        "A. Extrapolated Reliability by Condition",
        fontweight="bold", fontsize=FONT["title"],
    )
    ax1.set_ylabel(r"$\hat{\rho}_T$", fontsize=FONT["axis_label"])
    ax1.set_xlabel("")
    ax1.legend(loc="lower left", fontsize=FONT["legend_small"])

    # ── Panel B: Δρ̂T from full pipeline (exclude L_full) ──
    ablated = [c for c in CONDITION_ORDER if c != "L_full"]
    df_abl = df[df["condition"].isin(ablated)].copy()

    # Same per-condition colors as Panel A
    palette_b = [CONDITION_COLORS[c] for c in ablated]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sns.violinplot(
            data=df_abl, x="condition", y="delta_from_full",
            order=ablated,
            palette=palette_b, inner="box", linewidth=0, width=0.85,
            cut=2, bw_adjust=0.8, ax=ax2,
        )
    _set_violin_alpha(ax2)

    for i, cond in enumerate(ablated):
        vals = df_abl.loc[df_abl["condition"] == cond, "delta_from_full"].values
        _jitter_scatter(ax2, vals, i, rng, y_sigma=JITTER_Y_SIGMA_DELTA)
        _add_mean_sd_diamond(ax2, vals, i)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(range(len(ablated)))
    ax2.set_xticklabels(
        [CONDITION_LABELS[c] for c in ablated],
        fontsize=FONT["legend_small"], rotation=45, ha="right",
    )
    ax2.set_title(
        "B. Component Contribution (Leave-One-Out)",
        fontweight="bold", fontsize=FONT["title"],
    )
    ax2.set_ylabel(r"$\Delta\hat{\rho}_T$ from Full Pipeline", fontsize=FONT["axis_label"])
    ax2.set_xlabel("")

    plt.tight_layout(pad=3.0)
    return fig


def main() -> None:
    """Run Figure 3 generation pipeline."""
    parser = argparse.ArgumentParser(description="Figure 3: Component Necessity")
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
            f"  {CONDITION_LABELS[cond].replace(chr(10), ' '):<28} "
            f"ρ̂T={vals.mean():.3f}±{vals.std():.3f}  "
            f"Δ={delta.mean():+.3f}±{delta.std():.3f}"
        )

    fig = plot_figure3(df)
    save_figure(fig, "Figure3_ComponentNecessity.png")
    print("\nFigure 3 saved: Figure3_ComponentNecessity.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
