"""
Supplementary Figure S1: Full 6-Level Progressive Ablation by k-Group.

Shows the complete L0→L5 progressive component addition stratified by
extrapolation ratio (k = total_TRs / short_TRs). Demonstrates that:
  - k<2: BS-NET cannot improve (short ≈ total)
  - k=2–3: minimal improvement
  - k=3–4: improvement visible, L2 dip from bootstrap variance
  - k≥4: strong improvement, monotonic from L3 onward
  - ds000243 (k=10): strongest, cleanly monotonic

Justifies the k≥3 threshold in main Figure 2 and the merging of Bootstrap
into "Correction" in the 4-level progressive view.

Layout (3×2):
  A. ds000243 (k=10)     B. ABIDE k<2
  C. ABIDE k=2–3         D. ABIDE k=3–4
  E. ABIDE k≥4           F. Summary table

Usage:
    python -m src.visualization.plot_figure_s1_progressive_full
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.style import (
    FONT,
    apply_bsnet_theme,
    save_figure,
    style_axis,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

C_DS000243 = "#4A90E2"
C_FULL = "#4A90E2"

K_GROUPS = [
    ("k < 2",  0.0, 2.0, "#d7191c"),     # red — excluded
    ("k = 2–3", 2.0, 3.0, "#fdae61"),    # amber — marginal
    ("k = 3–4", 3.0, 4.0, "#abd9e9"),    # light blue — included
    ("k ≥ 4",  4.0, 99.0, "#4A90E2"),    # blue — ideal
]

PROG_6_LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]
PROG_6_LABELS = [
    "Raw\n$r_{FC}$", "+LW", "+Boot",
    "+SB+Att\n(no Prior)", "+Prior",
    "Full\nBS-NET",
]


# ============================================================================
# Helpers
# ============================================================================

def _subj_by_level(df: pd.DataFrame, level: str) -> np.ndarray:
    return df[df["level"] == level].groupby(
        "subject_id"
    )["rho_hat_T"].mean().values


def _annotate_abide_k(
    df: pd.DataFrame,
    npy_dir: str | Path,
    short_samples: int = 60,
) -> pd.DataFrame:
    """Add k_factor column to ABIDE progressive ablation DataFrame."""
    npy_dir = Path(npy_dir)
    k_map = {}
    for f in npy_dir.glob("*.npy"):
        total_trs = np.load(str(f), mmap_mode="r").shape[0]
        k_map[f.stem] = total_trs / short_samples
    df = df.copy()
    df["k_factor"] = df["subject_id"].map(k_map)
    return df


def _filter_k_group(
    df: pd.DataFrame, k_lo: float, k_hi: float,
) -> pd.DataFrame:
    return df[(df["k_factor"] >= k_lo) & (df["k_factor"] < k_hi)]


# ============================================================================
# 6-level boxplot panel
# ============================================================================

def _plot_6level_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    color: str,
    title: str,
    ylim_bottom: float = 0.5,
) -> None:
    """6-level boxplot for one dataset/group."""
    data = [_subj_by_level(df, lev) for lev in PROG_6_LEVELS]
    n_subj = len(data[0]) if data[0].size else 0
    x = np.arange(len(PROG_6_LEVELS))

    if n_subj == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#999999")
        ax.set_title(title, fontweight="bold", fontsize=FONT["title"],
                     pad=FONT["title_pad"])
        return

    bp = ax.boxplot(
        data, positions=x, widths=0.55,
        patch_artist=True, showfliers=True,
        flierprops=dict(marker=".", markersize=3,
                        markerfacecolor=color,
                        markeredgecolor="none", alpha=0.4),
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color=color, linewidth=1.0),
        capprops=dict(color=color, linewidth=1.0),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color)

    # Median annotations
    for i, d in enumerate(data):
        if d.size:
            med = np.median(d)
            ax.text(i, med + 0.010, f"{med:.3f}", ha="center", va="bottom",
                    fontsize=FONT["annotation"] - 2.5, color="#333333")

    # L2 dip arrow
    if data[1].size and data[2].size:
        med_l1 = np.median(data[1])
        med_l2 = np.median(data[2])
        if med_l2 < med_l1 - 0.02:
            y_arrow = med_l2 - 0.015
            y_text = max(ylim_bottom + 0.03, med_l2 - 0.08)
            if y_text < ylim_bottom + 0.02:
                y_text = ylim_bottom + 0.03
            ax.annotate(
                "Boot.\nvar.",
                xy=(2, y_arrow), xytext=(2.7, y_text),
                fontsize=FONT["annotation"] - 3,
                color="#d7191c", fontstyle="italic",
                arrowprops=dict(arrowstyle="->", color="#d7191c", lw=1.0),
                ha="center",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(PROG_6_LABELS, fontsize=FONT["tick"] - 1.5)
    ax.set_ylim(bottom=ylim_bottom)

    # Shade zones
    ax.axvspan(-0.5, 1.5, alpha=0.04, color="#95a5a6", zorder=0)
    ax.axvspan(1.5, 5.5, alpha=0.06, color=C_FULL, zorder=0)

    style_axis(ax, f"{title} (N={n_subj})",
               ylabel=r"$\hat{\rho}_T$")


# ============================================================================
# Summary table panel
# ============================================================================

def _plot_summary_table(
    ax: plt.Axes,
    df_ds: pd.DataFrame,
    df_ab: pd.DataFrame,
) -> None:
    """Summary table: median per level × k-group."""
    ax.axis("off")

    # Compute per k-group medians
    groups = [("ds000243\n(k=10)", df_ds)] + [
        (f"ABIDE\n{label}", _filter_k_group(df_ab, lo, hi))
        for label, lo, hi, _ in K_GROUPS
    ]

    header_names = ["ds000243", "k<2", "k=2–3", "k=3–4", "k≥4"]
    n_vals = [g[1]["subject_id"].nunique() for g in groups]

    lines = [
        "6-Level Progressive Ablation — Median by k-Group",
        "",
        f"{'Level':<14}" + "".join(f"{n:>10}" for n in header_names),
        f"{'(N)':<14}" + "".join(f"{'('+str(v)+')':>10}" for v in n_vals),
        "─" * (14 + 10 * len(header_names)),
    ]

    for lev, label in zip(PROG_6_LEVELS,
                          ["Raw", "+LW", "+Boot", "+SB+Att", "+Prior", "Full"]):
        row = f"{lev} {label:<9}"
        for _, g_df in groups:
            vals = _subj_by_level(g_df, lev)
            med = np.median(vals) if vals.size else np.nan
            row += f"{med:>10.3f}"
        lines.append(row)

    # Δ(Full − Raw) row
    lines.append("─" * (14 + 10 * len(header_names)))
    row = "Δ(Full−Raw)  "
    for _, g_df in groups:
        raw_v = _subj_by_level(g_df, "L0")
        full_v = _subj_by_level(g_df, "L5")
        if raw_v.size and full_v.size:
            delta = np.median(full_v) - np.median(raw_v)
            row += f"{delta:>+10.3f}"
        else:
            row += f"{'—':>10}"
    lines.append(row)

    lines.extend([
        "",
        "Observations:",
        "• k<2: short ≈ total → BS-NET cannot improve (Δ ≈ 0)",
        "• k=2–3: marginal improvement, non-monotonic (L2 dip)",
        "• k≥3: consistent improvement, Δ scales with k",
        "• L2 dip (Bootstrap) present in all ABIDE groups but",
        "  absorbed by subsequent correction steps (L3–L5)",
        "• In main Fig 2, Bootstrap merged into 'Correction'",
    ])

    ax.text(
        0.03, 0.97, "\n".join(lines),
        transform=ax.transAxes, fontsize=FONT["annotation"] - 1.5,
        fontfamily="monospace", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=1.2", facecolor="#f8f8f8",
                  edgecolor="#cccccc", alpha=0.9),
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplementary Figure S1: 6-level progressive by k-group",
    )
    parser.add_argument("--prog-ds",
                        default="artifacts/reports/progressive_ablation_ds000243_cc200_N52.csv")
    parser.add_argument("--prog-abide",
                        default="artifacts/reports/progressive_ablation_abide_cc200_N468.csv")
    parser.add_argument("--npy-dir",
                        default="data/abide/timeseries_cache/cc200",
                        help="ABIDE .npy directory for k-factor computation.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df_ds = pd.read_csv(args.prog_ds)
    df_ab = pd.read_csv(args.prog_abide)

    # Annotate ABIDE with k-factor
    df_ab = _annotate_abide_k(df_ab, args.npy_dir)
    logger.info(f"ABIDE k range: {df_ab['k_factor'].min():.1f}–{df_ab['k_factor'].max():.1f}")

    apply_bsnet_theme()

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.28)

    # Row 1: ds000243 + ABIDE k<2
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Row 2: ABIDE k=2-3 + k=3-4
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Row 3: ABIDE k≥4 + Summary table
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    _plot_6level_boxplot(ax_a, df_ds, C_DS000243, "A. ds000243 (k=10)")

    for ax, panel, (label, lo, hi, color) in zip(
        [ax_b, ax_c, ax_d, ax_e],
        ["B", "C", "D", "E"],
        K_GROUPS,
    ):
        g_df = _filter_k_group(df_ab, lo, hi)
        _plot_6level_boxplot(ax, g_df, color, f"{panel}. ABIDE {label}")

    _plot_summary_table(ax_f, df_ds, df_ab)

    save_figure(fig, "FigS1_Progressive_6Level.png")
    logger.info("Saved: FigS1_Progressive_6Level.png")


if __name__ == "__main__":
    main()
