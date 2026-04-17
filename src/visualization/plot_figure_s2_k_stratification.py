"""
Supplementary Figure S2: k-Factor Stratification Analysis.

Shows why subjects with low extrapolation ratio (k = total_TRs / short_TRs)
show minimal BS-NET improvement, justifying the k≥3 threshold in the main
Figure 2.

Panels:
  A. Progressive ablation by k-group (4 groups × 4 levels)
  B. Δ(Full − Raw) vs k-factor scatter + dose-response
  C. Per-site summary: N, k, Raw, Full, Δ

Usage:
    python -m src.visualization.plot_figure_s2_k_stratification
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
from matplotlib.patches import Patch

from src.visualization.style import (
    FONT,
    LINE,
    apply_bsnet_theme,
    save_figure,
    style_axis,
)

logger = logging.getLogger(__name__)

# k-group definitions
K_GROUPS = [
    ("k < 2", 0.0, 2.0, "#d7191c"),      # red — excluded
    ("k = 2–3", 2.0, 3.0, "#fdae61"),     # amber — marginal
    ("k = 3–4", 3.0, 4.0, "#abd9e9"),     # light blue — included
    ("k ≥ 4", 4.0, 99.0, "#4A90E2"),      # blue — ideal
]

PROG_4_LEVELS = ["L0", "L1", "L3", "L5"]
PROG_4_LABELS = ["Raw", "+LW", "+Corr", "Full"]

# ABIDE site → short name mapping for readability
SITE_SHORT = {
    "CALTECH": "Caltech", "CMU": "CMU", "KKI": "KKI",
    "LEUVEN_1": "Leuven-1", "LEUVEN_2": "Leuven-2",
    "MAX_MUN": "MaxMun", "NYU": "NYU", "OHSU": "OHSU",
    "OLIN": "Olin", "PITT": "Pitt", "SBL": "SBL",
    "SDSU": "SDSU", "STANFORD": "Stanford",
    "TRINITY": "Trinity", "UCLA_1": "UCLA-1", "UCLA_2": "UCLA-2",
    "UM_1": "UMich-1", "UM_2": "UMich-2", "USM": "USM",
    "YALE": "Yale",
}


def _load_and_annotate(
    prog_csv: str | Path,
    npy_dir: str | Path,
    pheno_csv: str | Path,
    short_samples: int = 60,
) -> pd.DataFrame:
    """Load progressive ablation CSV and annotate with k-factor and site."""
    df = pd.read_csv(prog_csv)

    # k-factor from .npy shapes
    npy_dir = Path(npy_dir)
    k_map = {}
    for f in npy_dir.glob("*.npy"):
        total_trs = np.load(str(f), mmap_mode="r").shape[0]
        k_map[f.stem] = total_trs / short_samples
    df["k_factor"] = df["subject_id"].map(k_map)

    # Site from phenotypic
    pheno = pd.read_csv(pheno_csv)
    site_map = dict(zip(
        pheno["SUB_ID"].astype(str),
        pheno["SITE_ID"],
    ))
    df["sub_id_num"] = df["subject_id"].str.replace("_cc200", "")
    df["site"] = df["sub_id_num"].map(site_map)

    # k-group
    def _assign_group(k):
        for label, lo, hi, _ in K_GROUPS:
            if lo <= k < hi:
                return label
        return K_GROUPS[-1][0]
    df["k_group"] = df["k_factor"].apply(_assign_group)

    return df


def _subj_by_level(df: pd.DataFrame, level: str) -> np.ndarray:
    return df[df["level"] == level].groupby(
        "subject_id"
    )["rho_hat_T"].mean().values


# ============================================================================
# Panel A: Progressive ablation by k-group
# ============================================================================

def _plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    """4 k-groups × 4 progressive levels as grouped boxplots."""
    n_groups = len(K_GROUPS)
    n_levels = len(PROG_4_LEVELS)
    total_positions = n_groups * n_levels + (n_groups - 1)  # gaps between groups

    box_data = []
    positions = []
    colors = []
    pos = 0

    for g_idx, (g_label, k_lo, k_hi, g_color) in enumerate(K_GROUPS):
        g_df = df[(df["k_factor"] >= k_lo) & (df["k_factor"] < k_hi)]
        n_g = g_df["subject_id"].nunique()

        for l_idx, lev in enumerate(PROG_4_LEVELS):
            vals = _subj_by_level(g_df, lev)
            box_data.append(vals if vals.size else np.array([np.nan]))
            positions.append(pos)
            colors.append(g_color)
            pos += 1
        pos += 1  # gap between groups

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.7,
        patch_artist=True, showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3,
                        markeredgecolor="none"),
        medianprops=dict(color="white", linewidth=1.3),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )

    for i, (patch, color) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color)
    for i, flier in enumerate(bp["fliers"]):
        flier.set_markerfacecolor(colors[i])

    # Group labels on x-axis
    group_centers = []
    pos = 0
    for g_idx, (g_label, k_lo, k_hi, _) in enumerate(K_GROUPS):
        center = pos + (n_levels - 1) / 2
        group_centers.append(center)
        n_g = df[(df["k_factor"] >= k_lo) & (df["k_factor"] < k_hi)]["subject_id"].nunique()

        # Level labels (small, below group label)
        for l_idx, l_label in enumerate(PROG_4_LABELS):
            ax.text(pos + l_idx, 0.47, l_label, ha="center", va="top",
                    fontsize=6, color="#666666", rotation=45)
        pos += n_levels + 1

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [f"{g[0]}\n(N={df[(df['k_factor'] >= g[1]) & (df['k_factor'] < g[2])]['subject_id'].nunique()})"
         for g in K_GROUPS],
        fontsize=FONT["tick"] - 0.5,
    )
    ax.set_ylim(bottom=0.45, top=1.02)

    # k≥3 threshold line annotation
    threshold_pos = 2 * (n_levels + 1) - 0.5  # between group 2 and 3
    ax.axvline(x=threshold_pos, color="#d7191c", linewidth=1.5,
               linestyle="--", alpha=0.6, zorder=0)
    ax.text(threshold_pos + 0.3, 0.98, "k≥3\nthreshold",
            fontsize=FONT["annotation"] - 2, color="#d7191c",
            va="top", fontstyle="italic")

    style_axis(ax, "A. Progressive Ablation by k-Group",
               ylabel=r"Estimated Reliability ($\hat{\rho}_T$)")


# ============================================================================
# Panel B: Δ(Full − Raw) vs k scatter
# ============================================================================

def _plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Scatter: per-subject Δ(Full−Raw) vs k-factor."""
    # Per-subject means
    subj = df.groupby(["subject_id", "level", "k_factor"]).agg(
        rho=("rho_hat_T", "mean"),
    ).reset_index()

    raw = subj[subj["level"] == "L0"][["subject_id", "k_factor", "rho"]].rename(
        columns={"rho": "raw"})
    full = subj[subj["level"] == "L5"][["subject_id", "rho"]].rename(
        columns={"rho": "full"})
    merged = raw.merge(full, on="subject_id")
    merged["delta"] = merged["full"] - merged["raw"]

    # Color by k-group
    for g_label, k_lo, k_hi, g_color in K_GROUPS:
        mask = (merged["k_factor"] >= k_lo) & (merged["k_factor"] < k_hi)
        subset = merged[mask]
        ax.scatter(subset["k_factor"], subset["delta"],
                   c=g_color, s=15, alpha=0.5, edgecolors="none",
                   label=f"{g_label} (N={len(subset)})", zorder=2)

    # Trend line (linear fit)
    k_vals = merged["k_factor"].values
    d_vals = merged["delta"].values
    valid = ~np.isnan(k_vals) & ~np.isnan(d_vals)
    if valid.sum() > 10:
        z = np.polyfit(k_vals[valid], d_vals[valid], 1)
        k_range = np.linspace(k_vals.min(), k_vals.max(), 100)
        ax.plot(k_range, np.polyval(z, k_range), "k--", linewidth=1.5,
                alpha=0.5, label=f"slope={z[0]:.3f}")

    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(x=3.0, color="#d7191c", linewidth=1.5, linestyle="--",
               alpha=0.6, label="k=3 threshold")

    # Annotation
    ax.text(1.5, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.1,
            "short ≈ total\n(no room to\nimprove)",
            fontsize=FONT["annotation"] - 2, color="#d7191c",
            ha="center", fontstyle="italic")

    style_axis(ax, "B. Improvement vs Extrapolation Ratio",
               xlabel="k = total TRs / short TRs",
               ylabel=r"$\Delta(\hat{\rho}_T - r_{FC})$")
    # Re-place legend outside plot area (right side)
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=FONT["legend_small"], borderaxespad=0,
        framealpha=0.9,
    )


# ============================================================================
# Panel C: Per-site summary table
# ============================================================================

def _plot_panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Per-site summary as a formatted table."""
    ax.axis("off")

    # Compute per-site stats
    subj = df.groupby(["subject_id", "level", "site", "k_factor"]).agg(
        rho=("rho_hat_T", "mean"),
    ).reset_index()

    sites = []
    for site in sorted(df["site"].dropna().unique()):
        s = subj[subj["site"] == site]
        n = s["subject_id"].nunique()
        k = s["k_factor"].iloc[0]
        raw_m = s[s["level"] == "L0"]["rho"].mean()
        full_m = s[s["level"] == "L5"]["rho"].mean()
        delta = full_m - raw_m
        sites.append({
            "site": SITE_SHORT.get(site, site),
            "n": n, "k": k,
            "raw": raw_m, "full": full_m, "delta": delta,
        })

    sites_df = pd.DataFrame(sites).sort_values("k")

    header = f"{'Site':<12} {'N':>4} {'k':>5} {'Raw':>8} {'Full':>8} {'Δ':>8}  {'Status':<10}"
    sep = "─" * 62
    lines = [
        "Per-Site Summary (ABIDE PCP, CC200)",
        "",
        header,
        sep,
    ]

    for _, row in sites_df.iterrows():
        status = "excluded" if row["k"] < 3 else "included"
        marker = "✗" if row["k"] < 3 else "✓"
        lines.append(
            f"{row['site']:<12} {row['n']:>4} {row['k']:>5.1f} "
            f"{row['raw']:>8.3f} {row['full']:>8.3f} {row['delta']:>+8.3f}  "
            f"{marker} {status}"
        )

    n_excl = sites_df[sites_df["k"] < 3]["n"].sum()
    n_incl = sites_df[sites_df["k"] >= 3]["n"].sum()
    lines.extend([
        sep,
        f"Included (k≥3): {n_incl} subjects from "
        f"{(sites_df['k'] >= 3).sum()} sites",
        f"Excluded (k<3): {n_excl} subjects from "
        f"{(sites_df['k'] < 3).sum()} sites",
        "",
        "Exclusion rationale: when short/total > 33% (k<3),",
        "reference FC is too noisy for meaningful validation.",
    ])

    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=FONT["annotation"] - 1.5,
        fontfamily="monospace", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=1.2", facecolor="#f8f8f8",
                  edgecolor="#cccccc", alpha=0.9),
    )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplementary Figure S2: k-factor stratification",
    )
    parser.add_argument("--prog-abide",
                        default="artifacts/reports/progressive_ablation_abide_cc200_N468.csv")
    parser.add_argument("--npy-dir",
                        default="data/abide/timeseries_cache/cc200")
    parser.add_argument("--pheno",
                        default="data/abide/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Loading and annotating data...")
    df = _load_and_annotate(args.prog_abide, args.npy_dir, args.pheno)
    logger.info(f"  {df['subject_id'].nunique()} subjects, "
                f"k range: {df['k_factor'].min():.1f}–{df['k_factor'].max():.1f}")

    apply_bsnet_theme()

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.45,
                          height_ratios=[1, 1])

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    _plot_panel_a(ax_a, df)
    _plot_panel_b(ax_b, df)
    _plot_panel_c(ax_c, df)

    save_figure(fig, "FigS2_k_Stratification.png")
    logger.info("Saved: FigS2_k_Stratification.png")


if __name__ == "__main__":
    main()
