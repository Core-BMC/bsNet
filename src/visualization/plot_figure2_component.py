"""
Figure 2: Component Necessity Analysis for BS-NET.

Four-panel figure answering "why each pipeline component is needed":
  A. Leave-one-out (ABIDE k≥3): Δ from full when each component is removed
  B. Progressive addition (4-level): boxplots, ds000243 + ABIDE k≥3
  C. Cross-dataset replication: Leave-one-out ds000243 vs ABIDE k≥3
  D. Per-subject distributions: Raw vs Full BS-NET

ABIDE subjects with k < K_MIN (default 3) are excluded from the main figure.
Rationale: short/total > 33% ⇒ reference FC unreliable. See Supp Fig S2.

Usage:
    python -m src.visualization.plot_figure2_component
    python -m src.visualization.plot_figure2_component --k-min 2.5
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
    CONDITION_PALETTE,
    FONT,
    LINE,
    MARKER,
    apply_bsnet_theme,
    save_figure,
    style_axis,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Colors
# ============================================================================
C_CRITICAL = "#d7191c"
C_MODERATE = "#fdae61"
C_NEGLIGIBLE = "#abd9e9"
C_FULL = "#4A90E2"
C_DS000243 = "#4A90E2"
C_ABIDE = "#fdae61"

# ============================================================================
# Condition / level definitions
# ============================================================================
LOO_CONDITIONS = ["L_full", "L_no_sb", "L_no_prior", "L_no_atten",
                  "L_no_boot", "L_no_lw"]
LOO_LABELS = {
    "L_full": "Full Pipeline",
    "L_no_sb": "w/o SB",
    "L_no_prior": "w/o Prior",
    "L_no_atten": "w/o Attenuation",
    "L_no_boot": "w/o Bootstrap",
    "L_no_lw": "w/o LW Shrinkage",
}

PROG_4_LEVELS = ["L0", "L1", "L3", "L5"]
PROG_4_LABELS = [
    "Raw $r_{FC}$", "+Shrinkage",
    "+Correction\n(Boot+SB+Atten)", "Full BS-NET\n(+Prior)",
]


# ============================================================================
# k-factor computation & filtering
# ============================================================================

def compute_abide_k_map(
    npy_dir: str | Path,
    short_samples: int = 60,
) -> pd.DataFrame:
    """Compute k-factor for each ABIDE subject from .npy file shapes.

    Args:
        npy_dir: Directory with subject .npy files.
        short_samples: Number of TRs used as short observation.

    Returns:
        DataFrame with columns: subject_id, total_trs, k_factor.
    """
    npy_dir = Path(npy_dir)
    rows = []
    for f in sorted(npy_dir.glob("*.npy")):
        total_trs = np.load(str(f), mmap_mode="r").shape[0]
        rows.append({
            "subject_id": f.stem,
            "total_trs": total_trs,
            "k_factor": total_trs / short_samples,
        })
    return pd.DataFrame(rows)


def filter_by_k(
    df: pd.DataFrame,
    k_map: pd.DataFrame,
    k_min: float,
) -> pd.DataFrame:
    """Filter DataFrame to subjects with k >= k_min.

    Args:
        df: Data with 'subject_id' column.
        k_map: DataFrame from compute_abide_k_map().
        k_min: Minimum k-factor threshold.

    Returns:
        Filtered DataFrame.
    """
    valid_ids = set(k_map[k_map["k_factor"] >= k_min]["subject_id"])
    return df[df["subject_id"].isin(valid_ids)].copy()


# ============================================================================
# Summary helpers
# ============================================================================

def _loo_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition mean ± std and Δ from full (subject-level)."""
    subj = df.groupby(["subject_id", "condition"])["rho_hat_T"].mean().reset_index()
    full_by_subj = subj[subj["condition"] == "L_full"].set_index("subject_id")["rho_hat_T"]
    full_mean = full_by_subj.mean()
    rows = []
    for cond in LOO_CONDITIONS:
        cond_by_subj = subj[subj["condition"] == cond].set_index("subject_id")["rho_hat_T"]
        common = cond_by_subj.index.intersection(full_by_subj.index)
        delta_vals = cond_by_subj.loc[common].values - full_by_subj.loc[common].values
        rows.append({
            "condition": cond,
            "label": LOO_LABELS[cond],
            "mean": cond_by_subj.mean(),
            "std": cond_by_subj.std(),
            "delta": cond_by_subj.mean() - full_mean,
            "delta_std": np.std(delta_vals),
        })
    return pd.DataFrame(rows)


def _subj_by_level(df: pd.DataFrame, level: str) -> np.ndarray:
    """Per-subject mean across seeds for a given level."""
    return df[df["level"] == level].groupby("subject_id")["rho_hat_T"].mean().values


# ============================================================================
# Panel A: Leave-one-out
# ============================================================================

def _plot_panel_a(ax: plt.Axes, df: pd.DataFrame, dataset_label: str) -> None:
    """Leave-one-out horizontal bars with error bars."""
    summary = _loo_summary(df)
    removed = summary[summary["condition"] != "L_full"].sort_values("delta")

    y_pos = np.arange(len(removed))
    deltas = removed["delta"].values
    delta_stds = removed["delta_std"].values
    labels_arr = removed["label"].values

    colors = [C_CRITICAL if abs(d) > 0.05
              else C_MODERATE if abs(d) > 0.01
              else C_NEGLIGIBLE for d in deltas]

    bars = ax.barh(y_pos, deltas, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.60, alpha=0.85, zorder=2)

    # Error bars aligned to bar tips
    for i, (delta, sd) in enumerate(zip(deltas, delta_stds)):
        ax.errorbar(delta, y_pos[i], xerr=sd, fmt="none",
                    ecolor="#333333", elinewidth=1.0, capsize=3, capthick=0.8,
                    zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_arr, fontsize=FONT["tick"])
    ax.axvline(x=0, color="black", linewidth=0.8)

    style_axis(ax, f"A. Leave-One-Out ({dataset_label})",
               xlabel=r"$\Delta$ from Full Pipeline")


# ============================================================================
# Panel B: Progressive addition — floating boxplots
# ============================================================================

def _plot_panel_b(
    ax: plt.Axes,
    df_ds: pd.DataFrame,
    df_abide: pd.DataFrame,
    levels: list[str],
    labels: list[str],
    ds_label: str = "ds000243",
    ab_label: str = "ABIDE",
) -> None:
    """Progressive addition: paired floating boxplots with outliers."""
    n_levels = len(levels)
    x = np.arange(n_levels)
    w = 0.32

    ds_data = [_subj_by_level(df_ds, lev) for lev in levels]
    ab_data = [_subj_by_level(df_abide, lev) for lev in levels]

    n_ds = len(ds_data[0]) if ds_data[0].size else 0
    n_ab = len(ab_data[0]) if ab_data[0].size else 0

    # ds000243 boxes (left)
    bp_ds = ax.boxplot(
        ds_data, positions=x - w / 2 - 0.02, widths=w * 0.85,
        patch_artist=True, showfliers=True,
        flierprops=dict(marker=".", markersize=3,
                        markerfacecolor=C_DS000243,
                        markeredgecolor="none", alpha=0.4),
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color=C_DS000243, linewidth=1.0),
        capprops=dict(color=C_DS000243, linewidth=1.0),
    )
    for patch in bp_ds["boxes"]:
        patch.set_facecolor(C_DS000243)
        patch.set_alpha(0.7)
        patch.set_edgecolor(C_DS000243)

    # ABIDE boxes (right)
    bp_ab = ax.boxplot(
        ab_data, positions=x + w / 2 + 0.02, widths=w * 0.85,
        patch_artist=True, showfliers=True,
        flierprops=dict(marker=".", markersize=3,
                        markerfacecolor=C_ABIDE,
                        markeredgecolor="none", alpha=0.4),
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(color=C_ABIDE, linewidth=1.0),
        capprops=dict(color=C_ABIDE, linewidth=1.0),
    )
    for patch in bp_ab["boxes"]:
        patch.set_facecolor(C_ABIDE)
        patch.set_alpha(0.7)
        patch.set_edgecolor(C_ABIDE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT["tick"] - 0.5)
    ax.set_ylim(bottom=0.5)

    # Shade reliability correction zone
    shade_start = 1.5
    shade_end = n_levels - 0.5
    ax.axvspan(shade_start, shade_end, alpha=0.06, color=C_FULL, zorder=0)
    ax.text((shade_start + shade_end) / 2, 0.97, "Reliability Correction",
            ha="center", va="top", fontsize=FONT["annotation"] - 1.5,
            color=C_FULL, fontstyle="italic",
            transform=ax.get_xaxis_transform())

    ax.legend(
        handles=[
            Patch(facecolor=C_DS000243, alpha=0.7,
                  label=f"{ds_label} (N={n_ds})"),
            Patch(facecolor=C_ABIDE, alpha=0.7,
                  label=f"{ab_label} (N={n_ab})"),
        ],
        loc="lower right", fontsize=FONT["legend"],
    )
    style_axis(ax, "B. Progressive Addition",
               ylabel=r"Estimated Reliability ($\hat{\rho}_T$)")


# ============================================================================
# Panel C: Cross-dataset leave-one-out
# ============================================================================

def _plot_panel_c(
    ax: plt.Axes,
    df_ds: pd.DataFrame,
    df_abide: pd.DataFrame,
    ds_label: str = "ds000243",
    ab_label: str = "ABIDE",
) -> None:
    """Cross-dataset grouped bars with error bars."""
    s_ab = _loo_summary(df_abide)
    s_ds = _loo_summary(df_ds)

    n_ds = df_ds["subject_id"].nunique()
    n_ab = df_abide["subject_id"].nunique()

    rm_ab = s_ab[s_ab["condition"] != "L_full"].sort_values("delta")
    order = rm_ab["condition"].tolist()
    rm_ds = s_ds[s_ds["condition"] != "L_full"].set_index("condition").loc[order]
    rm_ab_idx = rm_ab.set_index("condition").loc[order]

    x = np.arange(len(order))
    w = 0.35

    ax.bar(x - w / 2, rm_ds["delta"].values, width=w,
           color=C_DS000243, alpha=0.85, edgecolor="white", linewidth=0.5,
           label=f"{ds_label} (N={n_ds})", zorder=2)
    ax.errorbar(x - w / 2, rm_ds["delta"].values, yerr=rm_ds["delta_std"].values,
                fmt="none", ecolor="#333333", elinewidth=1.0,
                capsize=3, capthick=0.8, zorder=3)

    ax.bar(x + w / 2, rm_ab_idx["delta"].values, width=w,
           color=C_ABIDE, alpha=0.85, edgecolor="white", linewidth=0.5,
           label=f"{ab_label} (N={n_ab})", zorder=2)
    ax.errorbar(x + w / 2, rm_ab_idx["delta"].values,
                yerr=rm_ab_idx["delta_std"].values,
                fmt="none", ecolor="#333333", elinewidth=1.0,
                capsize=3, capthick=0.8, zorder=3)

    short_labels = [LOO_LABELS[c].replace("w/o ", "−") for c in order]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=FONT["tick"] - 0.5,
                       rotation=20, ha="right")
    ax.axhline(y=0, color="black", linewidth=0.8)

    style_axis(ax, "C. Cross-Dataset Replication",
               ylabel=r"$\Delta$ from Full", legend_loc="lower right")


# ============================================================================
# Panel D: Per-subject distribution
# ============================================================================

def _plot_panel_d(
    ax: plt.Axes,
    df_ds: pd.DataFrame,
    df_abide: pd.DataFrame,
    ds_label: str = "ds000243",
    ab_label: str = "ABIDE",
) -> None:
    """Raw vs Full per-subject violin+box."""
    ds_raw = _subj_by_level(df_ds, "L0")
    ds_full = _subj_by_level(df_ds, "L5")
    ab_raw = _subj_by_level(df_abide, "L0")
    ab_full = _subj_by_level(df_abide, "L5")

    n_ds = len(ds_raw)
    n_ab = len(ab_raw)

    positions = [0, 1, 3, 4]
    data = [ds_raw, ds_full, ab_raw, ab_full]
    colors_vp = [CONDITION_PALETTE["raw"], CONDITION_PALETTE["bsnet"],
                 CONDITION_PALETTE["raw"], CONDITION_PALETTE["bsnet"]]

    vp = ax.violinplot(data, positions=positions, showmedians=False,
                       showextrema=False, widths=0.7)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors_vp[i])
        body.set_alpha(0.45)
        body.set_edgecolor("none")

    bp = ax.boxplot(data, positions=positions, widths=0.25,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors_vp[i])
        patch.set_alpha(0.8)

    ax.set_ylim(bottom=0.5)
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels([f"{ds_label}\n(N={n_ds})", f"{ab_label}\n(N={n_ab})"],
                       fontsize=FONT["tick"])

    ax.legend(
        handles=[
            Patch(facecolor=CONDITION_PALETTE["raw"], alpha=0.6,
                  label="Raw $r_{FC}$"),
            Patch(facecolor=CONDITION_PALETTE["bsnet"], alpha=0.6,
                  label=r"BS-NET $\hat{\rho}_T$"),
        ],
        loc="lower right", fontsize=FONT["legend"],
    )
    style_axis(ax, "D. Per-Subject Distribution",
               ylabel="Estimated Reliability")


# ============================================================================
# Main figure assembly
# ============================================================================

def plot_figure2(
    loo_abide_csv: str | Path,
    loo_ds_csv: str | Path,
    prog_abide_csv: str | Path,
    prog_ds_csv: str | Path,
    abide_npy_dir: str | Path | None = None,
    k_min: float = 3.0,
) -> plt.Figure:
    """Create Figure 2 with ABIDE k-filtering.

    Args:
        loo_abide_csv: ABIDE leave-one-out CSV.
        loo_ds_csv: ds000243 leave-one-out CSV.
        prog_abide_csv: ABIDE progressive ablation CSV.
        prog_ds_csv: ds000243 progressive ablation CSV.
        abide_npy_dir: ABIDE .npy directory for k-factor computation.
        k_min: Minimum k-factor for ABIDE inclusion.

    Returns:
        matplotlib Figure.
    """
    apply_bsnet_theme()

    loo_abide = pd.read_csv(loo_abide_csv)
    loo_ds = pd.read_csv(loo_ds_csv)
    prog_abide = pd.read_csv(prog_abide_csv)
    prog_ds = pd.read_csv(prog_ds_csv)

    # k-filtering for ABIDE
    if abide_npy_dir is not None:
        k_map = compute_abide_k_map(abide_npy_dir, short_samples=60)
        n_before = loo_abide["subject_id"].nunique()
        loo_abide = filter_by_k(loo_abide, k_map, k_min)
        prog_abide = filter_by_k(prog_abide, k_map, k_min)
        n_after = loo_abide["subject_id"].nunique()
        logger.info(f"ABIDE k≥{k_min} filter: {n_before} → {n_after} subjects")

    n_ab = prog_abide["subject_id"].nunique()
    n_ds = prog_ds["subject_id"].nunique()
    ab_label = f"ABIDE k≥{k_min:.0f}" if abide_npy_dir else "ABIDE"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.38, wspace=0.30)

    _plot_panel_a(axes[0, 0], loo_abide, f"{ab_label} N={n_ab}")
    _plot_panel_b(axes[0, 1], prog_ds, prog_abide,
                  PROG_4_LEVELS, PROG_4_LABELS,
                  ds_label="ds000243", ab_label=ab_label)
    _plot_panel_c(axes[1, 0], loo_ds, loo_abide,
                  ds_label="ds000243", ab_label=ab_label)
    _plot_panel_d(axes[1, 1], prog_ds, prog_abide,
                  ds_label="ds000243", ab_label=ab_label)

    return fig


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """Generate Figure 2."""
    parser = argparse.ArgumentParser(description="Figure 2: Component Necessity")
    parser.add_argument("--loo-abide",
                        default="artifacts/reports/component_necessity_ABIDE_cc200_N468.csv")
    parser.add_argument("--loo-ds",
                        default="artifacts/reports/component_necessity_ds000243_cc200_N52.csv")
    parser.add_argument("--prog-abide",
                        default="artifacts/reports/progressive_ablation_abide_cc200_N468.csv")
    parser.add_argument("--prog-ds",
                        default="artifacts/reports/progressive_ablation_ds000243_cc200_N52.csv")
    parser.add_argument("--abide-npy-dir",
                        default="data/abide/timeseries_cache/cc200",
                        help="ABIDE .npy dir for k-factor. Set 'none' to skip.")
    parser.add_argument("--k-min", type=float, default=3.0,
                        help="Min k-factor for ABIDE (default: 3.0).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    npy_dir = None if args.abide_npy_dir == "none" else args.abide_npy_dir

    fig = plot_figure2(
        loo_abide_csv=args.loo_abide,
        loo_ds_csv=args.loo_ds,
        prog_abide_csv=args.prog_abide,
        prog_ds_csv=args.prog_ds,
        abide_npy_dir=npy_dir,
        k_min=args.k_min,
    )

    save_figure(fig, "Fig2_Component_Necessity.png")
    logger.info("Saved: Fig2_Component_Necessity.png")


if __name__ == "__main__":
    main()
