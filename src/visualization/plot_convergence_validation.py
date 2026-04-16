#!/usr/bin/env python3
"""Convergence Validation Figure: r_FC(τ_ref) → ρ̂T as reference grows.

Two-figure output:
  Figure A — Convergence (3-panel): τ_short = 60, 120, 180 s
    X = τ_ref (reference duration, seconds)
    Y = r_FC (observed FC similarity)
    Horizontal band = ρ̂T mean ± SD (BS-NET predicted ceiling)
    Individual subject thin lines + mean ± SD shaded band

  Figure B — τ_min Estimation (single panel):
    X = τ_short (seconds)
    Y = ρ̂T (mean ± SD across subjects)
    Plateau region highlighted, 95% peak threshold marked
    Seed SD as secondary axis or annotation

Data:
  data/ds000243/results/ds000243_xcpd_convergence_rfc_4s256parcels.csv
  data/ds000243/results/ds000243_xcpd_convergence_bsnet_4s256parcels.csv

Output:
  Fig_Convergence_Validation.png  (3-panel convergence)
  Fig_Tau_Min_Estimation.png      (τ_min plateau)

Usage:
    python src/visualization/plot_convergence_validation.py
    python src/visualization/plot_convergence_validation.py --data-dir data/ds000243/results
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

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

# ── Defaults ──
DEFAULT_DATA_DIR = Path("data/ds000243/results")
RFC_FILENAME = "ds000243_xcpd_convergence_rfc_4s256parcels.csv"
BSNET_FILENAME = "ds000243_xcpd_convergence_bsnet_4s256parcels.csv"

# Convergence panels: τ_short values to plot
CONVERGENCE_TAU_SHORTS = [60, 120, 180]

# Colors
COLOR_RFC = CONDITION_PALETTE["raw"]       # Amber — observed r_FC
COLOR_BSNET = CONDITION_PALETTE["bsnet"]   # Blue — ρ̂T ceiling
COLOR_REF = CONDITION_PALETTE["reference"] # Gray — reference band

# τ_min figure
COLOR_PLATEAU = "#2ecc71"  # Green highlight for plateau region
TAU_MIN_PLATEAU = (90, 180)  # seconds — empirical plateau range
PEAK_FRAC = 0.95  # 95% of peak ρ̂T for τ_min threshold


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════


def load_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load r_FC and BS-NET convergence CSVs.

    Args:
        data_dir: Directory containing the CSV files.

    Returns:
        Tuple of (rfc_df, bsnet_df) DataFrames.
    """
    rfc_path = data_dir / RFC_FILENAME
    bsnet_path = data_dir / BSNET_FILENAME

    rfc_df = pd.read_csv(rfc_path)
    bsnet_df = pd.read_csv(bsnet_path)

    logger.info(
        f"Loaded r_FC: {len(rfc_df)} rows, "
        f"BS-NET: {len(bsnet_df)} rows"
    )
    return rfc_df, bsnet_df


def compute_rfc_stats(
    rfc_df: pd.DataFrame,
    tau_short: int,
    min_subjects: int = 20,
) -> pd.DataFrame:
    """Compute mean ± SD of r_FC per τ_ref for a given τ_short.

    Only includes τ_ref points where at least min_subjects have data.

    Args:
        rfc_df: r_FC DataFrame.
        tau_short: Fixed short duration (seconds).
        min_subjects: Minimum N for a τ_ref point to be included.

    Returns:
        DataFrame with columns: tau_ref_sec, mean, std, count.
    """
    sub = rfc_df[rfc_df["tau_short_sec"] == tau_short].copy()
    stats = (
        sub.groupby("tau_ref_sec")["r_fc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats = stats[stats["count"] >= min_subjects]
    return stats


def compute_bsnet_stats(
    bsnet_df: pd.DataFrame,
    tau_short: int,
) -> dict[str, float]:
    """Compute ρ̂T mean ± SD across subjects (averaging seeds per subject first).

    Args:
        bsnet_df: BS-NET DataFrame.
        tau_short: Fixed short duration (seconds).

    Returns:
        Dict with keys: mean, std, ci_lower_mean, ci_upper_mean.
    """
    sub = bsnet_df[bsnet_df["tau_short_sec"] == tau_short].copy()
    per_subject = sub.groupby("sub_id")["rho_hat_T"].mean()
    return {
        "mean": per_subject.mean(),
        "std": per_subject.std(),
        "n": len(per_subject),
    }


def get_per_subject_rfc(
    rfc_df: pd.DataFrame,
    tau_short: int,
) -> dict[str, pd.DataFrame]:
    """Get per-subject r_FC(τ_ref) series for spaghetti lines.

    Args:
        rfc_df: r_FC DataFrame.
        tau_short: Fixed short duration (seconds).

    Returns:
        Dict mapping sub_id → DataFrame with tau_ref_sec, r_fc columns.
    """
    sub = rfc_df[rfc_df["tau_short_sec"] == tau_short].copy()
    result = {}
    for sid, grp in sub.groupby("sub_id"):
        result[sid] = grp[["tau_ref_sec", "r_fc"]].sort_values("tau_ref_sec")
    return result


# ═══════════════════════════════════════════════════════════════════════
# Figure A: Convergence Validation (3-panel)
# ═══════════════════════════════════════════════════════════════════════


def plot_convergence_figure(
    rfc_df: pd.DataFrame,
    bsnet_df: pd.DataFrame,
    output_name: str = "Fig_Convergence_Validation.png",
) -> None:
    """Create 3-panel convergence validation figure.

    Each panel fixes τ_short and shows r_FC(τ_ref) approaching ρ̂T.

    Args:
        rfc_df: r_FC DataFrame.
        bsnet_df: BS-NET DataFrame.
        output_name: Output filename.
    """
    apply_bsnet_theme()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 5.5),
        sharey=True,
    )

    panel_labels = ["A", "B", "C"]

    for idx, (ax, tau_s) in enumerate(
        zip(axes, CONVERGENCE_TAU_SHORTS)
    ):
        # --- r_FC stats ---
        rfc_stats = compute_rfc_stats(rfc_df, tau_s, min_subjects=20)
        tau_refs = rfc_stats["tau_ref_sec"].values
        rfc_mean = rfc_stats["mean"].values
        rfc_std = rfc_stats["std"].values

        # --- ρ̂T ceiling ---
        bsnet_stats = compute_bsnet_stats(bsnet_df, tau_s)
        rho_mean = bsnet_stats["mean"]
        rho_std = bsnet_stats["std"]

        # --- Per-subject spaghetti lines ---
        per_sub = get_per_subject_rfc(rfc_df, tau_s)
        for sid, sub_df in per_sub.items():
            # Only plot points that fall in the valid τ_ref range
            valid = sub_df[sub_df["tau_ref_sec"].isin(tau_refs)]
            if len(valid) > 0:
                ax.plot(
                    valid["tau_ref_sec"],
                    valid["r_fc"],
                    color=COLOR_RFC,
                    alpha=0.08,
                    linewidth=LINE["individual"],
                    zorder=1,
                )

        # --- ρ̂T horizontal band (±1 SD) ---
        ax.axhspan(
            rho_mean - rho_std,
            rho_mean + rho_std,
            color=COLOR_BSNET,
            alpha=0.15,
            zorder=2,
        )
        ax.axhline(
            rho_mean,
            color=COLOR_BSNET,
            linewidth=LINE["reference"],
            linestyle="--",
            zorder=3,
            label=f"ρ̂T = {rho_mean:.3f} ± {rho_std:.3f}",
        )

        # --- r_FC mean ± SD band ---
        ax.fill_between(
            tau_refs,
            rfc_mean - rfc_std,
            rfc_mean + rfc_std,
            color=COLOR_RFC,
            alpha=0.20,
            zorder=4,
        )
        ax.plot(
            tau_refs,
            rfc_mean,
            color=COLOR_RFC,
            linewidth=LINE["main"],
            marker="o",
            markersize=MARKER["secondary"],
            zorder=5,
            label=f"r_FC (N={rfc_stats['count'].iloc[0]})",
        )

        # --- Gap annotation ---
        max_rfc_at_peak = rfc_mean.max()
        gap = rho_mean - max_rfc_at_peak
        ax.annotate(
            f"gap = {gap:.3f}",
            xy=(tau_refs[np.argmax(rfc_mean)], max_rfc_at_peak),
            xytext=(0, 20),
            textcoords="offset points",
            fontsize=FONT["annotation"],
            color="#555555",
            ha="center",
            arrowprops=dict(
                arrowstyle="->",
                color="#999999",
                lw=1.0,
            ),
        )

        # --- Panel label and styling ---
        tau_min_str = f"{tau_s // 60}min" if tau_s >= 60 else f"{tau_s}s"
        style_axis(
            ax,
            title=f"{panel_labels[idx]}. τ_short = {tau_s}s ({tau_min_str})",
            xlabel="Reference duration τ_ref (s)",
            ylabel="FC similarity" if idx == 0 else "",
            legend_loc="lower right",
            legend_fontsize=FONT["legend"],
        )

        ax.set_ylim(0.15, 0.90)
        ax.set_xlim(0, max(tau_refs) + 30)

    # --- Annotation: non-stationarity note ---
    fig.text(
        0.5, -0.02,
        "Note: r_FC decline at long τ_ref reflects temporal non-stationarity "
        "— reference segment extends far from short segment.",
        ha="center",
        fontsize=FONT["annotation"] - 1,
        color="#777777",
        style="italic",
    )

    fig.tight_layout(pad=3.0)
    save_figure(fig, output_name)
    logger.info(f"Saved convergence figure: {output_name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure B: τ_min Estimation
# ═══════════════════════════════════════════════════════════════════════


def plot_tau_min_figure(
    bsnet_df: pd.DataFrame,
    output_name: str = "Fig_Tau_Min_Estimation.png",
) -> None:
    """Create τ_min estimation figure: ρ̂T(τ_short) plateau analysis.

    X-axis: τ_short (seconds)
    Y-axis: ρ̂T (mean ± SD across subjects)
    Highlights plateau region and 95% peak threshold.

    Args:
        bsnet_df: BS-NET DataFrame.
        output_name: Output filename.
    """
    apply_bsnet_theme()

    # Compute ρ̂T stats per τ_short
    tau_shorts = sorted(bsnet_df["tau_short_sec"].unique())
    means = []
    stds = []
    seed_sds = []
    for ts in tau_shorts:
        sub = bsnet_df[bsnet_df["tau_short_sec"] == ts]
        per_subj = sub.groupby("sub_id")["rho_hat_T"].mean()
        per_subj_seed_sd = sub.groupby("sub_id")["rho_hat_T"].std()
        means.append(per_subj.mean())
        stds.append(per_subj.std())
        seed_sds.append(per_subj_seed_sd.mean())

    means = np.array(means)
    stds = np.array(stds)
    seed_sds = np.array(seed_sds)
    tau_shorts = np.array(tau_shorts)

    # ── Figure layout: main + seed SD inset ──
    fig, ax_main = plt.subplots(1, 1, figsize=(10, 6))

    # --- Plateau region highlight ---
    ax_main.axvspan(
        TAU_MIN_PLATEAU[0],
        TAU_MIN_PLATEAU[1],
        color=COLOR_PLATEAU,
        alpha=0.10,
        label=f"Plateau [{TAU_MIN_PLATEAU[0]}–{TAU_MIN_PLATEAU[1]}s]",
    )

    # --- ρ̂T(τ_short) curve ---
    ax_main.fill_between(
        tau_shorts,
        means - stds,
        means + stds,
        color=COLOR_BSNET,
        alpha=0.20,
    )
    ax_main.plot(
        tau_shorts,
        means,
        color=COLOR_BSNET,
        linewidth=LINE["main"],
        marker="o",
        markersize=MARKER["secondary"],
        label=f"ρ̂T (N=49, 10 seeds)",
        zorder=5,
    )

    # --- Annotation: reference artifact zone ---
    # τ_short > 180s shows ρ̂T drop due to insufficient reference
    ax_main.axvspan(
        180,
        max(tau_shorts) + 10,
        color="#e74c3c",
        alpha=0.05,
    )
    ax_main.text(
        300,
        means.min() - 0.01,
        "Reference length\nartifact zone",
        fontsize=FONT["annotation"] - 1,
        color="#c0392b",
        ha="center",
        style="italic",
    )

    # --- Seed SD annotation as secondary line ---
    ax_seed = ax_main.twinx()
    ax_seed.plot(
        tau_shorts,
        seed_sds,
        color="#7BC8A4",
        linewidth=LINE["secondary"],
        linestyle="--",
        marker="s",
        markersize=MARKER["small"],
        alpha=0.7,
        label="Seed SD",
    )
    ax_seed.set_ylabel(
        "Seed SD",
        fontsize=FONT["axis_label"],
        color="#7BC8A4",
    )
    ax_seed.tick_params(axis="y", labelcolor="#7BC8A4", labelsize=FONT["tick"])
    ax_seed.set_ylim(0, max(seed_sds) * 3)

    # --- Axis limits: extend upper range for legend/summary space ---
    ax_main.set_ylim(None, max(means + stds) + 0.06)

    # --- Styling ---
    style_axis(
        ax_main,
        title="τ_min Estimation: ρ̂T(τ_short) Plateau",
        xlabel="Short scan duration τ_short (s)",
        ylabel="ρ̂T (mean ± SD across subjects)",
    )

    # Combine legends from both axes — upper right
    lines1, labels1 = ax_main.get_legend_handles_labels()
    lines2, labels2 = ax_seed.get_legend_handles_labels()
    ax_main.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=FONT["legend"],
    )

    # Summary text box — plateau-centric, positioned in green zone (upper left)
    plateau_mask = (tau_shorts >= TAU_MIN_PLATEAU[0]) & (tau_shorts <= TAU_MIN_PLATEAU[1])
    plateau_mean = means[plateau_mask].mean()
    plateau_range = means[plateau_mask].max() - means[plateau_mask].min()
    summary = (
        f"Plateau [{TAU_MIN_PLATEAU[0]}–{TAU_MIN_PLATEAU[1]}s]: "
        f"ρ̂T = {plateau_mean:.3f}, range = {plateau_range:.4f}\n"
        f"Recommended: 60–120s (>97% of plateau mean)\n"
        f"Seed SD at plateau: {seed_sds[plateau_mask].mean():.4f}"
    )
    ax_main.text(
        0.02,
        0.97,
        summary,
        transform=ax_main.transAxes,
        fontsize=FONT["annotation"] - 0.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    fig.tight_layout(pad=3.0)
    save_figure(fig, output_name)
    logger.info(f"Saved τ_min figure: {output_name}")


# ═══════════════════════════════════════════════════════════════════════
# Summary Statistics (console output)
# ═══════════════════════════════════════════════════════════════════════


def print_convergence_summary(
    rfc_df: pd.DataFrame,
    bsnet_df: pd.DataFrame,
) -> None:
    """Print key convergence statistics to console.

    Args:
        rfc_df: r_FC DataFrame.
        bsnet_df: BS-NET DataFrame.
    """
    print("\n" + "=" * 70)
    print("CONVERGENCE VALIDATION SUMMARY")
    print("=" * 70)

    for ts in CONVERGENCE_TAU_SHORTS:
        rfc_stats = compute_rfc_stats(rfc_df, ts, min_subjects=20)
        bsnet_stats = compute_bsnet_stats(bsnet_df, ts)

        rfc_peak = rfc_stats["mean"].max()
        rho = bsnet_stats["mean"]
        gap = rho - rfc_peak

        # Per-subject gap at max valid τ_ref
        max_ref = rfc_stats["tau_ref_sec"].iloc[-1]
        rfc_sub = rfc_df[
            (rfc_df["tau_short_sec"] == ts)
            & (rfc_df["tau_ref_sec"] == max_ref)
        ]
        bsnet_sub = bsnet_df[bsnet_df["tau_short_sec"] == ts]
        per_subj_rho = bsnet_sub.groupby("sub_id")["rho_hat_T"].mean()
        per_subj_rfc = rfc_sub.set_index("sub_id")["r_fc"]
        common = per_subj_rho.index.intersection(per_subj_rfc.index)
        per_subj_gap = per_subj_rho[common] - per_subj_rfc[common]
        n_positive = (per_subj_gap > 0).sum()

        print(f"\n  τ_short = {ts}s:")
        print(f"    ρ̂T = {rho:.4f} ± {bsnet_stats['std']:.4f}")
        print(f"    r_FC peak = {rfc_peak:.4f} (at τ_ref={rfc_stats.loc[rfc_stats['mean'].idxmax(), 'tau_ref_sec']}s)")
        print(f"    Gap (ρ̂T − r_FC_peak) = {gap:.4f}")
        print(f"    Per-subject gap at τ_ref={max_ref}s: "
              f"mean={per_subj_gap.mean():.4f}, "
              f"{n_positive}/{len(per_subj_gap)} positive")

    # τ_min summary
    print("\n" + "-" * 70)
    print("τ_min ESTIMATION")
    print("-" * 70)
    tau_shorts_all = sorted(bsnet_df["tau_short_sec"].unique())
    for ts in tau_shorts_all:
        sub = bsnet_df[bsnet_df["tau_short_sec"] == ts]
        per_subj = sub.groupby("sub_id")["rho_hat_T"].mean()
        seed_sd = sub.groupby("sub_id")["rho_hat_T"].std().mean()
        print(f"  τ_short={ts:4d}s: ρ̂T={per_subj.mean():.4f}±{per_subj.std():.4f}, "
              f"seed_SD={seed_sd:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for convergence validation plotting."""
    parser = argparse.ArgumentParser(
        description="Plot convergence validation figures for BS-NET"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing convergence CSV files",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary statistics without generating figures",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rfc_df, bsnet_df = load_data(args.data_dir)
    print_convergence_summary(rfc_df, bsnet_df)

    if not args.summary_only:
        plot_convergence_figure(rfc_df, bsnet_df)
        plot_tau_min_figure(bsnet_df)


if __name__ == "__main__":
    main()
