#!/usr/bin/env python3
"""Figure 1: BS-NET Method Overview (A: Pipeline, B: Convergence, C: τ_min).

Combined figure for the main manuscript:
  A. Pipeline schematic — conceptual block diagram
  B. Convergence validation — r_FC(τ_ref) → ρ̂T (3 sub-panels)
  C. τ_min estimation — ρ̂T(τ_short) plateau

Layout:
  Row 1: [────────────── A. Pipeline Schematic ──────────────]
  Row 2: [B1. τ=60s] [B2. τ=120s] [B3. τ=180s] [C. τ_min ]

Data (for B, C):
  data/ds000243/results/ds000243_xcpd_convergence_rfc_4s256parcels.csv
  data/ds000243/results/ds000243_xcpd_convergence_bsnet_4s256parcels.csv

Output: Fig1_Method_Overview.png

Usage:
    cd bsNet/
    python src/visualization/plot_figure1_combined.py
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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

# ── Data paths ──
DEFAULT_DATA_DIR = Path("data/ds000243/results")
RFC_FILENAME = "ds000243_xcpd_convergence_rfc_4s256parcels.csv"
BSNET_FILENAME = "ds000243_xcpd_convergence_bsnet_4s256parcels.csv"

# ── Colors ──
COLOR_RFC = CONDITION_PALETTE["raw"]       # Amber
COLOR_BSNET = CONDITION_PALETTE["bsnet"]   # Blue
COLOR_REF = CONDITION_PALETTE["reference"] # Gray
COLOR_PLATEAU = "#2ecc71"                  # Green

# Pipeline schematic colors
C_INPUT = "#fdae61"     # Amber — input (short scan)
C_STAGE = "#4A90E2"     # Blue — processing stages
C_STAGE_ALT = "#5BA3F5" # Lighter blue — alternate stages
C_OUTPUT = "#2ecc71"    # Green — output (ρ̂T)
C_ARROW = "#555555"     # Arrow color
C_BG = "#F7F9FC"        # Light background for pipeline

# Convergence panels
CONVERGENCE_TAU_SHORTS = [60, 120, 180]
TAU_MIN_PLATEAU = (90, 180)


# ═══════════════════════════════════════════════════════════════════════
# Panel A: Pipeline Schematic
# ═══════════════════════════════════════════════════════════════════════


def _draw_box(
    ax: plt.Axes,
    x: float, y: float, w: float, h: float,
    text: str,
    color: str,
    fontsize: float = 9.0,
    text_color: str = "white",
    bold: bool = True,
    subtext: str = "",
) -> None:
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)

    weight = "bold" if bold else "normal"
    if subtext:
        ax.text(x, y + 0.015, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=text_color, zorder=4)
        ax.text(x, y - 0.025, subtext, ha="center", va="center",
                fontsize=fontsize - 2.0, color=text_color, alpha=0.85, zorder=4)
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=text_color, zorder=4)


def _draw_arrow(ax: plt.Axes, x1: float, x2: float, y: float) -> None:
    """Draw a horizontal arrow between boxes."""
    ax.annotate(
        "",
        xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_ARROW,
            lw=1.8,
            mutation_scale=14,
        ),
        zorder=2,
    )


import matplotlib.image as mpimg

def plot_pipeline_schematic(ax: plt.Axes) -> None:
    """Draw BS-NET pipeline as a block diagram or use high-res illustration.

    Args:
        ax: Matplotlib Axes to draw on.
    """
    ax.axis("off")
    img_path = Path(__file__).resolve().parent / "fig1_panel_a_cropped_edited.png"

    if img_path.exists():
        # Load Nature-grade illustration
        img = mpimg.imread(str(img_path))
        
        # Get actual image shape to prevent distortion (e.g. 360/1024 = 0.35)
        img_h, img_w = img.shape[:2]
        ratio = img_h / img_w 
        
        # Set extent so X goes 0 to 1, and Y goes 0 to ratio. 
        # Matplotlib's default aspect="equal" will now perfectly fit the physical GridSpec.
        ax.imshow(img, extent=[0, 1, 0, ratio], interpolation="bicubic")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, ratio + 0.05)

        # Typography overlay styles
        kw_title = {"fontsize": FONT.get("title", 12), "fontweight": "bold", "color": "#2c3e50", "ha": "center", "va": "center", "zorder": 5}
        kw_sub = {"fontsize": FONT.get("annotation", 7) + 1, "color": "#606060", "style": "italic", "ha": "center", "va": "center", "zorder": 5}
        bbox = dict(boxstyle="round,pad=0.3", facecolor=(1, 1, 1, 0.85), edgecolor="none") # Semi-transparent white background

        # =========================================================
        # [TEXT ALIGNMENT CONFIGURATION]
        # =========================================================
        y_upper = ratio - 0.02
        y_lower = 0.02  # Precision tweak: perfect floating margin
        
        # 1. Left Brain (Input)
        pos_input = 0.12
        ax.text(pos_input, y_upper, "Short Scan FC (2 min)\n" + r"$r_{\mathrm{short}}$", bbox=bbox, **kw_sub)
        
        # 2. Middle-Left (Matrices)
        pos_matrix = 0.38
        lw_eq = r"$\Sigma^* = (1-\lambda)\hat{\Sigma} + \lambda T$"
        ax.text(pos_matrix, y_upper, "Ledoit-Wolf Shrinkage\n" + lw_eq, bbox=bbox, **kw_sub)
        
        # 3. Middle-Right (Bootstrap stacks)
        pos_stack = 0.62
        bs_eq = r"$\mathcal{B} = \{X_1^*, \dots, X_B^*\}$"
        ax.text(pos_stack, y_upper, "Block Bootstrap\n" + bs_eq, bbox=bbox, **kw_sub)

        # 4. Right Brain (Output Label)
        pos_output = 0.88
        kw_out = dict(kw_sub)
        kw_out["color"] = "#27ae60"
        ax.text(pos_output, y_upper + 0.01, "Reliability-Adjusted\n" + r"$\hat{\rho}_T$ Matrix", bbox=bbox, **kw_out)

        # ── Lower Mathematical Flow (Spearman-Brown -> Prior -> Fisher z) ──
        # Box 1: Spearman-Brown (Aligned exactly under Ledoit-Wolf)
        pos_sb = 0.38
        sb_eq = r"$\tilde{\rho} = \frac{k \cdot r_b}{1 + (k-1)r_b}$"
        ax.text(pos_sb, y_lower, "Spearman-Brown\n" + sb_eq, bbox=bbox, **kw_sub)
        
        # Arrow 1 (Midway between 0.38 and 0.64)
        arrprops = dict(arrowstyle="-|>", color="#7f8c8d", lw=1.2, mutation_scale=10)
        ax.annotate("", xy=(0.53, y_lower), xytext=(0.49, y_lower), arrowprops=arrprops, zorder=4)

        # Box 2: Bayesian Prior
        pos_prior = 0.64
        prior_eq = r"$\rho_{post} \propto P(D|\tilde{\rho})P(\tilde{\rho})$"
        ax.text(pos_prior, y_lower, "Bayesian Prior\n" + prior_eq, bbox=bbox, **kw_sub)

        # Arrow 2 (Midway between 0.64 and 0.90)
        ax.annotate("", xy=(0.79, y_lower), xytext=(0.75, y_lower), arrowprops=arrprops, zorder=4)

        # Box 3: Fisher z Attenuation (Ends directly beneath green brain output)
        pos_fisher = 0.90
        fisher_eq = r"$\hat{\rho} = \tanh( c \cdot \tanh^{-1}\rho_{post} )$" 
        ax.text(pos_fisher, y_lower, "Fisher $z$ Attenuation\n" + fisher_eq, bbox=bbox, **kw_sub)
        
        return

    # ── Fallback (Box Diagram) ── 
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.12, 0.12)
    ax.set_aspect("equal")

    # Background
    bg = FancyBboxPatch(
        (-0.03, -0.09), 1.06, 0.18,
        boxstyle="round,pad=0.01",
        facecolor=C_BG,
        edgecolor="#ddd",
        linewidth=0.8,
        zorder=0,
    )
    ax.add_patch(bg)

    # Box dimensions
    bw, bh = 0.130, 0.075
    y0 = 0.0

    # Pipeline stages: positions
    stages = [
        (0.06,  "Short Scan\nFC (2 min)",  C_INPUT,    "#333",   ""),
        (0.22,  "Ledoit-Wolf\nShrinkage",   C_STAGE,    "white",  "Regularization"),
        (0.39,  "Block\nBootstrap",          C_STAGE,    "white",  "Resampling"),
        (0.56,  "Spearman-\nBrown",          C_STAGE_ALT,"white",  "Extrapolation"),
        (0.73,  "Bayesian\nPrior",           C_STAGE_ALT,"white",  "Stabilization"),
        (0.895, "ρ̂T",                       C_OUTPUT,   "white",  "Reliability"),
    ]

    for i, (x, label, color, tc, sub) in enumerate(stages):
        fs = 9.5 if i < 5 else 14.0
        _draw_box(ax, x, y0, bw, bh, label, color,
                  fontsize=fs, text_color=tc, subtext=sub)
        if i > 0:
            prev_x = stages[i - 1][0]
            _draw_arrow(ax, prev_x + bw / 2 + 0.005, x - bw / 2 - 0.005, y0)

    # Attenuation correction annotation (below the arrow between Prior and ρ̂T)
    mid_x = (stages[4][0] + stages[5][0]) / 2
    ax.text(mid_x, y0 - 0.055, "Fisher z\nAttenuation\nCorrection",
            ha="center", va="top", fontsize=7.0, color="#666",
            style="italic", zorder=4)
    # Small upward arrow
    ax.annotate("", xy=(mid_x, y0 - 0.033), xytext=(mid_x, y0 - 0.048),
                arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.0,
                                mutation_scale=8), zorder=2)


# ═══════════════════════════════════════════════════════════════════════
# Panel B: Convergence Validation (reuse logic from standalone script)
# ═══════════════════════════════════════════════════════════════════════


def _compute_rfc_stats(
    rfc_df: pd.DataFrame, tau_short: int, min_subjects: int = 20,
) -> pd.DataFrame:
    sub = rfc_df[rfc_df["tau_short_sec"] == tau_short]
    stats = (
        sub.groupby("tau_ref_sec")["r_fc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return stats[stats["count"] >= min_subjects]


def _compute_bsnet_stats(
    bsnet_df: pd.DataFrame, tau_short: int,
) -> dict[str, float]:
    sub = bsnet_df[bsnet_df["tau_short_sec"] == tau_short]
    per_subj = sub.groupby("sub_id")["rho_hat_T"].mean()
    return {"mean": per_subj.mean(), "std": per_subj.std(), "n": len(per_subj)}


def plot_convergence_panel(
    ax: plt.Axes,
    rfc_df: pd.DataFrame,
    bsnet_df: pd.DataFrame,
    tau_short: int,
    panel_label: str,
    show_ylabel: bool = True,
) -> None:
    """Draw a single convergence sub-panel.

    Args:
        ax: Axes to draw on.
        rfc_df: r_FC DataFrame.
        bsnet_df: BS-NET DataFrame.
        tau_short: Fixed short duration (seconds).
        panel_label: e.g., "B1"
        show_ylabel: Whether to show y-axis label.
    """
    rfc_stats = _compute_rfc_stats(rfc_df, tau_short)
    tau_refs = rfc_stats["tau_ref_sec"].values
    rfc_mean = rfc_stats["mean"].values
    rfc_std = rfc_stats["std"].values

    bsnet_stats = _compute_bsnet_stats(bsnet_df, tau_short)
    rho_mean = bsnet_stats["mean"]
    rho_std = bsnet_stats["std"]

    # Per-subject spaghetti
    sub_data = rfc_df[rfc_df["tau_short_sec"] == tau_short]
    for sid, grp in sub_data.groupby("sub_id"):
        valid = grp[grp["tau_ref_sec"].isin(tau_refs)].sort_values("tau_ref_sec")
        if len(valid) > 0:
            ax.plot(valid["tau_ref_sec"], valid["r_fc"],
                    color=COLOR_RFC, alpha=0.07, linewidth=0.6, zorder=1)

    # ρ̂T band
    ax.axhspan(rho_mean - rho_std, rho_mean + rho_std,
               color=COLOR_BSNET, alpha=0.15, zorder=2)
    ax.axhline(rho_mean, color=COLOR_BSNET, linewidth=LINE["reference"],
               linestyle="--", zorder=3,
               label=f"ρ̂T = {rho_mean:.3f}")

    # r_FC mean ± SD
    ax.fill_between(tau_refs, rfc_mean - rfc_std, rfc_mean + rfc_std,
                    color=COLOR_RFC, alpha=0.20, zorder=4)
    ax.plot(tau_refs, rfc_mean, color=COLOR_RFC, linewidth=LINE["secondary"],
            marker="o", markersize=4, zorder=5,
            label=f"r_FC (N={rfc_stats['count'].iloc[0]})")

    # Gap
    gap = rho_mean - rfc_mean.max()
    ax.annotate(f"gap = {gap:.3f}",
                xy=(tau_refs[np.argmax(rfc_mean)], rfc_mean.max()),
                xytext=(0, 18), textcoords="offset points",
                fontsize=FONT["annotation"] - 1, color="#555", ha="center",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))

    tau_min_str = f"{tau_short // 60}min" if tau_short >= 60 else f"{tau_short}s"
    ax.set_title(f"{panel_label}. τ_short = {tau_short}s ({tau_min_str})",
                 fontsize=FONT["title"] - 2, fontweight="bold",
                 pad=FONT["title_pad"])
    ax.set_xlabel("Reference duration τ_ref (s)", fontsize=FONT["axis_label"])
    if show_ylabel:
        ax.set_ylabel("FC similarity", fontsize=FONT["axis_label"])
    ax.tick_params(labelsize=FONT["tick"])
    ax.set_ylim(0.2, 0.9)
    ax.set_xlim(0, max(tau_refs) + 30)
    ax.legend(fontsize=FONT["legend"] - 1, loc="lower right")


# ═══════════════════════════════════════════════════════════════════════
# Panel C: τ_min Estimation
# ═══════════════════════════════════════════════════════════════════════


def plot_tau_min_panel(
    ax: plt.Axes,
    bsnet_df: pd.DataFrame,
) -> None:
    """Draw τ_min estimation as a panel within the combined figure.

    Args:
        ax: Axes to draw on.
        bsnet_df: BS-NET DataFrame.
    """
    tau_shorts = sorted(bsnet_df["tau_short_sec"].unique())
    means, stds, seed_sds = [], [], []
    for ts in tau_shorts:
        sub = bsnet_df[bsnet_df["tau_short_sec"] == ts]
        per_subj = sub.groupby("sub_id")["rho_hat_T"].mean()
        means.append(per_subj.mean())
        stds.append(per_subj.std())
        seed_sds.append(sub.groupby("sub_id")["rho_hat_T"].std().mean())

    means = np.array(means)
    stds = np.array(stds)
    seed_sds = np.array(seed_sds)
    tau_shorts = np.array(tau_shorts)

    # Plateau highlight
    ax.axvspan(TAU_MIN_PLATEAU[0], TAU_MIN_PLATEAU[1],
               color=COLOR_PLATEAU, alpha=0.10,
               label=f"Plateau [{TAU_MIN_PLATEAU[0]}–{TAU_MIN_PLATEAU[1]}s]")

    # Reference artifact zone
    ax.axvspan(180, max(tau_shorts) + 10, color="#e74c3c", alpha=0.05)

    # ρ̂T curve
    ax.fill_between(tau_shorts, means - stds, means + stds,
                    color=COLOR_BSNET, alpha=0.20)
    ax.plot(tau_shorts, means, color=COLOR_BSNET, linewidth=LINE["main"],
            marker="o", markersize=MARKER["secondary"], zorder=5,
            label="ρ̂T (N=49, 10 seeds)")

    # Seed SD on twin axis
    ax_seed = ax.twinx()
    ax_seed.plot(tau_shorts, seed_sds, color="#7BC8A4", linewidth=LINE["secondary"],
                 linestyle="--", marker="s", markersize=MARKER["small"], alpha=0.7,
                 label="Seed SD")
    ax_seed.set_ylabel("Seed SD", fontsize=FONT["axis_label"],
                       color="#7BC8A4")
    ax_seed.tick_params(axis="y", labelcolor="#7BC8A4",
                        labelsize=FONT["tick"])
    ax_seed.set_ylim(0, max(seed_sds) * 3)

    # Extend y-axis for summary box
    ax.set_ylim(None, max(means + stds) + 0.05)

    # Summary text — positioned in plateau zone
    plateau_mask = (tau_shorts >= TAU_MIN_PLATEAU[0]) & (tau_shorts <= TAU_MIN_PLATEAU[1])
    plateau_mean = means[plateau_mask].mean()
    plateau_range = means[plateau_mask].max() - means[plateau_mask].min()
    summary = (
        f"Plateau [{TAU_MIN_PLATEAU[0]}–{TAU_MIN_PLATEAU[1]}s]: "
        f"ρ̂T = {plateau_mean:.3f}, range = {plateau_range:.4f}\n"
        f"Recommended: 60–120s (>97% of plateau)  |  "
        f"Seed SD: {seed_sds[plateau_mask].mean():.4f}"
    )
    ax.text(0.02, 0.96, summary, transform=ax.transAxes,
            fontsize=FONT["annotation"] - 1, 
            verticalalignment="top", horizontalalignment="left", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#ccc", alpha=0.9))

    # Artifact zone label
    ax.text(320, means.min() - 0.005, "Reference length\nartifact zone",
            fontsize=FONT["annotation"] - 1, color="#c0392b",
            ha="center", style="italic")

    ax.set_xlabel("Short scan duration τ_short (s)", fontsize=FONT["axis_label"])
    ax.set_ylabel("ρ̂T (mean ± SD across subjects)", fontsize=FONT["axis_label"])
    ax.tick_params(labelsize=FONT["tick"])

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_seed.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=FONT["legend"], loc="upper right")


# ═══════════════════════════════════════════════════════════════════════
# Combined Figure
# ═══════════════════════════════════════════════════════════════════════


def plot_figure1(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_name: str = "Fig1_Method_Overview.png",
) -> None:
    """Create the combined Figure 1.

    Args:
        data_dir: Directory with convergence CSVs.
        output_name: Output filename.
    """
    apply_bsnet_theme()

    # Load data
    rfc_df = pd.read_csv(data_dir / RFC_FILENAME)
    bsnet_df = pd.read_csv(data_dir / BSNET_FILENAME)
    logger.info(f"Loaded r_FC: {len(rfc_df)}, BS-NET: {len(bsnet_df)} rows")

    # ── Layout: 3-row stacked using independent GridSpecs for absolute gap control ──
    fig = plt.figure(figsize=(16, 16))
    
    # We define exact top/bottom boundaries (0.0 to 1.0) so you can directly control 
    # the gaps without Matplotlib forcing uniform HSPACE spacing.
    # 1. Panel A (height ~ 0.235)
    gs_a = gridspec.GridSpec(1, 1, figure=fig, left=0.05, right=0.95, top=0.96, bottom=0.715)
    
    # GAP A-B : 0.725 - 0.695 = 0.03 (Very tight, since Panel A has internal padding)
    
    # 2. Panel B (height ~ 0.230 - 4/5 size of previous)
    # wspace=0.14 makes elements tightly grouped
    gs_b = gridspec.GridSpec(1, 3, figure=fig, left=0.05, right=0.95, top=0.695, bottom=0.455, wspace=0.14)
    
    # GAP B-C : 0.465 - 0.365 = 0.10 (Wider, because Panel B has X-axis labels eating into space)
    
    # 3. Panel C (height ~ 0.314)
    # width_ratios=[1, 3, 1] centers the panel to occupy exactly 3/5 width
    gs_c = gridspec.GridSpec(1, 3, figure=fig, left=0.05, right=0.95, top=0.365, bottom=0.051, width_ratios=[1, 3, 1])

    # Row 1: A. Pipeline schematic
    ax_pipeline = fig.add_subplot(gs_a[0, 0])
    ax_pipeline.set_title("A. BS-NET Pipeline",
                          fontsize=FONT["title"],
                          fontweight="bold", pad=10, loc="left")
    plot_pipeline_schematic(ax_pipeline)

    # Row 2: B1, B2, B3 — Convergence Validation
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1], sharey=ax_b1)
    ax_b3 = fig.add_subplot(gs_b[0, 2], sharey=ax_b1)

    plot_convergence_panel(ax_b1, rfc_df, bsnet_df, 60, "B1", show_ylabel=True)
    plot_convergence_panel(ax_b2, rfc_df, bsnet_df, 120, "B2", show_ylabel=False)
    plot_convergence_panel(ax_b3, rfc_df, bsnet_df, 180, "B3", show_ylabel=False)
    plt.setp(ax_b2.get_yticklabels(), visible=False)
    plt.setp(ax_b3.get_yticklabels(), visible=False)

    # Non-stationarity note below Row 2 (Placed relatively to B2 to avoid absolute coord overlap)
    ax_b2.text(0.50, -0.15,
               "Note: r_FC decline at long τ_ref reflects temporal "
               "non-stationarity of resting-state FC.",
               transform=ax_b2.transAxes,
               ha="center", fontsize=FONT["annotation"] - 1.5,
               color="#888", style="italic")

    # Row 3: C. τ_min Estimation
    ax_c = fig.add_subplot(gs_c[0, 1])
    ax_c.set_title("C. τ_min Estimation",
                          fontsize=FONT["title"],
                          fontweight="bold", pad=10, loc="left")
    plot_tau_min_panel(ax_c, bsnet_df)

    save_figure(fig, output_name)
    logger.info(f"Saved: {output_name}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 1: BS-NET Method Overview")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", default="Fig1_Method_Overview.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plot_figure1(args.data_dir, args.output)


if __name__ == "__main__":
    main()
