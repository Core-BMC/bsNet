"""Generate Figure 6 series: ADHD-200 validation (single/multi-seed + atlas comparison).

Produces publication-quality figures matching Figure 1 style standards:
  - Figure6_ADHD_SingleSeed_{atlas}.png  (2x2: scatter, group improvement, distribution, CI)
  - Figure6_ADHD_MultiSeed_{atlas}.png   (2x2: scatter±err, group box, improvement, stability)
  - Figure6_ADHD_Atlas_Comparison.png    (2x2: ρ̂T dist, improvement, paired scatter, summary)

Data source:
  - data/adhd/results/adhd_bsnet_{atlas}.csv
  - data/adhd/results/adhd_multiseed_{atlas}_10seeds.csv
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import (
    ATLAS_PALETTE,
    FIGSIZE,
    FONT,
    GROUP_PALETTE,
    LINE,
    MARKER,
    PALETTE,
    apply_bsnet_theme,
    save_figure,
    style_axis,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/adhd/results")


# ============================================================================
# Data loaders
# ============================================================================

def load_single_seed(csv_path: str | Path) -> dict:
    """Load ADHD single-seed results CSV."""
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    return {
        "sub_idx": [int(r["sub_idx"]) for r in rows],
        "r_fc_raw": np.array([float(r["r_fc_raw"]) for r in rows]),
        "rho_hat_T": np.array([float(r["rho_hat_T"]) for r in rows]),
        "ci_lower": np.array([float(r["ci_lower"]) for r in rows]),
        "ci_upper": np.array([float(r["ci_upper"]) for r in rows]),
        "improvement": np.array([float(r["improvement"]) for r in rows]),
        "group": [r.get("group", "unknown") for r in rows],
        "n_rois": int(rows[0].get("n_rois", 0)),
        "atlas": rows[0].get("atlas", "unknown"),
        "n_subjects": len(rows),
    }


def load_multi_seed(csv_path: str | Path) -> dict:
    """Load ADHD multi-seed results CSV."""
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    return {
        "sub_idx": [int(r["sub_idx"]) for r in rows],
        "r_fc_raw": np.array([float(r["r_fc_raw"]) for r in rows]),
        "rho_mean": np.array([float(r["rho_mean"]) for r in rows]),
        "rho_std": np.array([float(r["rho_std"]) for r in rows]),
        "rho_min": np.array([float(r["rho_min"]) for r in rows]),
        "rho_max": np.array([float(r["rho_max"]) for r in rows]),
        "group": [r.get("group", "unknown") for r in rows],
        "n_rois": int(rows[0].get("n_rois", 0)),
        "atlas": rows[0].get("atlas", "unknown"),
        "n_subjects": len(rows),
    }


# ============================================================================
# Helper: group scatter
# ============================================================================

def _scatter_by_group(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    yerr: np.ndarray | None = None,
) -> None:
    """Scatter plot with group-colored points.

    Args:
        ax: Matplotlib axes.
        x: X values.
        y: Y values.
        groups: Group labels per point.
        yerr: Optional error bars (1.96 * std for CI).
    """
    for grp, label in [("control", "Control"), ("adhd", "ADHD")]:
        mask = np.array([g == grp for g in groups])
        if not mask.any():
            continue
        color = GROUP_PALETTE.get(grp, GROUP_PALETTE["unknown"])
        if yerr is not None:
            ax.errorbar(
                x[mask], y[mask], yerr=yerr[mask],
                fmt="o", color=color, ecolor=color, capsize=2,
                ms=MARKER["secondary"], alpha=0.7, elinewidth=0.8,
                label=label,
            )
        else:
            ax.scatter(
                x[mask], y[mask], c=color, alpha=0.7,
                edgecolors="white", lw=0.5, s=MARKER["scatter"],
                label=label,
            )


# ============================================================================
# Plot: Single-seed (2x2)
# ============================================================================

def plot_single_seed(data: dict, atlas: str) -> None:
    """Generate 2x2 single-seed validation figure for ADHD.

    Panels:
      A. Scatter r_FC vs ρ̂T colored by group
      B. Improvement by group (box + strip)
      C. Distribution comparison (r_FC vs ρ̂T)
      D. Per-subject CI (sorted)
    """
    apply_bsnet_theme()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["2x2"])

    r_raw = data["r_fc_raw"]
    rho_hat = data["rho_hat_T"]
    improvement = data["improvement"]
    groups = data["group"]
    n = data["n_subjects"]
    n_rois = data["n_rois"]

    # ── Panel A: Scatter by group ──
    ax = axes[0, 0]
    _scatter_by_group(ax, r_raw, rho_hat, groups)
    lims = (min(r_raw.min(), rho_hat.min()) - 0.05, 1.02)
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=LINE["thin"], label="identity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    style_axis(
        ax,
        title=f"A. Raw FC vs BS-NET ({atlas.upper()}, {n_rois} ROIs, N={n})",
        xlabel="r_FC (raw, 2 min vs full)",
        ylabel="ρ̂T (BS-NET)",
        legend_loc="lower right",
        legend_fontsize=FONT["legend_small"],
    )

    # ── Panel B: Improvement by group (box + strip) ──
    ax = axes[0, 1]
    grp_data: dict[str, list[float]] = {}
    for g, imp in zip(groups, improvement):
        grp_data.setdefault(g, []).append(imp)
    grp_names = sorted(grp_data.keys())

    bp = ax.boxplot(
        [grp_data[g] for g in grp_names],
        tick_labels=[g.capitalize() for g in grp_names],
        patch_artist=True, vert=True, widths=0.5,
    )
    for patch, name in zip(bp["boxes"], grp_names):
        patch.set_facecolor(GROUP_PALETTE.get(name, GROUP_PALETTE["unknown"]))
        patch.set_alpha(0.4)
        patch.set_edgecolor("white")
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color(PALETTE["true"])
            line.set_linewidth(LINE["thin"])

    # Strip overlay
    rng = np.random.default_rng(42)
    for i, name in enumerate(grp_names):
        jitter = rng.uniform(-0.12, 0.12, len(grp_data[name]))
        ax.scatter(
            np.full(len(grp_data[name]), i + 1) + jitter,
            grp_data[name],
            c=GROUP_PALETTE.get(name, GROUP_PALETTE["unknown"]),
            edgecolors="white", lw=0.5, s=MARKER["scatter_small"],
            alpha=0.7, zorder=5,
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    style_axis(
        ax,
        title="B. Improvement by Group",
        ylabel="Δ (ρ̂T − r_FC)",
    )

    # ── Panel C: Distribution comparison ──
    ax = axes[1, 0]
    bins = np.linspace(0.5, 1.05, 25)
    ax.hist(
        r_raw, bins=bins, alpha=0.4,
        label=f"r_FC (μ={np.mean(r_raw):.3f})",
        color=PALETTE["raw"], edgecolor="white", linewidth=0.5,
    )
    ax.hist(
        rho_hat, bins=bins, alpha=0.5,
        label=f"ρ̂T (μ={np.mean(rho_hat):.3f})",
        color=PALETTE["bsnet"], edgecolor="white", linewidth=0.5,
    )
    ax.axvline(np.mean(r_raw), color=PALETTE["raw"], ls="--", lw=LINE["thin"])
    ax.axvline(np.mean(rho_hat), color=PALETTE["bsnet"], ls="--", lw=LINE["thin"])
    style_axis(
        ax,
        title="C. Distribution Comparison",
        xlabel="Correlation",
        ylabel="Count",
        legend_loc="upper left",
    )

    # ── Panel D: Per-subject CI (sorted by ρ̂T) ──
    ax = axes[1, 1]
    sort_idx = np.argsort(rho_hat)
    ci_lo = data["ci_lower"]
    ci_hi = data["ci_upper"]

    for pos, j in enumerate(sort_idx):
        clr = GROUP_PALETTE.get(groups[j], GROUP_PALETTE["unknown"])
        ax.errorbar(
            rho_hat[j], pos,
            xerr=[[rho_hat[j] - ci_lo[j]], [ci_hi[j] - rho_hat[j]]],
            fmt="o", color=clr, ecolor=PALETTE["ci_fill"],
            capsize=2, ms=MARKER["small"], elinewidth=0.8,
        )
    ax.scatter(
        r_raw[sort_idx], range(len(sort_idx)),
        marker="x", color=PALETTE["accent"], s=MARKER["scatter_small"],
        zorder=5, label="r_FC (raw)",
    )
    ax.set_yticks([])
    ax.set_ylabel(f"Subjects (N={n}, sorted)", fontsize=FONT["axis_label"])
    # Custom legend for groups
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_PALETTE["control"],
               markersize=8, label="Control"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_PALETTE["adhd"],
               markersize=8, label="ADHD"),
        Line2D([0], [0], marker="x", color=PALETTE["accent"],
               markersize=8, label="r_FC (raw)", linestyle="None"),
    ]
    ax.legend(handles=legend_elements, fontsize=FONT["legend_small"], loc="lower right")
    style_axis(
        ax,
        title="D. BS-NET with 95% CI (sorted by ρ̂T)",
        xlabel="Correlation",
    )

    plt.tight_layout(pad=3.0)
    save_figure(fig, f"Figure6_ADHD_SingleSeed_{atlas.upper()}.png")
    logger.info(f"Figure6_ADHD_SingleSeed_{atlas.upper()} saved.")


# ============================================================================
# Plot: Multi-seed (2x2)
# ============================================================================

def plot_multi_seed(data: dict, atlas: str) -> None:
    """Generate 2x2 multi-seed validation figure for ADHD.

    Panels:
      A. Scatter ± seed error by group
      B. Group comparison (box + strip) of ρ̂T
      C. Improvement distribution by group
      D. Seed stability (σ sorted, colored by group)
    """
    apply_bsnet_theme()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["2x2"])

    r_raw = data["r_fc_raw"]
    rho_mean = data["rho_mean"]
    rho_std = data["rho_std"]
    groups = data["group"]
    n = data["n_subjects"]
    improvement = rho_mean - r_raw

    # ── Panel A: Scatter with seed error bars ──
    ax = axes[0, 0]
    _scatter_by_group(ax, r_raw, rho_mean, groups, yerr=rho_std * 1.96)
    lims = (min(r_raw.min(), rho_mean.min()) - 0.05, 1.05)
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=LINE["thin"], label="identity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    style_axis(
        ax,
        title=f"A. Raw FC vs BS-NET ({atlas.upper()}, N={n}, multi-seed)",
        xlabel="r_FC (raw)",
        ylabel="ρ̂T (mean ± 1.96σ)",
        legend_loc="lower right",
        legend_fontsize=FONT["legend_small"],
    )

    # ── Panel B: Group comparison (box + strip) ──
    ax = axes[0, 1]
    grp_rho: dict[str, list[float]] = {}
    for g, rho in zip(groups, rho_mean):
        grp_rho.setdefault(g, []).append(rho)
    grp_names = sorted(grp_rho.keys())

    bp = ax.boxplot(
        [grp_rho[g] for g in grp_names],
        tick_labels=[g.capitalize() for g in grp_names],
        patch_artist=True, vert=True, widths=0.5,
    )
    for patch, name in zip(bp["boxes"], grp_names):
        patch.set_facecolor(GROUP_PALETTE.get(name, GROUP_PALETTE["unknown"]))
        patch.set_alpha(0.4)
        patch.set_edgecolor("white")
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color(PALETTE["true"])
            line.set_linewidth(LINE["thin"])

    rng = np.random.default_rng(42)
    for i, name in enumerate(grp_names):
        jitter = rng.uniform(-0.12, 0.12, len(grp_rho[name]))
        ax.scatter(
            np.full(len(grp_rho[name]), i + 1) + jitter,
            grp_rho[name],
            c=GROUP_PALETTE.get(name, GROUP_PALETTE["unknown"]),
            edgecolors="white", lw=0.5, s=MARKER["scatter_small"],
            alpha=0.7, zorder=5,
        )
    style_axis(
        ax,
        title="B. ρ̂T by Group (mean across seeds)",
        ylabel="ρ̂T",
    )

    # ── Panel C: Improvement distribution by group ──
    ax = axes[1, 0]
    bins = np.linspace(min(improvement) - 0.02, max(improvement) + 0.02, 15)
    for grp, label in [("control", "Control"), ("adhd", "ADHD")]:
        mask = np.array([g == grp for g in groups])
        if mask.any():
            ax.hist(
                improvement[mask], bins=bins, alpha=0.5,
                label=label, color=GROUP_PALETTE[grp],
                edgecolor="white", linewidth=0.5,
            )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(
        np.mean(improvement), color=PALETTE["highlight"],
        linewidth=LINE["secondary"], linestyle="-",
        label=f"mean = {np.mean(improvement):.3f}",
    )
    style_axis(
        ax,
        title="C. Improvement Distribution by Group",
        xlabel="Δ (ρ̂T_mean − r_FC)",
        ylabel="Count",
        legend_loc="upper left",
        legend_fontsize=FONT["legend_small"],
    )

    # ── Panel D: Seed stability per subject ──
    ax = axes[1, 1]
    sort_idx = np.argsort(rho_std)
    bar_colors = [GROUP_PALETTE.get(groups[j], GROUP_PALETTE["unknown"]) for j in sort_idx]
    ax.barh(
        range(n), rho_std[sort_idx],
        color=bar_colors, edgecolor="white", linewidth=0.3, height=0.8,
    )
    ax.axvline(
        np.mean(rho_std), color=PALETTE["highlight"],
        linewidth=LINE["secondary"], linestyle="--",
        label=f"mean σ = {np.mean(rho_std):.4f}",
    )
    ax.set_yticks([])
    ax.set_ylabel(f"Subjects (N={n}, sorted by σ)", fontsize=FONT["axis_label"])
    # Group legend
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GROUP_PALETTE["control"],
               markersize=8, label="Control"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=GROUP_PALETTE["adhd"],
               markersize=8, label="ADHD"),
    ]
    ax.legend(handles=legend_elements, fontsize=FONT["legend_small"], loc="lower right")
    style_axis(
        ax,
        title=f"D. Seed Stability (mean σ={np.mean(rho_std):.4f})",
        xlabel="σ(ρ̂T) across seeds",
    )

    plt.tight_layout(pad=3.0)
    save_figure(fig, f"Figure6_ADHD_MultiSeed_{atlas.upper()}.png")
    logger.info(f"Figure6_ADHD_MultiSeed_{atlas.upper()} saved.")


# ============================================================================
# Plot: Atlas comparison (2x2)
# ============================================================================

def plot_atlas_comparison() -> None:
    """Generate 2x2 atlas comparison (CC200 vs CC400) figure.

    Panels:
      A. ρ̂T distribution by atlas
      B. Improvement distribution by atlas
      C. Paired scatter CC200 vs CC400
      D. Summary bar chart (r_FC vs ρ̂T by atlas)
    """
    datasets: dict[str, dict] = {}
    for atlas in ("cc200", "cc400"):
        csv_path = RESULTS_DIR / f"adhd_bsnet_{atlas}.csv"
        if csv_path.exists():
            datasets[atlas] = load_single_seed(str(csv_path))

    if len(datasets) < 2:
        logger.warning(
            f"Atlas comparison requires CC200 and CC400. Found: {list(datasets.keys())}"
        )
        return

    apply_bsnet_theme()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["2x2"])

    # ── Panel A: ρ̂T distribution by atlas ──
    ax = axes[0, 0]
    bins = np.linspace(0.6, 1.02, 25)
    for atlas, d in datasets.items():
        ax.hist(
            d["rho_hat_T"], bins=bins, alpha=0.5,
            label=f"{atlas.upper()} (μ={np.mean(d['rho_hat_T']):.3f})",
            color=ATLAS_PALETTE[atlas], edgecolor="white", linewidth=0.5,
        )
    style_axis(
        ax,
        title="A. ρ̂T Distribution by Atlas",
        xlabel="ρ̂T",
        ylabel="Count",
        legend_loc="upper left",
    )

    # ── Panel B: Improvement distribution ──
    ax = axes[0, 1]
    bins = np.linspace(-0.15, 0.35, 20)
    for atlas, d in datasets.items():
        ax.hist(
            d["improvement"], bins=bins, alpha=0.5,
            label=f"{atlas.upper()} (μ={np.mean(d['improvement']):+.3f})",
            color=ATLAS_PALETTE[atlas], edgecolor="white", linewidth=0.5,
        )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    style_axis(
        ax,
        title="B. Improvement by Atlas",
        xlabel="Δ (ρ̂T − r_FC)",
        ylabel="Count",
        legend_loc="upper right",
    )

    # ── Panel C: Paired scatter CC200 vs CC400 ──
    ax = axes[1, 0]
    d200 = datasets["cc200"]
    d400 = datasets["cc400"]

    # Match by sub_idx
    idx_set = set(d200["sub_idx"]) & set(d400["sub_idx"])
    rho200 = {s: r for s, r in zip(d200["sub_idx"], d200["rho_hat_T"])}
    rho400 = {s: r for s, r in zip(d400["sub_idx"], d400["rho_hat_T"])}
    groups_200 = {s: g for s, g in zip(d200["sub_idx"], d200["group"])}

    sorted_idx = sorted(idx_set)
    paired_200 = np.array([rho200[s] for s in sorted_idx])
    paired_400 = np.array([rho400[s] for s in sorted_idx])
    grp_colors = [
        GROUP_PALETTE.get(groups_200.get(s, "unknown"), GROUP_PALETTE["unknown"])
        for s in sorted_idx
    ]

    ax.scatter(
        paired_200, paired_400, c=grp_colors,
        edgecolors="white", lw=0.5, s=MARKER["scatter"], alpha=0.7,
    )
    lims = (
        min(paired_200.min(), paired_400.min()) - 0.02,
        max(paired_200.max(), paired_400.max()) + 0.02,
    )
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=LINE["thin"])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    r_corr = np.corrcoef(paired_200, paired_400)[0, 1]
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_PALETTE["control"],
               markersize=8, label="Control"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_PALETTE["adhd"],
               markersize=8, label="ADHD"),
    ]
    ax.legend(handles=legend_elements, fontsize=FONT["legend_small"])
    style_axis(
        ax,
        title=f"C. Paired Comparison (r={r_corr:.3f})",
        xlabel="ρ̂T (CC200)",
        ylabel="ρ̂T (CC400)",
    )

    # ── Panel D: Summary bar chart ──
    ax = axes[1, 1]
    atlas_names = list(datasets.keys())
    x = np.arange(len(atlas_names))
    r_means = [np.mean(datasets[a]["r_fc_raw"]) for a in atlas_names]
    rho_means = [np.mean(datasets[a]["rho_hat_T"]) for a in atlas_names]
    r_stds = [np.std(datasets[a]["r_fc_raw"]) for a in atlas_names]
    rho_stds = [np.std(datasets[a]["rho_hat_T"]) for a in atlas_names]

    w = 0.3
    ax.bar(
        x - w / 2, r_means, w, yerr=r_stds,
        label="r_FC (raw)", color=PALETTE["raw"],
        edgecolor="white", capsize=5, error_kw={"linewidth": LINE["error"]},
    )
    ax.bar(
        x + w / 2, rho_means, w, yerr=rho_stds,
        label="ρ̂T (BS-NET)", color=PALETTE["bsnet"],
        edgecolor="white", capsize=5, error_kw={"linewidth": LINE["error"]},
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in atlas_names], fontsize=FONT["tick"])
    # ROI count annotations
    for i, a in enumerate(atlas_names):
        nr = datasets[a]["n_rois"]
        ax.text(
            i, 0.55, f"{nr} ROIs",
            ha="center", fontsize=FONT["legend_small"], style="italic",
            color=PALETTE["true"],
        )
    style_axis(
        ax,
        title="D. Summary: r_FC vs ρ̂T by Atlas",
        ylabel="Correlation",
        legend_loc="upper right",
    )

    plt.tight_layout(pad=3.0)
    save_figure(fig, "Figure6_ADHD_Atlas_Comparison.png")
    logger.info("Figure6_ADHD_Atlas_Comparison saved.")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """Generate all ADHD Figure 6 series."""
    parser = argparse.ArgumentParser(description="Plot ADHD Figure 6 series")
    parser.add_argument(
        "--atlas", nargs="+", default=["cc200", "cc400"],
        help="Atlas(es) for single/multi-seed plots (default: cc200 cc400)",
    )
    parser.add_argument(
        "--skip-atlas-compare", action="store_true",
        help="Skip atlas comparison figure",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    for atlas in args.atlas:
        # Single-seed
        ss_csv = RESULTS_DIR / f"adhd_bsnet_{atlas}.csv"
        if ss_csv.exists():
            logger.info(f"Generating SingleSeed figure for {atlas.upper()}...")
            ss_data = load_single_seed(str(ss_csv))
            plot_single_seed(ss_data, atlas)
        else:
            logger.warning(f"Single-seed CSV not found: {ss_csv}")

        # Multi-seed
        ms_csvs = sorted(RESULTS_DIR.glob(f"adhd_multiseed_{atlas}_*seeds.csv"))
        if ms_csvs:
            ms_csv = ms_csvs[-1]  # latest
            logger.info(f"Generating MultiSeed figure for {atlas.upper()}...")
            ms_data = load_multi_seed(str(ms_csv))
            plot_multi_seed(ms_data, atlas)
        else:
            logger.warning(f"Multi-seed CSV not found for {atlas}")

    # Atlas comparison
    if not args.skip_atlas_compare:
        logger.info("Generating Atlas Comparison figure...")
        plot_atlas_comparison()


if __name__ == "__main__":
    main()
