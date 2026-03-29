"""Generate Figure 5 series: ABIDE PCP validation (multi-seed + ceiling analysis).

Produces publication-quality figures matching Figure 1 style standards:
  - Figure5_ABIDE_MultiSeed_CC200.png  (2x2: scatter, improvement dist, seed stability, summary)
  - Figure5_ABIDE_MultiSeed_CC400.png  (same layout for CC400)
  - Figure5_ABIDE_Ceiling_CC200.png    (2x2: 4-method comparison)
  - Figure5_ABIDE_Ceiling_CC400.png    (same layout for CC400)

Data source:
  - data/abide/results/abide_multiseed_{atlas}_10seeds.csv
  - data/abide/results/ceiling_analysis_{atlas}.csv
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import (
    CORRECTION_PALETTE,
    FIGSIZE,
    FONT,
    LINE,
    MARKER,
    PALETTE,
    apply_bsnet_theme,
    save_figure,
    style_axis,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/abide/results")


# ============================================================================
# Data loaders
# ============================================================================

def load_multiseed(csv_path: str | Path) -> dict:
    """Load ABIDE multi-seed results CSV.

    Args:
        csv_path: Path to abide_multiseed_{atlas}_10seeds.csv.

    Returns:
        Dict with arrays: sub_id, site, r_fc_raw, rho_mean, rho_std, etc.
    """
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    return {
        "sub_id": [r["sub_id"] for r in rows],
        "site": [r["site"] for r in rows],
        "r_fc_raw": np.array([float(r["r_fc_raw"]) for r in rows]),
        "rho_mean": np.array([float(r["rho_hat_T_mean"]) for r in rows]),
        "rho_std": np.array([float(r["rho_hat_T_std"]) for r in rows]),
        "rho_min": np.array([float(r["rho_hat_T_min"]) for r in rows]),
        "rho_max": np.array([float(r["rho_hat_T_max"]) for r in rows]),
        "ci_lower_mean": np.array([float(r["ci_lower_mean"]) for r in rows]),
        "ci_upper_mean": np.array([float(r["ci_upper_mean"]) for r in rows]),
        "n_subjects": len(rows),
    }


def load_ceiling(csv_path: str | Path) -> dict:
    """Load ceiling analysis CSV with 4-method comparison.

    Args:
        csv_path: Path to ceiling_analysis_{atlas}.csv.

    Returns:
        Dict with per-method arrays: rho_{method}, r_fc_raw, etc.
    """
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    methods = ["original", "fisher_z", "partial", "soft_clamp"]
    data: dict = {
        "sub_id": [r["sub_id"] for r in rows],
        "r_fc_raw": np.array([float(r["r_fc_raw"]) for r in rows]),
        "n_subjects": len(rows),
    }
    for m in methods:
        data[f"rho_{m}"] = np.array([float(r[f"rho_{m}"]) for r in rows])
        data[f"ci_lo_{m}"] = np.array([float(r[f"ci_lo_{m}"]) for r in rows])
        data[f"ci_hi_{m}"] = np.array([float(r[f"ci_hi_{m}"]) for r in rows])
    return data


# ============================================================================
# Plot: Multi-seed validation (2x2)
# ============================================================================

def plot_multiseed(data: dict, atlas: str) -> None:
    """Generate 2x2 multi-seed validation figure for ABIDE.

    Panels:
      A. Scatter: r_FC vs ρ̂T (mean ± 1.96σ)
      B. Improvement distribution histogram
      C. Seed stability (σ per subject, sorted)
      D. Summary statistics box
    """
    apply_bsnet_theme()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["2x2"])

    r_raw = data["r_fc_raw"]
    rho_mean = data["rho_mean"]
    rho_std = data["rho_std"]
    n = data["n_subjects"]
    improvement = rho_mean - r_raw
    n_improved = np.sum(improvement > 0)

    # ── Panel A: Scatter with error bars ──
    ax = axes[0, 0]
    ax.errorbar(
        r_raw, rho_mean, yerr=rho_std * 1.96,
        fmt="o", color=PALETTE["bsnet"], ecolor=PALETTE["ci_fill"],
        capsize=2, ms=MARKER["small"], alpha=0.6, elinewidth=0.8,
    )
    lims = (min(r_raw.min(), rho_mean.min()) - 0.05, 1.05)
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=LINE["thin"], label="identity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    style_axis(
        ax,
        title=f"A. Raw FC vs BS-NET ({atlas.upper()}, N={n})",
        xlabel="r_FC (raw, 2 min vs full)",
        ylabel="ρ̂T (mean ± 1.96σ across seeds)",
        legend_loc="lower right",
    )

    # ── Panel B: Improvement distribution ──
    ax = axes[0, 1]
    bins = np.linspace(
        min(improvement.min(), -0.05),
        max(improvement.max(), 0.3),
        30,
    )
    ax.hist(
        improvement, bins=bins, color=PALETTE["bsnet"],
        edgecolor="white", alpha=0.8, linewidth=0.5,
    )
    ax.axvline(0, color="black", linewidth=LINE["thin"], linestyle="--")
    ax.axvline(
        np.mean(improvement), color=PALETTE["highlight"],
        linewidth=LINE["secondary"], linestyle="-",
        label=f"mean = {np.mean(improvement):.3f}",
    )
    pct = n_improved / n * 100
    ax.text(
        0.95, 0.95, f"{pct:.1f}% improved",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=FONT["annotation"], fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["ci_fill"], alpha=0.8),
    )
    style_axis(
        ax,
        title="B. Improvement Distribution (ρ̂T − r_FC)",
        xlabel="Δ (ρ̂T − r_FC)",
        ylabel="Count",
        legend_loc="upper left",
    )

    # ── Panel C: Seed stability (sorted σ) ──
    ax = axes[1, 0]
    sort_idx = np.argsort(rho_std)
    y_pos = np.arange(n)
    ax.barh(
        y_pos, rho_std[sort_idx],
        color=PALETTE["accent"], edgecolor="white", linewidth=0.3, height=0.8,
    )
    ax.axvline(
        np.mean(rho_std), color=PALETTE["highlight"],
        linewidth=LINE["secondary"], linestyle="--",
        label=f"mean σ = {np.mean(rho_std):.4f}",
    )
    ax.set_yticks([])
    ax.set_ylabel(f"Subjects (N={n}, sorted by σ)", fontsize=FONT["axis_label"])
    style_axis(
        ax,
        title="C. Prediction Stability Across Seeds",
        xlabel="σ(ρ̂T) across seeds",
        legend_loc="lower right",
    )

    # ── Panel D: Summary statistics ──
    ax = axes[1, 1]
    ax.axis("off")

    ceiling_count = np.sum(rho_mean > 0.999)
    ceiling_pct = ceiling_count / n * 100

    summary_text = (
        f"── ABIDE {atlas.upper()} Multi-Seed Summary ──\n\n"
        f"N subjects:       {n}\n"
        f"N seeds:          10\n\n"
        f"r_FC (raw):       {np.mean(r_raw):.3f} ± {np.std(r_raw):.3f}\n"
        f"ρ̂T (BS-NET):     {np.mean(rho_mean):.3f} ± {np.std(rho_mean):.3f}\n\n"
        f"Improvement:      {np.mean(improvement):+.3f} ± {np.std(improvement):.3f}\n"
        f"Improved:         {n_improved}/{n} ({pct:.1f}%)\n"
        f"Ceiling (>0.999): {ceiling_count}/{n} ({ceiling_pct:.1f}%)\n\n"
        f"Seed σ (mean):    {np.mean(rho_std):.4f}\n"
        f"Seed σ (max):     {np.max(rho_std):.4f}"
    )
    ax.text(
        0.1, 0.95, summary_text,
        transform=ax.transAxes, va="top", ha="left",
        fontsize=12, fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=PALETTE["ci_fill"],
            alpha=0.3,
            edgecolor=PALETTE["true"],
        ),
    )
    ax.set_title(
        "D. Summary Statistics",
        fontweight=FONT["title_weight"],
        fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )

    plt.tight_layout(pad=3.0)
    save_figure(fig, f"Figure5_ABIDE_MultiSeed_{atlas.upper()}.png")
    logger.info(f"Figure5_ABIDE_MultiSeed_{atlas.upper()} saved.")


# ============================================================================
# Plot: Ceiling analysis (2x2)
# ============================================================================

def plot_ceiling(data: dict, atlas: str) -> None:
    """Generate 2x2 ceiling analysis figure comparing 4 correction methods.

    Panels:
      A. Distribution of ρ̂T per method (overlapped histograms)
      B. Ceiling effect rate per method (bar chart)
      C. Scatter: Fisher z vs Original
      D. Method comparison summary (improvement per method)
    """
    apply_bsnet_theme()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["2x2"])

    r_raw = data["r_fc_raw"]
    n = data["n_subjects"]
    methods = ["original", "fisher_z", "partial", "soft_clamp"]
    method_labels = {
        "original": "Original",
        "fisher_z": "Fisher z",
        "partial": "Partial (α=0.5)",
        "soft_clamp": "Soft Clamp",
    }

    # ── Panel A: Distribution of ρ̂T per method ──
    ax = axes[0, 0]
    bins = np.linspace(0.6, 1.02, 30)
    for m in methods:
        rho = data[f"rho_{m}"]
        ax.hist(
            rho, bins=bins, alpha=0.5,
            color=CORRECTION_PALETTE[m],
            edgecolor="white", linewidth=0.3,
            label=f"{method_labels[m]} (μ={np.mean(rho):.3f})",
        )
    style_axis(
        ax,
        title=f"A. ρ̂T Distribution by Method ({atlas.upper()})",
        xlabel="ρ̂T",
        ylabel="Count",
        legend_loc="upper left",
        legend_fontsize=FONT["legend_small"],
    )

    # ── Panel B: Ceiling rate bar chart ──
    ax = axes[0, 1]
    ceiling_rates = []
    for m in methods:
        rho = data[f"rho_{m}"]
        rate = np.sum(rho > 0.999) / n * 100
        ceiling_rates.append(rate)

    x = np.arange(len(methods))
    colors = [CORRECTION_PALETTE[m] for m in methods]
    bars = ax.bar(
        x, ceiling_rates, color=colors,
        edgecolor="white", linewidth=0.8, width=0.6,
    )
    for bar, rate in zip(bars, ceiling_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center", va="bottom",
            fontsize=FONT["annotation"], fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [method_labels[m] for m in methods],
        rotation=15, ha="right", fontsize=FONT["tick"],
    )
    ax.set_ylim(0, max(ceiling_rates) + 15 if max(ceiling_rates) > 0 else 10)
    style_axis(
        ax,
        title="B. Ceiling Effect Rate (ρ̂T > 0.999)",
        ylabel="% Subjects with Ceiling",
    )

    # ── Panel C: Scatter Fisher z vs Original ──
    ax = axes[1, 0]
    rho_orig = data["rho_original"]
    rho_fz = data["rho_fisher_z"]
    ax.scatter(
        rho_orig, rho_fz,
        c=PALETTE["bsnet"], alpha=0.5, s=MARKER["scatter_small"],
        edgecolors="white", linewidth=0.3,
    )
    lims = (min(rho_orig.min(), rho_fz.min()) - 0.02, 1.02)
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=LINE["thin"], label="identity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # Highlight ceiling zone
    ax.axvspan(0.999, 1.02, alpha=0.1, color=PALETTE["highlight"], label="ceiling zone")
    style_axis(
        ax,
        title="C. Original vs Fisher z Correction",
        xlabel="ρ̂T (Original)",
        ylabel="ρ̂T (Fisher z)",
        legend_loc="upper left",
        legend_fontsize=FONT["legend_small"],
    )

    # ── Panel D: Mean improvement per method ──
    ax = axes[1, 1]
    improvements = []
    imp_stds = []
    for m in methods:
        rho = data[f"rho_{m}"]
        imp = rho - r_raw
        improvements.append(np.mean(imp))
        imp_stds.append(np.std(imp))

    bars = ax.bar(
        x, improvements, yerr=imp_stds,
        color=colors, edgecolor="white", linewidth=0.8, width=0.6,
        capsize=4, error_kw={"linewidth": LINE["error"]},
    )
    for bar, imp_val, std_val in zip(bars, improvements, imp_stds):
        y_pos = bar.get_height() + std_val + 0.005
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{imp_val:+.3f}",
            ha="center", va="bottom",
            fontsize=FONT["annotation"], fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [method_labels[m] for m in methods],
        rotation=15, ha="right", fontsize=FONT["tick"],
    )
    ax.axhline(0, color="black", linewidth=0.8)
    style_axis(
        ax,
        title="D. Mean Improvement by Method (ρ̂T − r_FC)",
        ylabel="Mean Δ (ρ̂T − r_FC)",
    )

    plt.tight_layout(pad=3.0)
    save_figure(fig, f"Figure5_ABIDE_Ceiling_{atlas.upper()}.png")
    logger.info(f"Figure5_ABIDE_Ceiling_{atlas.upper()} saved.")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """Generate all ABIDE Figure 5 series."""
    parser = argparse.ArgumentParser(description="Plot ABIDE Figure 5 series")
    parser.add_argument(
        "--atlas", nargs="+", default=["cc200", "cc400"],
        help="Atlas(es) to plot (default: cc200 cc400)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    for atlas in args.atlas:
        # Multi-seed
        ms_csv = RESULTS_DIR / f"abide_multiseed_{atlas}_10seeds.csv"
        if ms_csv.exists():
            logger.info(f"Generating MultiSeed figure for {atlas.upper()}...")
            ms_data = load_multiseed(ms_csv)
            plot_multiseed(ms_data, atlas)
        else:
            logger.warning(f"Multi-seed CSV not found: {ms_csv}")

        # Ceiling
        ceil_csv = RESULTS_DIR / f"ceiling_analysis_{atlas}.csv"
        if ceil_csv.exists():
            logger.info(f"Generating Ceiling figure for {atlas.upper()}...")
            ceil_data = load_ceiling(ceil_csv)
            plot_ceiling(ceil_data, atlas)
        else:
            logger.warning(f"Ceiling CSV not found: {ceil_csv}")


if __name__ == "__main__":
    main()
