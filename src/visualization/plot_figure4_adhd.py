"""Figure 4: Cross-Dataset Generalization — ADHD-200 PCP (N=399).

Two-row layout (matching Fig 3 ABIDE style):
  Row 1 (A): CONSORT-style quality-filtering flowchart
             768 converted → 497 known DX → 496 k≥2 → 399 ref≥5min
  Row 2 (B–E): Validation panels
    B. Scatter: r_FC (raw) vs ρ̂T (BS-NET) per subject, group-colored
    C. Floating box: Raw FC vs BS-NET ρ̂T distribution
    D. Floating box: Improvement (Δ = ρ̂T − r_FC)
    E. Floating box: Seed σ (cross-seed stability)

Style: BS-NET theme (style.py), IQR floating bar + scatter dots + red diamond
Color: Amber (Raw) + Blue (BS-NET) — Fig 3–6 schema
       Scatter: Red (ADHD) + Blue (Control)

Data: data/adhd/pcp/results/adhd200_multiseed_cc200_10seeds_filtered_strict.csv
Output: Fig4_ADHD_Validation.png

Usage:
    PYTHONPATH=. python src/visualization/plot_figure4_adhd.py
    PYTHONPATH=. python src/visualization/plot_figure4_adhd.py --atlas cc200
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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

RESULTS_DIR = Path("data/adhd/pcp/results")

# ── Colors ──
COLOR_RAW = CONDITION_PALETTE["raw"]
COLOR_BSNET = CONDITION_PALETTE["bsnet"]
COLOR_IMPROVE = ACCENT_COLORS["improvement"]
COLOR_SEED = ACCENT_COLORS["seed_sigma"]
COLOR_ADHD = ACCENT_COLORS["adhd_group"]
COLOR_CONTROL = ACCENT_COLORS["control_group"]

# Floating box plot style (matching Fig 3)
DOT_SIZE = 4.0
DOT_ALPHA = 0.15
MAX_DOTS = 250
BAR_ALPHA = 0.70
BAR_WIDTH = 0.55
JITTER_WIDTH = 0.10

# CONSORT box colors (matching Fig 3)
CONSORT_COLORS = {
    "enrolled": "#E8F5E9",
    "stage": "#E3F2FD",
    "excluded": "#FFEBEE",
    "final": "#FFF9C4",
    "edge": "#666666",
    "edge_excl": "#E57373",
    "arrow": "#333333",
    "arrow_excl": "#999999",
}


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_multiseed(atlas: str) -> dict:
    """Load ADHD-200 PCP multi-seed CSV.

    Args:
        atlas: Atlas name (cc200).

    Returns:
        Dict with r_fc, rho_mean, rho_std, group, site, n.
    """
    csv_path = RESULTS_DIR / f"adhd200_multiseed_{atlas}_10seeds_filtered_strict.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))
    return {
        "r_fc": np.array([float(r["r_fc_raw"]) for r in rows]),
        "rho_mean": np.array([float(r["rho_hat_T_mean"]) for r in rows]),
        "rho_std": np.array([float(r["rho_hat_T_std"]) for r in rows]),
        "group": np.array([r["group"].lower().strip() for r in rows]),
        "site": [r["site"] for r in rows],
        "n": len(rows),
    }


def load_consort_stages(atlas: str) -> list[dict]:
    """Load CONSORT stages from summary JSON.

    Returns:
        List of stage dicts with name, n, n_excluded.
    """
    json_path = RESULTS_DIR / f"adhd200_filtered_strict_{atlas}_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {json_path}")

    with open(json_path) as f:
        summary = json.load(f)

    stages = summary.get("stages", [])
    if not stages:
        raise ValueError(f"No stages found in {json_path}")

    # Add site count from summary for display
    n_sites = summary.get("n_sites", 6)
    if stages:
        stages[-1]["n_sites"] = n_sites

    return stages


# ═══════════════════════════════════════════════════════════════════════
# CONSORT panel drawing
# ═══════════════════════════════════════════════════════════════════════

def _draw_consort_panel(ax: plt.Axes, stages: list[dict]) -> None:
    """Draw CONSORT flowchart on a single Axes.

    ADHD-200 PCP stages:
      768 Converted → 497 Known DX → 496 k≥2 → 399 ref≥5min → Analysis Set
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 10.5)
    ax.axis("off")

    # Append Analysis Set node
    last = stages[-1]
    display = list(stages) + [{
        "name": "Analysis Set",
        "n": last["n"],
        "n_sites": last.get("n_sites", ""),
        "n_excluded": 0,
        "_final": True,
    }]
    n_display = len(display)

    box_w, box_h = 2.8, 1.3
    excl_w, excl_h = 2.5, 1.1
    center_x = 2.8
    excl_x = 6.8
    pad = 0.08
    arrow_gap = 0.08
    y_top = 9.5
    y_step = (y_top - 0.5) / max(n_display - 1, 1)

    font_box = FONT["tick"]
    font_excl = FONT["legend_small"]

    def _box(x, y, w, h, text, color, edge, fs):
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle=f"round,pad={pad}",
            facecolor=color, edgecolor=edge, linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fs, linespacing=1.35)

    def _arrow_down(x, y1, y2):
        ax.annotate("", xy=(x, y2 + pad + arrow_gap),
                     xytext=(x, y1 - pad - arrow_gap),
                     arrowprops=dict(arrowstyle="->",
                                     color=CONSORT_COLORS["arrow"], lw=1.3))

    def _arrow_right(x1, y, x2):
        ax.annotate("", xy=(x2 - pad - arrow_gap, y),
                     xytext=(x1 + pad + arrow_gap, y),
                     arrowprops=dict(arrowstyle="->",
                                     color=CONSORT_COLORS["arrow_excl"],
                                     lw=1.0, linestyle="--"))

    # Exclusion reasons for each stage
    excl_reasons = {
        "Known DX": "Unknown DX\n(competition test set)",
        "k ≥ 2.0 (ref ≥ short)": "k < 2\n(ref < short scan)",
        "Reference ≥ 5 min": "Reference < 5 min\n(KKI site)",
    }

    for i, stage in enumerate(display):
        y = y_top - i * y_step
        is_first = i == 0
        is_final = stage.get("_final", False)

        n_sites = stage.get("n_sites", "")
        site_str = f", {n_sites} sites" if n_sites else ""

        if is_first:
            text = f"ADHD-200 PCP\nN = {stage['n']} (9 sites)"
            color = CONSORT_COLORS["enrolled"]
        elif is_final:
            text = f"Analysis Set\nN = {stage['n']}{site_str}"
            color = CONSORT_COLORS["final"]
        else:
            text = f"{stage['name']}\nN = {stage['n']}"
            color = CONSORT_COLORS["stage"]

        _box(center_x, y, box_w, box_h, text,
             color, CONSORT_COLORS["edge"], font_box)

        # Arrow to next
        if i < n_display - 1:
            _arrow_down(center_x, y - box_h / 2,
                        y - y_step + box_h / 2)

        # Exclusion box
        n_excl = stage.get("n_excluded", 0)
        if n_excl > 0 and not is_first:
            reason = excl_reasons.get(stage["name"], "")
            excl_text = f"Excluded: n = {n_excl}"
            if reason:
                excl_text += f"\n{reason}"

            _box(excl_x, y, excl_w, excl_h, excl_text,
                 CONSORT_COLORS["excluded"], CONSORT_COLORS["edge_excl"],
                 font_excl)
            _arrow_right(center_x + box_w / 2, y,
                         excl_x - excl_w / 2)

    ax.set_title("A. Quality Filtering (CONSORT)",
                 fontweight="bold", fontsize=FONT["title"],
                 pad=FONT["title_pad"])


# ═══════════════════════════════════════════════════════════════════════
# Validation panel helpers (matching Fig 3 exactly)
# ═══════════════════════════════════════════════════════════════════════

def _scatter_strip(
    ax: plt.Axes, vals: np.ndarray, x_pos: float,
    rng: np.random.RandomState, color: str,
) -> None:
    """Jittered scatter strip."""
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
    """Red diamond mean±SD marker."""
    m, sd = np.nanmean(vals), np.nanstd(vals)
    ax.errorbar(
        x_pos, m, yerr=sd,
        fmt="D", color="red", markersize=5.5,
        markeredgecolor="darkred", markeredgewidth=0.8,
        ecolor="darkred", elinewidth=1.2, capsize=3.5, capthick=1.0,
        zorder=10,
    )


def _floating_box(
    ax: plt.Axes, vals: np.ndarray, x_pos: float,
    rng: np.random.RandomState, color: str,
) -> None:
    """Draw IQR floating bar + scatter strip + mean diamond."""
    q1, q3 = np.nanpercentile(vals, [25, 75])
    ax.bar(
        x_pos, q3 - q1, bottom=q1, width=BAR_WIDTH,
        color=color, alpha=BAR_ALPHA, edgecolor="white",
        linewidth=0.5, zorder=2,
    )
    _scatter_strip(ax, vals, x_pos, rng, color=color)
    _mean_diamond(ax, vals, x_pos)


# ═══════════════════════════════════════════════════════════════════════
# Main figure
# ═══════════════════════════════════════════════════════════════════════

def plot_figure4(data: dict, atlas: str, stages: list[dict]) -> plt.Figure:
    """Create Figure 4: ADHD-200 PCP Validation with CONSORT + 4 panels.

    Layout:
        Row 1: [A. CONSORT flowchart — spans full width]
        Row 2: [B. Scatter (group-colored)] [C. Distribution] [D. Improvement] [E. Seed σ]

    Args:
        data: Dict from load_multiseed().
        atlas: Atlas name.
        stages: CONSORT stage dicts.

    Returns:
        Matplotlib Figure.
    """
    apply_bsnet_theme()

    r_fc = data["r_fc"]
    rho = data["rho_mean"]
    rho_std = data["rho_std"]
    group = data["group"]
    improvement = rho - r_fc
    n = data["n"]

    is_adhd = group == "adhd"
    is_ctrl = ~is_adhd

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        height_ratios=[1.1, 1.0],
        width_ratios=[1.2, 0.7, 0.5, 0.5],
        hspace=0.32, wspace=0.25,
    )

    rng = np.random.RandomState(42)

    # ── Row 1: CONSORT (spans all 4 columns) ──
    ax_consort = fig.add_subplot(gs[0, :])
    _draw_consort_panel(ax_consort, stages)

    # ── Row 2, Panel B: Scatter r_FC vs ρ̂T (group-colored) ──
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_scatter.scatter(
        r_fc[is_ctrl], rho[is_ctrl],
        s=18, c=COLOR_CONTROL, alpha=0.55,
        edgecolors="white", linewidth=0.3, zorder=3,
        label=f"Control (n={int(np.sum(is_ctrl))})",
        marker="o",
    )
    ax_scatter.scatter(
        r_fc[is_adhd], rho[is_adhd],
        s=18, c=COLOR_ADHD, alpha=0.55,
        edgecolors="white", linewidth=0.3, zorder=3,
        label=f"ADHD (n={int(np.sum(is_adhd))})",
        marker="^",
    )
    lims = [min(r_fc.min(), rho.min()) - 0.05,
            max(r_fc.max(), rho.max()) + 0.05]
    ax_scatter.plot(lims, lims, "k--", alpha=0.3,
                    linewidth=LINE["thin"], label="identity")
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_title(
        f"B. Raw FC vs BS-NET ({atlas.upper()}, N={n})",
        fontweight="bold", fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )
    ax_scatter.set_xlabel(r"$r_{FC}$ (raw, 2 min vs full)",
                          fontsize=FONT["axis_label"])
    ax_scatter.set_ylabel(r"$\hat{\rho}_T$ (BS-NET, mean across seeds)",
                          fontsize=FONT["axis_label"])
    ax_scatter.legend(loc="lower right", fontsize=FONT["legend_small"])

    # ── Row 2, Panel C: Floating box — Raw FC vs BS-NET ──
    ax_comp = fig.add_subplot(gs[1, 1])
    _floating_box(ax_comp, r_fc, 0, rng, COLOR_RAW)
    _floating_box(ax_comp, rho, 1, rng, COLOR_BSNET)

    n_improved = int(np.sum(improvement > 0))
    pct = n_improved / n * 100
    ax_comp.set_xticks([0, 1])
    ax_comp.set_xticklabels(["Raw FC\n(2 min)", "BS-NET\n(2 min)"],
                            fontsize=FONT["legend_small"])
    ax_comp.set_title(
        "C. Distribution Comparison",
        fontweight="bold", fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )
    ax_comp.set_ylabel("FC similarity", fontsize=FONT["axis_label"])
    ax_comp.set_xlabel("")
    ax_comp.text(
        0.5, 0.02,
        f"{pct:.1f}% improved ({n_improved}/{n})",
        transform=ax_comp.transAxes, ha="center",
        fontsize=FONT["legend_small"],
        fontstyle="italic", color="#555555",
    )

    # ── Row 2, Panel D: Floating box — Improvement (Δ) ──
    ax_imp = fig.add_subplot(gs[1, 2])
    _floating_box(ax_imp, improvement, 0, rng, COLOR_IMPROVE)
    ax_imp.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_imp.set_xticks([0])
    ax_imp.set_xticklabels([r"Δ ($\hat{\rho}_T - r_{FC}$)"],
                           fontsize=FONT["legend_small"])
    ax_imp.set_title(
        "D. Improvement (Δ)",
        fontweight="bold", fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )
    ax_imp.set_ylabel(r"Δ ($\hat{\rho}_T - r_{FC}$)",
                      fontsize=FONT["axis_label"])
    ax_imp.set_xlabel("")

    # ── Row 2, Panel E: Floating box — Seed σ ──
    ax_seed = fig.add_subplot(gs[1, 3])
    _floating_box(ax_seed, rho_std, 0, rng, COLOR_SEED)
    ax_seed.set_xticks([0])
    ax_seed.set_xticklabels(["Seed σ"], fontsize=FONT["legend_small"])
    ax_seed.set_title(
        "E. Seed Stability",
        fontweight="bold", fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )
    ax_seed.set_ylabel("Seed σ (cross-seed SD)",
                       fontsize=FONT["axis_label"])
    ax_seed.set_xlabel("")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Generate Figure 4."""
    parser = argparse.ArgumentParser(
        description="Figure 4: ADHD-200 PCP Validation (N=399)"
    )
    parser.add_argument("--atlas", default="cc200", choices=["cc200", "cc400"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    data = load_multiseed(args.atlas)
    stages = load_consort_stages(args.atlas)

    r_fc = data["r_fc"]
    rho = data["rho_mean"]
    group = data["group"]
    improvement = rho - r_fc

    print(f"ADHD-200 PCP {args.atlas.upper()} — N={data['n']}")
    print(f"  r_FC:  {r_fc.mean():.3f} ± {r_fc.std():.3f}")
    print(f"  ρ̂T:   {rho.mean():.3f} ± {rho.std():.3f}")
    print(f"  Δ:     {improvement.mean():+.3f} ± {improvement.std():.3f}")
    print(f"  Improved: {np.sum(improvement > 0)}/{data['n']} "
          f"({np.mean(improvement > 0)*100:.1f}%)")
    print(f"  Seed σ: {data['rho_std'].mean():.4f} (mean)")
    for g in ["adhd", "control"]:
        mask = group == g
        print(f"  {g.upper()} (n={mask.sum()}): "
              f"ρ̂T={rho[mask].mean():.3f}±{rho[mask].std():.3f}")

    fig = plot_figure4(data, args.atlas, stages)
    save_figure(fig, "Fig4_ADHD_Validation.png")
    print("\nFigure 4 saved: Fig4_ADHD_Validation.png")


if __name__ == "__main__":
    main()
