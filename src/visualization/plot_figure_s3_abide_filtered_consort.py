"""Supplementary Figure S3: ABIDE strict-filtered validation + CONSORT.

Two-row layout:
  Row 1 (A): CONSORT-style quality-filtering flowchart
  Row 2 (B–E): Validation panels
    B. Scatter: r_FC (raw) vs ρ̂T (BS-NET) per subject
    C. Floating box: Raw FC vs BS-NET ρ̂T distribution (IQR bar + scatter + diamond)
    D. Floating box: Improvement (Δ = ρ̂T − r_FC)
    E. Floating box: Seed σ (cross-seed stability)

Style: BS-NET theme (style.py), IQR floating bar + scatter dots + red diamond mean±SD
       (consistent with Fig 2 Panel A)
Color: Amber (Raw) + Blue (BS-NET) — Fig 3–6 schema

Data: data/abide/results/abide_multiseed_cc200_10seeds_filtered_strict.csv
Output: FigS3_ABIDE_Filtered_CONSORT.png
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import logging
import sys
from collections import Counter
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

RESULTS_DIR = Path("data/abide/results")

# ── Colors ──
COLOR_RAW = CONDITION_PALETTE["raw"]
COLOR_BSNET = CONDITION_PALETTE["bsnet"]
COLOR_IMPROVE = ACCENT_COLORS["improvement"]
COLOR_SEED = ACCENT_COLORS["seed_sigma"]

# Violin+box style (matching Fig 2 Panel D scheme)
VIOLIN_ALPHA = 0.45
BOX_ALPHA = 0.80
VIOLIN_WIDTH = 0.5
BOX_WIDTH = 0.24
OUTLIER_MARKER_SIZE = 3.0
OUTLIER_ALPHA = 0.35

# CONSORT box colors — using muted tones consistent with BS-NET palette
CONSORT_COLORS = {
    "enrolled": "#E8F5E9",   # light green
    "stage": "#E3F2FD",      # light blue (matches bsnet palette)
    "excluded": "#FFEBEE",   # light red
    "final": "#FFF9C4",      # light yellow
    "edge": "#666666",
    "edge_excl": "#E57373",
    "arrow": "#333333",
    "arrow_excl": "#999999",
}

SHORT_TRS = 60


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_multiseed(atlas: str, filtered: bool = True) -> dict:
    """Load ABIDE multi-seed CSV.

    Args:
        atlas: Atlas name.
        filtered: If True, load filtered_strict CSV.

    Returns:
        Dict with r_fc, rho_mean, rho_std, site, n.
    """
    if filtered:
        csv_path = RESULTS_DIR / f"abide_multiseed_{atlas}_10seeds_filtered_strict.csv"
    else:
        csv_path = RESULTS_DIR / f"abide_multiseed_{atlas}_10seeds.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))
    return {
        "r_fc": np.array([float(r["r_fc_raw"]) for r in rows]),
        "rho_mean": np.array([float(r["rho_hat_T_mean"]) for r in rows]),
        "rho_std": np.array([float(r["rho_hat_T_std"]) for r in rows]),
        "site": [r["site"] for r in rows],
        "n": len(rows),
    }


def load_consort_stages(atlas: str) -> list[dict] | None:
    """Load CONSORT stages from summary JSON.

    Returns:
        List of stage dicts, or None if not found.
    """
    json_path = RESULTS_DIR / f"abide_filtered_strict_{atlas}_summary.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        summary = json.load(f)
    return summary.get("stages", None)


def compute_consort_data(atlas: str) -> list[dict]:
    """Compute CONSORT stages from raw subject data (fallback).

    Returns:
        List of stage dicts with name, n, n_excluded, excl_sites.
    """
    subj_path = RESULTS_DIR / f"abide_subjects_{atlas}.json"
    with open(subj_path) as f:
        raw = json.load(f)

    subjects = []
    for s in raw:
        ts_path = s["ts_path"]
        if not Path(ts_path).exists():
            continue
        ts = np.load(ts_path)
        n_trs = ts.shape[0]
        tr = s.get("tr", 2.0)
        subjects.append({
            "sub_id": s["sub_id"], "site": s["site"],
            "tr": tr, "n_trs": n_trs,
            "total_s": n_trs * tr,
            "ref_s": (n_trs - SHORT_TRS) * tr,
            "k": n_trs / SHORT_TRS,
        })

    stages = []
    current = subjects

    stages.append({
        "name": "ABIDE PCP (Controls)",
        "n": len(current),
        "n_sites": len(set(s["site"] for s in current)),
        "n_excluded": 0,
        "excl_sites": {},
    })

    # k >= 2
    excluded = [s for s in current if s["k"] < 2.0]
    current = [s for s in current if s["k"] >= 2.0]
    stages.append({
        "name": "k ≥ 2.0",
        "n": len(current),
        "n_sites": len(set(s["site"] for s in current)),
        "n_excluded": len(excluded),
        "excl_sites": dict(Counter(s["site"] for s in excluded)),
        "reason": "k < 2\n(ref < short scan)",
    })

    # ref >= 5 min
    excluded = [s for s in current if s["ref_s"] < 300]
    current = [s for s in current if s["ref_s"] >= 300]
    stages.append({
        "name": "Reference ≥ 5 min",
        "n": len(current),
        "n_sites": len(set(s["site"] for s in current)),
        "n_excluded": len(excluded),
        "excl_sites": dict(Counter(s["site"] for s in excluded)),
        "reason": "Reference < 5 min",
    })

    return stages


# ═══════════════════════════════════════════════════════════════════════
# CONSORT panel drawing
# ═══════════════════════════════════════════════════════════════════════

def _draw_consort_panel(ax: plt.Axes, stages: list[dict]) -> None:
    """Draw CONSORT flowchart on a single Axes.

    Appends an "Analysis Set" box after the last filtering stage.
    Uses BS-NET FONT sizes and color palette.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 10.5)
    ax.axis("off")

    # Build display list: original stages + final "Analysis Set" node
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
    pad = 0.08  # FancyBboxPatch padding
    arrow_gap = 0.08  # small gap between arrow tip and box edge
    y_top = 9.5
    # Compact spacing: total vertical span = y_top - 0.5, distribute evenly
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
        # Start/end just outside the pad region
        ax.annotate("", xy=(x, y2 + pad + arrow_gap),
                     xytext=(x, y1 - pad - arrow_gap),
                     arrowprops=dict(arrowstyle="->",
                                     color=CONSORT_COLORS["arrow"], lw=1.3))

    def _arrow_right(x1, y, x2):
        # Start after main box pad, end before excl box pad
        ax.annotate("", xy=(x2 - pad - arrow_gap, y),
                     xytext=(x1 + pad + arrow_gap, y),
                     arrowprops=dict(arrowstyle="->",
                                     color=CONSORT_COLORS["arrow_excl"],
                                     lw=1.0, linestyle="--"))

    for i, stage in enumerate(display):
        y = y_top - i * y_step
        is_first = i == 0
        is_final = stage.get("_final", False)

        n_sites = stage.get("n_sites", "")
        site_str = f", {n_sites} sites" if n_sites else ""

        if is_first:
            text = f"{stage['name']}\nN = {stage['n']}{site_str}"
            color = CONSORT_COLORS["enrolled"]
        elif is_final:
            text = f"Analysis Set\nN = {stage['n']}{site_str}"
            color = CONSORT_COLORS["final"]
        else:
            text = f"{stage['name']}\nN = {stage['n']}{site_str}"
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
            excl_sites = stage.get("excl_sites", {})
            reason = stage.get("reason", "")
            site_list = ", ".join(
                f"{s}({n})" for s, n in
                sorted(excl_sites.items(), key=lambda x: -x[1])[:4]
            )
            if len(excl_sites) > 4:
                site_list += f" +{len(excl_sites)-4}"
            excl_text = f"Excluded: n = {n_excl}\n{reason}"
            if site_list:
                excl_text += f"\n{site_list}"

            _box(excl_x, y, excl_w, excl_h, excl_text,
                 CONSORT_COLORS["excluded"], CONSORT_COLORS["edge_excl"],
                 font_excl)
            _arrow_right(center_x + box_w / 2, y,
                         excl_x - excl_w / 2)

    # Panel label
    ax.set_title("A. Quality Filtering (CONSORT)",
                 fontweight="bold", fontsize=FONT["title"],
                 pad=FONT["title_pad"])


# ═══════════════════════════════════════════════════════════════════════
# Validation panel helpers
# ═══════════════════════════════════════════════════════════════════════

def _violin_box_sd(
    ax: plt.Axes,
    data: list[np.ndarray],
    positions: list[float],
    colors: list[str],
) -> None:
    """Draw violin + boxplot + mean±SD + outlier dots."""
    vp = ax.violinplot(
        data,
        positions=positions,
        showmedians=False,
        showextrema=False,
        widths=VIOLIN_WIDTH,
    )
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(VIOLIN_ALPHA)
        body.set_edgecolor("none")

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(
            marker=".",
            markersize=OUTLIER_MARKER_SIZE,
            markerfacecolor="#444444",
            markeredgecolor="none",
            alpha=OUTLIER_ALPHA,
        ),
        medianprops=dict(color="white", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(BOX_ALPHA)
        patch.set_edgecolor(colors[i])

# ═══════════════════════════════════════════════════════════════════════
# Main figure
# ═══════════════════════════════════════════════════════════════════════

def plot_figure3(data: dict, atlas: str, stages: list[dict]) -> plt.Figure:
    """Create Figure 3: ABIDE Validation with CONSORT + 4 validation panels.

    Layout:
        Row 1: [A. CONSORT flowchart — spans full width]
        Row 2: [B. Scatter] [C. Distribution] [D. Improvement] [E. Seed σ]

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
    improvement = rho - r_fc
    n = data["n"]

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        height_ratios=[1.1, 1.0],
        width_ratios=[1.2, 0.7, 0.5, 0.5],
        hspace=0.32, wspace=0.25,
    )

    # ── Row 1: CONSORT (spans all 4 columns) ──
    ax_consort = fig.add_subplot(gs[0, :])
    _draw_consort_panel(ax_consort, stages)

    # ── Row 2, Panel B: Scatter r_FC vs ρ̂T ──
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_scatter.scatter(
        r_fc, rho, s=18, c=COLOR_BSNET, alpha=0.6,
        edgecolors="white", linewidth=0.3, zorder=3,
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

    # ── Row 2, Panel C: Violin+box — Raw FC vs BS-NET ──
    ax_comp = fig.add_subplot(gs[1, 1])
    _violin_box_sd(
        ax_comp,
        data=[r_fc, rho],
        positions=[0, 1],
        colors=[COLOR_RAW, COLOR_BSNET],
    )

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

    # ── Row 2, Panel D: Violin+box — Improvement (Δ) ──
    ax_imp = fig.add_subplot(gs[1, 2])
    _violin_box_sd(
        ax_imp,
        data=[improvement],
        positions=[0],
        colors=[COLOR_IMPROVE],
    )
    ax_imp.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_imp.set_xticks([0])
    ax_imp.set_xticklabels(["Δ (ρ̂T − r_FC)"], fontsize=FONT["legend_small"])
    ax_imp.set_title(
        "D. Improvement (Δ)",
        fontweight="bold", fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )
    ax_imp.set_ylabel(r"Δ ($\hat{\rho}_T - r_{FC}$)",
                      fontsize=FONT["axis_label"])
    ax_imp.set_xlabel("")

    # ── Row 2, Panel E: Violin+box — Seed σ ──
    ax_seed = fig.add_subplot(gs[1, 3])
    _violin_box_sd(
        ax_seed,
        data=[rho_std],
        positions=[0],
        colors=[COLOR_SEED],
    )
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
    """Generate Supplementary Figure S3."""
    parser = argparse.ArgumentParser(
        description="Supplementary Figure S3: ABIDE filtered + CONSORT",
    )
    parser.add_argument("--atlas", default="cc200", choices=["cc200", "cc400"])
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Use unfiltered N=468 CSV (legacy sensitivity).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    use_filtered = not args.no_filter
    data = load_multiseed(args.atlas, filtered=use_filtered)

    r_fc = data["r_fc"]
    rho = data["rho_mean"]
    improvement = rho - r_fc

    print(f"ABIDE {args.atlas.upper()} — N={data['n']}")
    print(f"  r_FC:  {r_fc.mean():.3f} ± {r_fc.std():.3f}")
    print(f"  ρ̂T:   {rho.mean():.3f} ± {rho.std():.3f}")
    print(f"  Δ:     {improvement.mean():+.3f} ± {improvement.std():.3f}")
    print(f"  Improved: {np.sum(improvement > 0)}/{data['n']} "
          f"({np.mean(improvement > 0)*100:.1f}%)")
    print(f"  Seed σ: {data['rho_std'].mean():.4f} (mean)")

    # Compute CONSORT stages from raw subject data (includes site-level detail)
    stages = compute_consort_data(args.atlas)

    fig = plot_figure3(data, args.atlas, stages)
    save_figure(fig, "FigS3_ABIDE_Filtered_CONSORT.png")
    print("\nSupplementary Figure S3 saved: FigS3_ABIDE_Filtered_CONSORT.png")


if __name__ == "__main__":
    main()
