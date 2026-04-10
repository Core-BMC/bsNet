"""Generate Figure 7: ADHD vs Control Classification (Option C — 1×2 bar).

Two-panel layout:
  A. Classification Accuracy — grouped bar (3 conditions × 2 atlases) + seed dots
  B. Area Under ROC Curve   — grouped bar (3 conditions × 2 atlases) + seed dots

Style: Fig 3–6 통일 색상 스키마, chance line, red diamond mean±SD
Color: Gray (#95a5a6, Reference) + Amber (#fdae61, Raw) + Blue (#4A90E2, BS-NET)

Data: data/adhd/results/adhd_classification_{atlas}.csv + _seeds.csv
Output: Figure7_ADHD_Classification.png
"""

from __future__ import annotations

import csv as csv_mod
import logging
import sys
from pathlib import Path

# Allow direct script execution:
#   python src/visualization/plot_figure7_classification.py
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import (
    CONDITION_PALETTE,
    FONT,
    apply_bsnet_theme,
    save_figure,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/adhd/results")

# Condition order & colors (Gray/Amber/Blue — Fig 3–6 schema, from style.py)
CONDITION_ORDER = ["raw_short", "bsnet", "reference"]
CONDITION_LABELS = ["Raw FC\n(2 min)", "BS-NET\n(2 min)", "Reference\n(full)"]
CONDITION_COLORS = [
    CONDITION_PALETTE["raw"],        # Amber
    CONDITION_PALETTE["bsnet"],      # Blue
    CONDITION_PALETTE["reference"],  # Gray
]

# Atlas hatching for visual separation within grouped bars
ATLAS_HATCHES = {"cc200": None, "cc400": "///"}
ATLAS_EDGE_ALPHA = 0.9

DOT_SIZE = 22
DOT_ALPHA = 0.55
BAR_ALPHA = 0.75
BAR_WIDTH = 0.28
GROUP_GAP = 0.35


# ── Data loaders ──────────────────────────────────────────────────────────


def _load_summary(atlas: str) -> dict[str, dict]:
    csv_path = RESULTS_DIR / f"adhd_classification_{atlas}.csv"
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))
    results = {}
    for r in rows:
        entry = {
            "acc_mean": float(r["acc_mean"]),
            "acc_std": float(r["acc_std"]),
            "auc_mean": float(r["auc_mean"]),
            "auc_std": float(r["auc_std"]),
            "n_features": int(r["n_features"]),
        }
        if "n_seeds" in r and r["n_seeds"]:
            entry["n_seeds"] = int(r["n_seeds"])
            entry["acc_seed_std"] = float(r["acc_seed_std"])
            entry["auc_seed_std"] = float(r["auc_seed_std"])
        results[r["condition"]] = entry
    return results


def _load_seeds(atlas: str) -> dict[str, dict[str, np.ndarray]]:
    seeds_path = RESULTS_DIR / f"adhd_classification_{atlas}_seeds.csv"
    if not seeds_path.exists():
        return {}
    with open(seeds_path) as f:
        rows = list(csv_mod.DictReader(f))
    seeds: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        cond = r["condition"]
        if cond not in seeds:
            seeds[cond] = {"acc": [], "auc": []}
        seeds[cond]["acc"].append(float(r["acc_mean"]))
        seeds[cond]["auc"].append(float(r["auc_mean"]))
    return {
        c: {"acc": np.array(v["acc"]), "auc": np.array(v["auc"])}
        for c, v in seeds.items()
    }


# ── Plot helpers ──────────────────────────────────────────────────────────


def _draw_panel(
    ax: plt.Axes,
    metric: str,
    ylabel: str,
    title: str,
    summaries: dict[str, dict[str, dict]],
    seeds: dict[str, dict[str, dict[str, np.ndarray]]],
    atlases: list[str],
) -> None:
    """Draw grouped bar + seed dot overlay for one metric."""
    rng = np.random.RandomState(42)
    n_cond = len(CONDITION_ORDER)
    n_atlas = len(atlases)
    x_base = np.arange(n_cond) * GROUP_GAP * (n_atlas + 1)

    for j, atlas in enumerate(atlases):
        summary = summaries[atlas]
        seed_data = seeds.get(atlas, {})
        x_positions = x_base + j * (BAR_WIDTH + 0.01)

        for i, cond in enumerate(CONDITION_ORDER):
            entry = summary.get(cond)
            if entry is None:
                continue

            mean_val = entry[f"{metric}_mean"]
            # Prefer seed SD (cross-seed stability) over fold SD
            seed_std_key = f"{metric}_seed_std"
            std_val = entry.get(seed_std_key, entry[f"{metric}_std"])
            color = CONDITION_COLORS[i]
            hatch = ATLAS_HATCHES[atlas]

            # Bar
            ax.bar(
                x_positions[i], mean_val, BAR_WIDTH,
                color=color, alpha=BAR_ALPHA,
                edgecolor="white", linewidth=0.8,
                hatch=hatch, zorder=3,
            )

            # Seed dots overlay
            sd = seed_data.get(cond, {}).get(metric)
            if sd is not None and len(sd) > 0:
                x_jit = rng.normal(0, BAR_WIDTH * 0.15, len(sd))
                ax.scatter(
                    x_positions[i] + x_jit, sd,
                    s=DOT_SIZE, c="#333333", alpha=DOT_ALPHA,
                    edgecolors="none", zorder=5,
                )

            # Red diamond mean±SD
            ax.errorbar(
                x_positions[i], mean_val, yerr=std_val,
                fmt="D", color="red", markersize=5,
                markeredgecolor="darkred", markeredgewidth=0.8,
                ecolor="darkred", elinewidth=1.0, capsize=2.5, capthick=1.0,
                zorder=15,
            )

    # Chance line
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    # x-axis: condition labels centered across atlas pairs
    x_centers = x_base + (n_atlas - 1) * (BAR_WIDTH + 0.01) / 2
    ax.set_xticks(x_centers)
    ax.set_xticklabels(CONDITION_LABELS, fontsize=FONT["tick"])
    ax.set_ylabel(ylabel, fontsize=FONT["axis_label"])
    ax.set_xlabel("")
    ax.set_title(title, fontweight="bold", fontsize=FONT["title"])

    # y-axis range (set per-panel by caller)

    # Atlas legend (hatching distinction)
    from matplotlib.patches import Patch
    legend_handles = []
    for atlas in atlases:
        legend_handles.append(Patch(
            facecolor="#cccccc", alpha=0.6,
            hatch=ATLAS_HATCHES[atlas],
            edgecolor="#666666", linewidth=0.5,
            label=atlas.upper(),
        ))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=FONT["legend_small"])


# ── Main ──────────────────────────────────────────────────────────────────


def plot_classification_figure() -> None:
    """Generate 1×2 classification comparison figure."""
    summaries: dict[str, dict[str, dict]] = {}
    seeds: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for atlas in ("cc200", "cc400"):
        csv_path = RESULTS_DIR / f"adhd_classification_{atlas}.csv"
        if csv_path.exists():
            summaries[atlas] = _load_summary(atlas)
            seeds[atlas] = _load_seeds(atlas)

    if not summaries:
        logger.error("No classification CSVs found.")
        return

    atlases = list(summaries.keys())

    apply_bsnet_theme()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"wspace": 0.25},
        constrained_layout=True,
    )

    _draw_panel(ax1, "acc", "Accuracy", "A. Classification Accuracy",
                summaries, seeds, atlases)
    ax1.set_ylim(0.5, 0.8)

    _draw_panel(ax2, "auc", "AUC", "B. Area Under ROC Curve",
                summaries, seeds, atlases)
    ax2.set_ylim(0.2, 1.0)

    save_figure(fig, "Figure7_ADHD_Classification.png")

    # Print summary
    for atlas in atlases:
        s = summaries[atlas]
        print(f"\n{atlas.upper()} ({s['raw_short']['n_features']:,} features):")
        for cond in CONDITION_ORDER:
            e = s[cond]
            delta = ""
            if cond == "bsnet":
                d = e["acc_mean"] - s["raw_short"]["acc_mean"]
                delta = f"  Δ={d:+.1%}"
            print(f"  {cond:<12} Acc={e['acc_mean']:.3f}±{e['acc_std']:.3f}"
                  f"  AUC={e['auc_mean']:.3f}±{e['auc_std']:.3f}{delta}")


def main() -> None:
    """Generate Figure 7."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    plot_classification_figure()


if __name__ == "__main__":
    main()
