"""Generate Figure 7: ADHD vs Control classification performance comparison.

Produces publication-quality figure matching Figure 1 style standards:
  - Figure7_ADHD_Classification.png (2x2 layout)

Panel A: Accuracy bar chart (3 conditions × 2 atlases) + per-fold scatter
Panel B: AUC bar chart (3 conditions × 2 atlases) + per-fold scatter
Panel C: Δ Accuracy relative to Raw FC (BS-NET gain & Reference gap)
Panel D: Summary text panel

Data source:
  - data/adhd/results/adhd_classification_cc200.csv
  - data/adhd/results/adhd_classification_cc200_folds.csv
  - data/adhd/results/adhd_classification_cc400.csv
  - data/adhd/results/adhd_classification_cc400_folds.csv
"""

from __future__ import annotations

import csv as csv_mod
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import (
    ATLAS_PALETTE,
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

RESULTS_DIR = Path("data/adhd/results")

# Condition display order and colors
CONDITION_ORDER = ["raw_short", "bsnet", "reference"]
CONDITION_LABELS = {
    "raw_short": "Raw FC\n(2 min)",
    "bsnet": "BS-NET\n(2 min)",
    "reference": "Reference FC\n(full scan)",
}
CONDITION_COLORS = {
    "raw_short": PALETTE["raw"],
    "bsnet": PALETTE["bsnet"],
    "reference": PALETTE["true"],
}


# ============================================================================
# Data loading
# ============================================================================

def load_classification_results(atlas: str) -> dict[str, dict]:
    """Load classification summary CSV for an atlas.

    Args:
        atlas: Atlas name (cc200 or cc400).

    Returns:
        Dict keyed by condition with acc/auc mean/std.
    """
    csv_path = RESULTS_DIR / f"adhd_classification_{atlas}.csv"
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    results = {}
    for r in rows:
        results[r["condition"]] = {
            "label": r["label"],
            "n_features": int(r["n_features"]),
            "acc_mean": float(r["acc_mean"]),
            "acc_std": float(r["acc_std"]),
            "auc_mean": float(r["auc_mean"]),
            "auc_std": float(r["auc_std"]),
        }
    return results


def load_fold_results(atlas: str) -> dict[str, dict[str, np.ndarray]]:
    """Load per-fold classification results CSV.

    Args:
        atlas: Atlas name (cc200 or cc400).

    Returns:
        Dict keyed by condition, each with 'accuracy' and 'auc' arrays.
    """
    folds_path = RESULTS_DIR / f"adhd_classification_{atlas}_folds.csv"
    if not folds_path.exists():
        return {}

    with open(folds_path) as f:
        rows = list(csv_mod.DictReader(f))

    folds: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        cond = r["condition"]
        if cond not in folds:
            folds[cond] = {"accuracy": [], "auc": []}
        folds[cond]["accuracy"].append(float(r["accuracy"]))
        folds[cond]["auc"].append(float(r["auc"]))

    return {
        c: {"accuracy": np.array(v["accuracy"]), "auc": np.array(v["auc"])}
        for c, v in folds.items()
    }


# ============================================================================
# Scatter helpers
# ============================================================================

def _add_fold_scatter(
    ax: plt.Axes,
    x_center: float,
    values: np.ndarray,
    color: str,
    jitter_width: float = 0.08,
    seed: int = 0,
) -> None:
    """Overlay jittered scatter points for per-fold values on a bar.

    Args:
        ax: Target axes.
        x_center: X position of the bar center.
        values: 1D array of per-fold metric values.
        color: Scatter point face color.
        jitter_width: Half-width of uniform jitter.
        seed: Random seed for reproducible jitter.
    """
    rng = np.random.RandomState(seed)
    jitter = rng.uniform(-jitter_width, jitter_width, size=len(values))
    ax.scatter(
        x_center + jitter,
        values,
        s=MARKER["scatter_small"],
        facecolors=color,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.45,
        zorder=5,
    )


# ============================================================================
# Plot
# ============================================================================

def plot_classification_figure() -> None:
    """Generate 2x2 classification comparison figure.

    Panels:
      A. Accuracy by condition (grouped bars + per-fold scatter)
      B. AUC by condition (grouped bars + per-fold scatter)
      C. Δ Accuracy relative to Raw FC (BS-NET gain & Reference gap)
      D. Summary text panel
    """
    # Load data
    datasets: dict[str, dict[str, dict]] = {}
    fold_data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for atlas in ("cc200", "cc400"):
        csv_path = RESULTS_DIR / f"adhd_classification_{atlas}.csv"
        if csv_path.exists():
            datasets[atlas] = load_classification_results(atlas)
            fold_data[atlas] = load_fold_results(atlas)

    if not datasets:
        logger.error("No classification result CSVs found.")
        return

    atlases = list(datasets.keys())
    n_cond = len(CONDITION_ORDER)
    has_folds = any(bool(fd) for fd in fold_data.values())

    apply_bsnet_theme()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE["2x2"])

    # ── Panel A: Accuracy grouped bars + scatter ──
    ax = axes[0, 0]
    x = np.arange(n_cond)
    w = 0.30
    for i, atlas in enumerate(atlases):
        d = datasets[atlas]
        fd = fold_data.get(atlas, {})
        means = [d[c]["acc_mean"] for c in CONDITION_ORDER]
        stds = [d[c]["acc_std"] for c in CONDITION_ORDER]
        offset = (i - 0.5) * w
        bars = ax.bar(
            x + offset, means, w, yerr=stds,
            label=atlas.upper(),
            color=ATLAS_PALETTE[atlas],
            edgecolor="white", linewidth=0.8,
            capsize=4, error_kw={"linewidth": LINE["error"]},
            alpha=0.85,
        )
        # Scatter overlay
        if has_folds:
            for j, c in enumerate(CONDITION_ORDER):
                if c in fd:
                    _add_fold_scatter(
                        ax, x[j] + offset, fd[c]["accuracy"],
                        color=ATLAS_PALETTE[atlas],
                        seed=i * 100 + j,
                    )
        # Value labels — placed above error bar cap
        for bar, m, s in zip(bars, means, stds):
            y_label = m + s + 0.015
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_label,
                f"{m:.3f}",
                ha="center", va="bottom",
                fontsize=FONT["legend_small"], fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS[c] for c in CONDITION_ORDER],
        fontsize=FONT["tick"],
    )
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Chance")
    style_axis(
        ax,
        title="A. Classification Accuracy",
        ylabel="Accuracy (5-fold CV × 5 repeats)",
        legend_loc="upper right",
        legend_fontsize=FONT["legend_small"],
    )

    # ── Panel B: AUC grouped bars + scatter ──
    ax = axes[0, 1]
    for i, atlas in enumerate(atlases):
        d = datasets[atlas]
        fd = fold_data.get(atlas, {})
        means = [d[c]["auc_mean"] for c in CONDITION_ORDER]
        stds = [d[c]["auc_std"] for c in CONDITION_ORDER]
        offset = (i - 0.5) * w
        bars = ax.bar(
            x + offset, means, w, yerr=stds,
            label=atlas.upper(),
            color=ATLAS_PALETTE[atlas],
            edgecolor="white", linewidth=0.8,
            capsize=4, error_kw={"linewidth": LINE["error"]},
            alpha=0.85,
        )
        # Scatter overlay
        if has_folds:
            for j, c in enumerate(CONDITION_ORDER):
                if c in fd:
                    _add_fold_scatter(
                        ax, x[j] + offset, fd[c]["auc"],
                        color=ATLAS_PALETTE[atlas],
                        seed=i * 200 + j,
                    )
        # Value labels above error bar cap
        for bar, m, s in zip(bars, means, stds):
            y_label = m + s + 0.015
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_label,
                f"{m:.3f}",
                ha="center", va="bottom",
                fontsize=FONT["legend_small"], fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS[c] for c in CONDITION_ORDER],
        fontsize=FONT["tick"],
    )
    ax.set_ylim(0.2, 1.1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Chance")
    style_axis(
        ax,
        title="B. Area Under ROC Curve (AUC)",
        ylabel="AUC",
        legend_loc="upper right",
        legend_fontsize=FONT["legend_small"],
    )

    # ── Panel C: Δ Accuracy relative to Raw FC ──
    ax = axes[1, 0]
    delta_conditions = ["bsnet", "reference"]
    delta_labels = ["BS-NET\nvs Raw FC", "Reference\nvs Raw FC"]
    x2 = np.arange(len(delta_conditions))

    for i, atlas in enumerate(atlases):
        d = datasets[atlas]
        raw_acc = d["raw_short"]["acc_mean"]
        deltas = [d[c]["acc_mean"] - raw_acc for c in delta_conditions]
        offset = (i - 0.5) * w
        bars = ax.bar(
            x2 + offset, deltas, w,
            label=atlas.upper(),
            color=ATLAS_PALETTE[atlas],
            edgecolor="white", linewidth=0.8,
            alpha=0.85,
        )
        for bar, delta in zip(bars, deltas):
            if delta >= 0:
                y_pos = delta + 0.005
                va = "bottom"
            else:
                y_pos = delta - 0.005
                va = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_pos,
                f"{delta:+.3f}",
                ha="center", va=va,
                fontsize=FONT["annotation"], fontweight="bold",
            )

    ax.set_xticks(x2)
    ax.set_xticklabels(delta_labels, fontsize=FONT["tick"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-0.12, 0.06)
    style_axis(
        ax,
        title="C. Accuracy Change vs Raw FC",
        ylabel="Δ Accuracy",
        legend_loc="lower left",
        legend_fontsize=FONT["legend_small"],
    )

    # ── Panel D: Summary table ──
    ax = axes[1, 1]
    ax.axis("off")

    lines = [
        "── Track H: ADHD vs Control Classification ──\n",
        "Classifier:   Linear SVM (C=1.0)",
        "CV:           Stratified 5-fold × 5 repeats",
        "Labels:       ADHD (n=20) vs Control (n=20)\n",
    ]

    for atlas in atlases:
        d = datasets[atlas]
        n_feat = d["raw_short"]["n_features"]
        lines.append(f"── {atlas.upper()} ({n_feat:,} features) ──")
        for c in CONDITION_ORDER:
            r = d[c]
            tag = ""
            if c == "bsnet":
                delta = r["acc_mean"] - d["raw_short"]["acc_mean"]
                tag = f"  (Δ={delta:+.1%})"
            lines.append(
                f"  {CONDITION_LABELS[c].replace(chr(10), ' '):<22}"
                f"Acc={r['acc_mean']:.3f}  AUC={r['auc_mean']:.3f}{tag}"
            )
        lines.append("")

    lines.append("Key: BS-NET ≥ Raw FC in both atlases")
    lines.append("     Reference FC < Raw FC (see Discussion)")

    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10.5, fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=PALETTE["ci_fill"],
            alpha=0.3,
            edgecolor=PALETTE["true"],
        ),
    )
    ax.set_title(
        "D. Summary",
        fontweight=FONT["title_weight"],
        fontsize=FONT["title"],
        pad=FONT["title_pad"],
    )

    plt.tight_layout(pad=3.0)
    save_figure(fig, "Figure7_ADHD_Classification.png")
    logger.info("Figure7_ADHD_Classification saved.")


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
