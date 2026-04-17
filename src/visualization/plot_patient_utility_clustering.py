"""Plot reliability-aware clustering utility figure for ADHD-200 PCP.

Reads:
  data/adhd/pcp/results/adhd200_reliability_clustering_summary.csv

Writes:
  docs/figure/FigS6_Reliability_Aware_Clustering.png
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.visualization.style import FONT, apply_bsnet_theme, save_figure

logger = logging.getLogger(__name__)

SUMMARY_CSV = Path("data/adhd/pcp/results/adhd200_reliability_clustering_summary.csv")

METHOD_ORDER = ["correlation", "partial correlation", "tangent"]
METHOD_LABEL = {
    "correlation": "Correlation",
    "partial correlation": "Partial",
    "tangent": "Tangent",
}
METHOD_COLOR = {
    "correlation": "#fdae61",
    "partial correlation": "#4A90E2",
    "tangent": "#95a5a6",
}

STRATUM_ORDER = ["T1_low", "T2_mid", "T3_high"]
STRATUM_LABEL = {
    "T1_low": "T1 (Low ρ̂T)",
    "T2_mid": "T2 (Mid ρ̂T)",
    "T3_high": "T3 (High ρ̂T)",
}


def _load_summary() -> list[dict]:
    def _to_float(v: str | None) -> float:
        if v is None or v == "":
            return float("nan")
        try:
            return float(v)
        except ValueError:
            return float("nan")

    with open(SUMMARY_CSV) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            "stratum": r["stratum"],
            "fc_method": r["fc_method"],
            "algorithm": r["algorithm"],
            "n_subjects": int(r["n_subjects"]),
            "rho_mean": float(r["rho_mean"]),
            "ari_mean": float(r["ari_mean"]),
            "ari_std": float(r["ari_std"]),
            "bal_acc_mean": float(r["bal_acc_mean"]),
            "bal_acc_std": float(r["bal_acc_std"]),
            "ari_p_perm_mean": _to_float(r.get("ari_p_perm_mean")),
            "bal_acc_p_perm_mean": _to_float(r.get("bal_acc_p_perm_mean")),
        })
    return out


def _select(rows: list[dict], algorithm: str) -> list[dict]:
    return [r for r in rows if r["algorithm"] == algorithm and r["stratum"] in STRATUM_ORDER]


def _plot_grouped_metric(
    ax: plt.Axes,
    rows: list[dict],
    metric_mean: str,
    metric_std: str,
    title: str,
    ylabel: str,
) -> None:
    group_x = np.arange(len(STRATUM_ORDER))
    width = 0.24
    offsets = [-width, 0.0, width]

    for i, method in enumerate(METHOD_ORDER):
        method_rows = {r["stratum"]: r for r in rows if r["fc_method"] == method}
        y = [method_rows[s][metric_mean] if s in method_rows else np.nan for s in STRATUM_ORDER]
        yerr = [method_rows[s][metric_std] if s in method_rows else np.nan for s in STRATUM_ORDER]

        ax.bar(
            group_x + offsets[i],
            y,
            width=width * 0.92,
            color=METHOD_COLOR[method],
            alpha=0.80,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.errorbar(
            group_x + offsets[i],
            y,
            yerr=yerr,
            fmt="none",
            ecolor="#333333",
            elinewidth=1.0,
            capsize=2.8,
            capthick=1.0,
            zorder=4,
        )

    ax.set_xticks(group_x)
    ax.set_xticklabels([STRATUM_LABEL[s] for s in STRATUM_ORDER], fontsize=FONT["tick"])
    ax.set_ylabel(ylabel, fontsize=FONT["axis_label"])
    ax.set_title(title, fontsize=FONT["title"], fontweight="bold", pad=FONT["title_pad"])
    ax.grid(axis="y", alpha=0.25)


def _annotate_best_p(
    ax: plt.Axes,
    rows: list[dict],
    metric_mean: str,
    p_key: str,
) -> None:
    """Annotate best-method permutation p-value per stratum."""
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    y_text = y_max - 0.06 * y_span

    for i, stratum in enumerate(STRATUM_ORDER):
        cands = [r for r in rows if r["stratum"] == stratum]
        if not cands:
            continue
        best = max(cands, key=lambda r: r[metric_mean])
        p = best.get(p_key, float("nan"))
        if np.isnan(p):
            txt = "p_perm=N/A"
            color = "#666666"
        else:
            txt = f"best p={p:.3f}"
            color = "#666666" if p >= 0.05 else "#b2182b"
        ax.text(
            i, y_text, txt,
            ha="center", va="top",
            fontsize=FONT["legend_small"] - 0.5,
            color=color,
        )


def _plot_heatmap(ax: plt.Axes, rows: list[dict]) -> None:
    mat = np.full((len(STRATUM_ORDER), len(METHOD_ORDER)), np.nan, dtype=float)
    for i, s in enumerate(STRATUM_ORDER):
        for j, m in enumerate(METHOD_ORDER):
            cand = [r for r in rows if r["stratum"] == s and r["fc_method"] == m]
            if cand:
                mat[i, j] = cand[0]["ari_mean"]

    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=FONT["tick"])
    ax.set_yticks(np.arange(len(STRATUM_ORDER)))
    ax.set_yticklabels([STRATUM_LABEL[s] for s in STRATUM_ORDER], fontsize=FONT["tick"])
    ax.set_title("C. ARI Heatmap (KMeans)", fontsize=FONT["title"], fontweight="bold", pad=FONT["title_pad"])

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                continue
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=FONT["legend_small"])

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not SUMMARY_CSV.exists():
        logger.error(f"Missing summary CSV: {SUMMARY_CSV}")
        return

    rows = _load_summary()
    kmeans_rows = _select(rows, algorithm="kmeans")

    apply_bsnet_theme()
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    _plot_grouped_metric(
        ax1,
        kmeans_rows,
        metric_mean="ari_mean",
        metric_std="ari_std",
        title="A. HC/Patients Alignment (ARI, KMeans)",
        ylabel="Adjusted Rand Index",
    )
    _plot_grouped_metric(
        ax2,
        kmeans_rows,
        metric_mean="bal_acc_mean",
        metric_std="bal_acc_std",
        title="B. Label Recovery (Balanced Accuracy, KMeans)",
        ylabel="Balanced Accuracy",
    )
    _plot_heatmap(ax3, kmeans_rows)

    _annotate_best_p(ax1, kmeans_rows, metric_mean="ari_mean", p_key="ari_p_perm_mean")
    _annotate_best_p(ax2, kmeans_rows, metric_mean="bal_acc_mean", p_key="bal_acc_p_perm_mean")

    legend_handles = [
        Patch(facecolor=METHOD_COLOR[m], edgecolor="white", label=METHOD_LABEL[m], alpha=0.80)
        for m in METHOD_ORDER
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=FONT["legend_small"])

    # Non-significance guardrail: watermark + caption
    fig.text(
        0.5,
        0.51,
        "EXPLORATORY (No robust significant separation)",
        ha="center",
        va="center",
        fontsize=18,
        color="#444444",
        alpha=0.10,
        rotation=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        (
            "Interpretation guardrail: permutation tests (KMeans) do not support a strong HC/Patients "
            "separation claim. Use as exploratory reliability-stratified trend evidence only."
        ),
        ha="center",
        va="center",
        fontsize=FONT["legend_small"],
        color="#444444",
    )

    save_figure(fig, "FigS6_Reliability_Aware_Clustering.png")
    logger.info("Saved: docs/figure/FigS6_Reliability_Aware_Clustering.png")


if __name__ == "__main__":
    main()
