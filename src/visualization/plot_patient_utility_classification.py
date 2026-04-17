"""Plot reliability-aware supervised classification utility figure (ADHD-200 PCP).

Reads:
  data/adhd/pcp/results/adhd200_reliability_classification_summary*.csv

Writes:
  docs/figure/FigS7_Reliability_Aware_Classification.png
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import FONT, apply_bsnet_theme, save_figure

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY = Path(
    "data/adhd/pcp/results/"
    "adhd200_reliability_classification_summary_loso_nobal_nocov_perm1000_repeat10.csv",
)
FALLBACK_SUMMARY = Path("data/adhd/pcp/results/adhd200_reliability_classification_summary.csv")

STRATUM_ORDER = ["all", "T1_low", "T2_mid", "T3_high"]
STRATUM_LABEL = {
    "all": "All",
    "T1_low": "T1 (Low ρ̂T)",
    "T2_mid": "T2 (Mid ρ̂T)",
    "T3_high": "T3 (High ρ̂T)",
}

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


def _to_float(v: str | None) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def _resolve_input(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    if DEFAULT_SUMMARY.exists():
        return DEFAULT_SUMMARY
    return FALLBACK_SUMMARY


def _load_summary(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    out: list[dict] = []
    for r in rows:
        out.append({
            "stratum": r["stratum"],
            "eval_scheme": r.get("eval_scheme", ""),
            "fc_method": r["fc_method"],
            "model": r["model"],
            "bal_acc_mean": _to_float(r.get("bal_acc_mean")),
            "bal_acc_std": _to_float(r.get("bal_acc_std")),
            "roc_auc_mean": _to_float(r.get("roc_auc_mean")),
            "roc_auc_std": _to_float(r.get("roc_auc_std")),
            "bal_acc_p_perm_mean": _to_float(r.get("bal_acc_p_perm_mean")),
            "roc_auc_p_perm_mean": _to_float(r.get("roc_auc_p_perm_mean")),
            "n_subjects_analysis_mean": _to_float(r.get("n_subjects_analysis_mean")),
            "rho_mean_analysis_mean": _to_float(r.get("rho_mean_analysis_mean")),
        })
    return out


def _select_primary(rows: list[dict], eval_scheme: str, fc_method: str, model: str) -> list[dict]:
    return [
        r for r in rows
        if r["eval_scheme"] == eval_scheme
        and r["fc_method"] == fc_method
        and r["model"] == model
        and r["stratum"] in STRATUM_ORDER
    ]


def _select_by_model(rows: list[dict], eval_scheme: str, model: str) -> list[dict]:
    return [
        r for r in rows
        if r["eval_scheme"] == eval_scheme
        and r["model"] == model
        and r["stratum"] in STRATUM_ORDER
        and r["fc_method"] in METHOD_ORDER
    ]


def _plot_primary_metric(
    ax: plt.Axes,
    rows: list[dict],
    metric_mean: str,
    metric_std: str,
    p_key: str,
    title: str,
    ylabel: str,
) -> None:
    by_s = {r["stratum"]: r for r in rows}
    x = np.arange(len(STRATUM_ORDER))
    y = [by_s[s][metric_mean] if s in by_s else np.nan for s in STRATUM_ORDER]
    yerr = [by_s[s][metric_std] if s in by_s else np.nan for s in STRATUM_ORDER]

    ax.bar(
        x,
        y,
        width=0.62,
        color="#4A90E2",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="none",
        ecolor="#333333",
        elinewidth=1.0,
        capsize=2.8,
        capthick=1.0,
        zorder=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([STRATUM_LABEL[s] for s in STRATUM_ORDER], fontsize=FONT["tick"])
    ax.set_ylabel(ylabel, fontsize=FONT["axis_label"])
    ax.set_title(title, fontsize=FONT["title"], fontweight="bold", pad=FONT["title_pad"])
    ax.grid(axis="y", alpha=0.25)

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    y_text = y_max - 0.06 * y_span
    for i, s in enumerate(STRATUM_ORDER):
        if s not in by_s:
            continue
        p = by_s[s].get(p_key, float("nan"))
        if np.isnan(p):
            txt = "p=N/A"
            color = "#666666"
        else:
            txt = f"p={p:.3f}"
            color = "#666666" if p >= 0.05 else "#b2182b"
        ax.text(
            i,
            y_text,
            txt,
            ha="center",
            va="top",
            fontsize=FONT["legend_small"] - 0.5,
            color=color,
        )


def _plot_method_heatmap(ax: plt.Axes, rows: list[dict], metric_key: str, title: str) -> None:
    mat = np.full((len(STRATUM_ORDER), len(METHOD_ORDER)), np.nan, dtype=float)
    for i, s in enumerate(STRATUM_ORDER):
        for j, m in enumerate(METHOD_ORDER):
            cand = [r for r in rows if r["stratum"] == s and r["fc_method"] == m]
            if cand:
                mat[i, j] = cand[0][metric_key]

    vmin = np.nanmin(mat) if np.isfinite(np.nanmin(mat)) else 0.0
    vmax = np.nanmax(mat) if np.isfinite(np.nanmax(mat)) else 1.0
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], fontsize=FONT["tick"])
    ax.set_yticks(np.arange(len(STRATUM_ORDER)))
    ax.set_yticklabels([STRATUM_LABEL[s] for s in STRATUM_ORDER], fontsize=FONT["tick"])
    ax.set_title(title, fontsize=FONT["title"], fontweight="bold", pad=FONT["title_pad"])

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                continue
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=FONT["legend_small"])

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_pvals(ax: plt.Axes, rows: list[dict]) -> None:
    x = np.arange(len(STRATUM_ORDER))
    width = 0.32
    by_s = {r["stratum"]: r for r in rows}
    p_bal = [by_s[s]["bal_acc_p_perm_mean"] if s in by_s else np.nan for s in STRATUM_ORDER]
    p_auc = [by_s[s]["roc_auc_p_perm_mean"] if s in by_s else np.nan for s in STRATUM_ORDER]

    ax.bar(
        x - width / 2,
        p_bal,
        width=width,
        color="#fdae61",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.85,
        label="p(BalAcc)",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        p_auc,
        width=width,
        color="#95a5a6",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.85,
        label="p(AUC)",
        zorder=3,
    )

    ax.axhline(0.05, color="#b2182b", linestyle="--", linewidth=1.2, alpha=0.9, zorder=4)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([STRATUM_LABEL[s] for s in STRATUM_ORDER], fontsize=FONT["tick"])
    ax.set_ylabel("Permutation p-value", fontsize=FONT["axis_label"])
    ax.set_title("D. Primary Permutation p-values", fontsize=FONT["title"], fontweight="bold", pad=FONT["title_pad"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=FONT["legend_small"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot supervised reliability-aware classification utility figure.")
    parser.add_argument("--summary-csv", type=str, default=None, help="Summary CSV path.")
    parser.add_argument("--eval-scheme", choices=["loso", "stratified_kfold"], default="loso")
    parser.add_argument("--primary-fc", choices=METHOD_ORDER, default="tangent")
    parser.add_argument("--primary-model", choices=["logistic_l2", "linear_svm"], default="logistic_l2")
    parser.add_argument(
        "--output-name",
        type=str,
        default="FigS7_Reliability_Aware_Classification.png",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary_csv = _resolve_input(args.summary_csv)
    if not summary_csv.exists():
        logger.error(f"Missing summary CSV: {summary_csv}")
        return

    rows = _load_summary(summary_csv)
    primary_rows = _select_primary(
        rows,
        eval_scheme=args.eval_scheme,
        fc_method=args.primary_fc,
        model=args.primary_model,
    )
    method_rows = _select_by_model(
        rows,
        eval_scheme=args.eval_scheme,
        model=args.primary_model,
    )
    if not primary_rows:
        logger.error("No primary rows found for requested settings.")
        return

    apply_bsnet_theme()
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    _plot_primary_metric(
        ax1,
        primary_rows,
        metric_mean="bal_acc_mean",
        metric_std="bal_acc_std",
        p_key="bal_acc_p_perm_mean",
        title="A. Primary Balanced Accuracy (LOSO)",
        ylabel="Balanced Accuracy",
    )
    _plot_primary_metric(
        ax2,
        primary_rows,
        metric_mean="roc_auc_mean",
        metric_std="roc_auc_std",
        p_key="roc_auc_p_perm_mean",
        title="B. Primary AUC (LOSO)",
        ylabel="ROC-AUC",
    )
    _plot_method_heatmap(
        ax3,
        method_rows,
        metric_key="bal_acc_mean",
        title="C. BalAcc Heatmap (Model fixed, FC methods)",
    )
    _plot_pvals(ax4, primary_rows)

    fig.text(
        0.5,
        0.50,
        "EXPLORATORY (Discrimination signal is modest)",
        ha="center",
        va="center",
        fontsize=18,
        color="#444444",
        alpha=0.10,
        rotation=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        (
            "Interpretation guardrail: treat this as reliability-aware exploratory utility evidence "
            "(not a standalone clinical biomarker claim). BalAcc is primary; AUC is supportive."
        ),
        ha="center",
        va="center",
        fontsize=FONT["legend_small"],
        color="#444444",
    )

    save_figure(fig, args.output_name)
    logger.info(f"Input: {summary_csv}")
    logger.info(f"Saved: docs/figure/{args.output_name}")


if __name__ == "__main__":
    main()
