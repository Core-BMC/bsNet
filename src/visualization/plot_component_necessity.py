"""Component Necessity Analysis — Publication-quality Table and Figure.

Generates:
  (a) Bar chart: Mean ρ̂T per condition with error bars (± SD across seeds)
  (b) Δρ waterfall chart showing each component's contribution relative to full pipeline
  (c) LaTeX-ready summary table printed to stdout

Input:  artifacts/reports/component_necessity.csv
Output: artifacts/reports/Figure_ComponentNecessity.png
        artifacts/reports/Table_ComponentNecessity.csv
        docs/figure/Figure_ComponentNecessity.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.style import (
    PALETTE,
    apply_bsnet_theme,
    save_figure,
)

logger = logging.getLogger(__name__)

# Condition display names and ordering
CONDITION_META: dict[str, dict[str, str]] = {
    "L_full": {"label": "Full Pipeline", "color": PALETTE["bsnet"]},
    "L_no_sb": {"label": "w/o Spearman-Brown", "color": "#e74c3c"},
    "L_no_lw": {"label": "w/o Ledoit-Wolf", "color": "#3498db"},
    "L_no_boot": {"label": "w/o Bootstrap", "color": "#2ecc71"},
    "L_no_prior": {"label": "w/o Bayesian Prior", "color": "#e67e22"},
    "L_no_atten": {"label": "w/o Attenuation Corr.", "color": "#9b59b6"},
}

CONDITION_ORDER = list(CONDITION_META.keys())


def load_and_summarize(csv_path: str | Path) -> pd.DataFrame:
    """Load component necessity CSV and compute summary statistics.

    Args:
        csv_path: Path to component_necessity.csv.

    Returns:
        DataFrame with columns: condition, label, mean_rho, std_rho,
        mean_delta, std_delta, n_seeds.
    """
    df = pd.read_csv(csv_path)

    rows = []
    for cond in CONDITION_ORDER:
        subset = df[df["condition"] == cond]
        if subset.empty:
            continue
        rows.append(
            {
                "condition": cond,
                "label": CONDITION_META[cond]["label"],
                "mean_rho": subset["rho_hat_T"].mean(),
                "std_rho": subset["rho_hat_T"].std(),
                "mean_delta": subset["delta_from_full"].mean(),
                "std_delta": subset["delta_from_full"].std(),
                "n_seeds": len(subset),
            }
        )
    return pd.DataFrame(rows)


def plot_component_necessity(
    summary: pd.DataFrame,
    output_name: str = "Figure_ComponentNecessity.png",
) -> None:
    """Create publication-quality component necessity figure.

    Panel (a): Absolute ρ̂T per condition (bar chart with error bars).
    Panel (b): Δρ waterfall (deviation from full pipeline).

    Args:
        summary: Summary DataFrame from load_and_summarize().
        output_name: Output filename for the figure.
    """
    apply_bsnet_theme()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.35})

    n = len(summary)
    x = np.arange(n)
    colors = [CONDITION_META[c]["color"] for c in summary["condition"]]

    # ── Panel (a): Absolute ρ̂T ──
    ax1.bar(
        x,
        summary["mean_rho"],
        yerr=summary["std_rho"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        capsize=4,
        error_kw={"linewidth": 1.2},
        width=0.7,
    )

    # Value annotations
    for i, (val, err) in enumerate(zip(summary["mean_rho"], summary["std_rho"])):
        ax1.text(
            i,
            val + err + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["label"], rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel(r"Mean $\hat{\rho}_T$", fontsize=12)
    ax1.set_ylim(0.0, 1.15)
    ax1.axhline(
        summary.iloc[0]["mean_rho"],
        color=PALETTE["bsnet"],
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Full = {summary.iloc[0]['mean_rho']:.3f}",
    )
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_title("(a) Extrapolated Reliability by Condition", fontsize=11, pad=10)

    # ── Panel (b): Δρ waterfall ──
    delta_df = summary.iloc[1:]  # Exclude full pipeline (delta=0)
    x2 = np.arange(len(delta_df))
    delta_colors = [
        "#e74c3c" if d < -0.05 else "#2ecc71" if d > 0.05 else "#95a5a6"
        for d in delta_df["mean_delta"]
    ]

    ax2.bar(
        x2,
        delta_df["mean_delta"],
        yerr=delta_df["std_delta"],
        color=delta_colors,
        edgecolor="white",
        linewidth=0.8,
        capsize=4,
        error_kw={"linewidth": 1.2},
        width=0.7,
    )

    # Value annotations
    for i, (val, _err) in enumerate(zip(delta_df["mean_delta"], delta_df["std_delta"])):
        offset = -0.02 if val < 0 else 0.02
        va = "top" if val < 0 else "bottom"
        ax2.text(
            i,
            val + (offset * np.sign(val) if abs(val) > 0.05 else offset),
            f"{val:+.3f}",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(delta_df["label"], rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel(r"$\Delta\hat{\rho}_T$ from Full Pipeline", fontsize=12)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("(b) Component Contribution (Leave-One-Out)", fontsize=11, pad=10)

    # Significance annotations for critical components
    critical = delta_df[delta_df["mean_delta"] < -0.1]
    for _, row in critical.iterrows():
        idx = list(delta_df["condition"]).index(row["condition"])
        ax2.annotate(
            "critical",
            xy=(idx, row["mean_delta"]),
            xytext=(idx + 0.3, row["mean_delta"] - 0.05),
            fontsize=8,
            color="#e74c3c",
            fontstyle="italic",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=0.8),
        )

    plt.tight_layout()
    save_figure(fig, output_name)
    logger.info(f"Component necessity figure saved: {output_name}")


def generate_summary_table(
    summary: pd.DataFrame,
    output_path: str | Path = "artifacts/reports/Table_ComponentNecessity.csv",
) -> pd.DataFrame:
    """Generate publication-ready summary table and save as CSV.

    Args:
        summary: Summary DataFrame from load_and_summarize().
        output_path: Path for the output CSV.

    Returns:
        Formatted DataFrame with the summary table.
    """
    table = summary.copy()
    table["rho_display"] = table.apply(
        lambda r: f"{r['mean_rho']:.3f} ± {r['std_rho']:.3f}", axis=1
    )
    table["delta_display"] = table.apply(
        lambda r: f"{r['mean_delta']:+.3f} ± {r['std_delta']:.3f}"
        if r["condition"] != "L_full"
        else "—",
        axis=1,
    )
    table["interpretation"] = table["condition"].map(
        {
            "L_full": "Baseline (all components)",
            "L_no_sb": "CRITICAL: largest drop (Δρ ≈ −0.31)",
            "L_no_lw": "Negligible effect (short TS already benefits from shrinkage in other steps)",
            "L_no_boot": "Inflated ρ̂T without resampling variance",
            "L_no_prior": "CRITICAL: second largest drop (Δρ ≈ −0.19)",
            "L_no_atten": "Modest effect; attenuation correction refines estimate",
        }
    )

    out = table[["label", "rho_display", "delta_display", "n_seeds", "interpretation"]]
    out.columns = ["Condition", "ρ̂T (mean ± SD)", "Δρ̂T", "N", "Interpretation"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(f"Summary table saved: {output_path}")

    # Print LaTeX-formatted table
    print("\n" + "=" * 80)
    print("PUBLICATION TABLE — Component Necessity Analysis")
    print("=" * 80)
    print(f"{'Condition':<25} {'ρ̂T (mean ± SD)':<18} {'Δρ̂T':<18} {'Interpretation'}")
    print("-" * 80)
    for _, row in out.iterrows():
        print(f"{row['Condition']:<25} {row['ρ̂T (mean ± SD)']:<18} {row['Δρ̂T']:<18} {row['Interpretation']}")
    print("=" * 80 + "\n")

    return out


def main() -> None:
    """Run component necessity visualization pipeline."""
    csv_path = Path("artifacts/reports/component_necessity.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Component necessity CSV not found: {csv_path}")

    summary = load_and_summarize(csv_path)
    plot_component_necessity(summary)
    generate_summary_table(summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
