"""Plot held-out prediction validation results.

Generates a 4-panel figure showing whether BS-NET's ρ̂T accurately predicts
held-out FC performance:

  Panel A: Paired scatter — r_FC(A,C) vs ρ̂T, coloured by r_FC(B,C) ceiling
  Panel B: Bar chart — mean of 4 metrics with individual subject dots
  Panel C: Scatter — ρ̂T vs r_FC(A, B+C) [ground truth], identity line
  Panel D: Strong-FC subset comparison (|FC_C| >= fc_thresh)

Usage:
    python src/visualization/plot_held_out_validation.py \\
        --csv data/ds000243/results/held_out_validation_4s256parcels.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import apply_bsnet_theme, save_figure

# ── Color scheme (Fig 3-7 standard) ──────────────────────────────────────────
C_REF   = "#95a5a6"   # Gray — reference / ceiling
C_RAW   = "#fdae61"   # Amber — raw FC
C_BSNET = "#4A90E2"   # Blue — BS-NET
C_FULL  = "#2c7bb6"   # Dark blue — ground truth (more data)
C_CI    = "#abd9e9"   # CI fill

BAR_LABELS = {
    "r_fc_AC":   "Raw FC\n(A→C)",
    "rho_hat_T": "BS-NET\nρ̂T",
    "r_fc_BC":   "Ceiling\n(B→C)",
    "r_fc_full": "Ground truth\n(A→B+C)",
}
BAR_COLORS = [C_RAW, C_BSNET, C_REF, C_FULL]


def load_results(csv: Path) -> pd.DataFrame:
    """Load and aggregate held-out validation CSV.

    Args:
        csv: Path to CSV produced by run_held_out_validation.py.

    Returns:
        DataFrame averaged across seeds per subject.
    """
    df = pd.read_csv(csv)
    cols = ["r_fc_AC", "rho_hat_T", "r_fc_BC", "r_fc_full",
            "r_fc_AC_strong", "r_fc_full_strong", "ci_lower", "ci_upper",
            "n_strong", "sec_A", "sec_B", "sec_C"]
    cols = [c for c in cols if c in df.columns]
    agg = df.groupby("sub_id")[cols].mean().reset_index()
    return agg


def _fmt(v: float | None) -> str:
    """Format float safely.

    Args:
        v: Value to format.

    Returns:
        Formatted string or 'N/A'.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.3f}"


def plot_held_out(csv: Path, fc_thresh: float = 0.20) -> None:
    """Generate 4-panel held-out validation figure.

    Args:
        csv: Path to results CSV.
        fc_thresh: |FC| threshold used for strong-connection subset label.
    """
    apply_bsnet_theme()
    df = load_results(csv)
    n = len(df)

    if n == 0:
        print("No data found in CSV.")
        return

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.38, wspace=0.28,
        left=0.07, right=0.97, top=0.92, bottom=0.08,
    )

    # ── Panel A: Paired scatter ───────────────────────────────────────────────
    ax_A = fig.add_subplot(gs[0, 0])

    sc = ax_A.scatter(
        df["r_fc_AC"], df["rho_hat_T"],
        c=df["r_fc_BC"], cmap="Blues",
        vmin=0.3, vmax=1.0,
        s=60, edgecolors="#333", linewidths=0.5, zorder=3,
    )
    cb = plt.colorbar(sc, ax=ax_A, fraction=0.046, pad=0.04)
    cb.set_label("Ceiling r(B,C)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    lims = [
        min(df["r_fc_AC"].min(), df["rho_hat_T"].min()) - 0.05,
        max(df["r_fc_AC"].max(), df["rho_hat_T"].max()) + 0.05,
    ]
    ax_A.plot(lims, lims, "k--", lw=1.0, alpha=0.5, label="identity")
    # Horizontal reference: mean ρ̂T
    ax_A.axhline(df["rho_hat_T"].mean(), color=C_BSNET, lw=1.2, ls=":", alpha=0.7)
    ax_A.axvline(df["r_fc_AC"].mean(),  color=C_RAW,   lw=1.2, ls=":", alpha=0.7)

    r_A, _ = pearsonr(df["r_fc_AC"], df["rho_hat_T"])
    ax_A.set_xlim(lims)
    ax_A.set_ylim(lims)
    ax_A.set_xlabel("Raw FC correlation  r(A, C)", fontsize=11)
    ax_A.set_ylabel("BS-NET  ρ̂T(A→C)", fontsize=11)
    ax_A.set_title(
        f"A.  Raw vs. BS-NET by Subject  (N={n})\n"
        f"Pearson r = {r_A:.3f}",
        fontsize=11, fontweight="bold",
    )
    ax_A.legend(fontsize=9, loc="upper left")

    # ── Panel B: Bar chart of 4 metrics ──────────────────────────────────────
    ax_B = fig.add_subplot(gs[0, 1])

    keys   = ["r_fc_AC", "rho_hat_T", "r_fc_BC", "r_fc_full"]
    means  = [df[k].mean() for k in keys]
    sds    = [df[k].std()  for k in keys]
    labels = [BAR_LABELS[k] for k in keys]
    x      = np.arange(len(keys))

    bars = ax_B.bar(
        x, means, color=BAR_COLORS,
        width=0.55, edgecolor="#333", linewidth=0.8, zorder=2,
    )
    ax_B.errorbar(
        x, means, yerr=sds,
        fmt="none", color="#333", capsize=5, linewidth=1.2, zorder=3,
    )

    # Individual subject dots
    rng = np.random.default_rng(42)
    for xi, k in enumerate(keys):
        jitter = rng.uniform(-0.12, 0.12, size=n)
        ax_B.scatter(
            np.full(n, xi) + jitter, df[k],
            s=18, color="#333333", alpha=0.5, zorder=4,
        )

    # Value labels on bars
    for bar_, mean_ in zip(bars, means):
        ax_B.text(
            bar_.get_x() + bar_.get_width() / 2,
            mean_ + 0.012,
            f"{mean_:.3f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )

    ax_B.set_xticks(x)
    ax_B.set_xticklabels(labels, fontsize=9.5)
    ax_B.set_ylabel("Spearman ρ (FC similarity)", fontsize=11)
    ax_B.set_ylim(0, min(1.05, max(means) + max(sds) + 0.12))
    ax_B.set_title(
        "B.  Four-Metric Summary\n"
        "mean ± SD across subjects",
        fontsize=11, fontweight="bold",
    )
    ax_B.axhline(df["r_fc_BC"].mean(), color=C_REF, lw=1.0, ls="--", alpha=0.6,
                 label=f"ceiling mean = {df['r_fc_BC'].mean():.3f}")
    ax_B.legend(fontsize=9)

    # ── Panel C: ρ̂T vs ground truth ──────────────────────────────────────────
    ax_C = fig.add_subplot(gs[1, 0])

    ax_C.scatter(
        df["r_fc_full"], df["rho_hat_T"],
        c=C_BSNET, s=60, edgecolors="#333", linewidths=0.5, zorder=3,
    )

    # Identity line
    lims_C = [
        min(df["r_fc_full"].min(), df["rho_hat_T"].min()) - 0.04,
        max(df["r_fc_full"].max(), df["rho_hat_T"].max()) + 0.04,
    ]
    ax_C.plot(lims_C, lims_C, "k--", lw=1.2, alpha=0.5, label="identity")

    # Linear regression line
    from numpy.polynomial.polynomial import polyfit as poly1d_fit
    coefs = np.polyfit(df["r_fc_full"], df["rho_hat_T"], 1)
    xfit  = np.linspace(lims_C[0], lims_C[1], 100)
    ax_C.plot(xfit, np.polyval(coefs, xfit), color=C_BSNET, lw=1.8, alpha=0.7,
              label=f"fit (slope={coefs[0]:.2f})")

    r_C, _ = pearsonr(df["r_fc_full"], df["rho_hat_T"])
    bias    = (df["rho_hat_T"] - df["r_fc_full"]).mean()

    ax_C.set_xlim(lims_C)
    ax_C.set_ylim(lims_C)
    ax_C.set_xlabel("Ground truth  r(A, B+C)", fontsize=11)
    ax_C.set_ylabel("BS-NET  ρ̂T", fontsize=11)
    ax_C.set_title(
        f"C.  BS-NET vs. Ground Truth\n"
        f"Pearson r = {r_C:.3f}  |  bias = {bias:+.3f}",
        fontsize=11, fontweight="bold",
    )
    ax_C.legend(fontsize=9)

    # Annotate bias arrow
    ax_C.axhline(0, color="none")  # placeholder for spacing
    ax_C.text(
        0.04, 0.96,
        f"ρ̂T {'overestimates' if bias > 0 else 'underestimates'}\nby {abs(bias):.3f} on avg",
        transform=ax_C.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(fc="white", alpha=0.75, ec="#cccccc", pad=3),
    )

    # ── Panel D: Strong-FC subset ─────────────────────────────────────────────
    ax_D = fig.add_subplot(gs[1, 1])

    d_strong = df.dropna(subset=["r_fc_AC_strong", "r_fc_full_strong"])
    n_strong_sub = len(d_strong)

    keys_D   = ["r_fc_AC_strong", "rho_hat_T", "r_fc_BC", "r_fc_full_strong"]
    # Note: rho_hat_T is the same (BS-NET runs on all pairs; stratify only for evaluation)
    labels_D = [
        f"Raw FC\n(strong, |FC|≥{fc_thresh})",
        "BS-NET\nρ̂T (all pairs)",
        "Ceiling\n(B→C, all)",
        f"Ground truth\n(strong, |FC|≥{fc_thresh})",
    ]
    means_D  = [d_strong[k].mean() for k in keys_D]
    sds_D    = [d_strong[k].std()  for k in keys_D]
    x_D      = np.arange(len(keys_D))

    bars_D = ax_D.bar(
        x_D, means_D,
        color=[C_RAW, C_BSNET, C_REF, C_FULL],
        width=0.55, edgecolor="#333", linewidth=0.8, zorder=2,
    )
    ax_D.errorbar(
        x_D, means_D, yerr=sds_D,
        fmt="none", color="#333", capsize=5, linewidth=1.2, zorder=3,
    )

    rng2 = np.random.default_rng(43)
    for xi, k in enumerate(keys_D):
        jitter = rng2.uniform(-0.12, 0.12, size=n_strong_sub)
        ax_D.scatter(
            np.full(n_strong_sub, xi) + jitter, d_strong[k],
            s=18, color="#333333", alpha=0.5, zorder=4,
        )

    for bar_, mean_ in zip(bars_D, means_D):
        ax_D.text(
            bar_.get_x() + bar_.get_width() / 2,
            mean_ + 0.012,
            f"{mean_:.3f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )

    ax_D.set_xticks(x_D)
    ax_D.set_xticklabels(labels_D, fontsize=8.5)
    ax_D.set_ylabel("Spearman ρ (FC similarity)", fontsize=11)
    ax_D.set_ylim(0, min(1.05, max(means_D) + max(sds_D) + 0.12))
    avg_n_strong = int(d_strong["n_strong"].mean()) if "n_strong" in d_strong.columns else "?"
    ax_D.set_title(
        f"D.  Strong-Connection Subset  (|FC|≥{fc_thresh})\n"
        f"N={n_strong_sub} subjects, mean {avg_n_strong} pairs/subject",
        fontsize=11, fontweight="bold",
    )

    # ── Suptitle ──────────────────────────────────────────────────────────────
    atlas_tag = csv.stem.replace("held_out_validation_", "")
    fig.suptitle(
        f"Held-out Prediction Validation — ds000243 XCP-D ({atlas_tag})\n"
        f"3-segment design: A={df['sec_A'].mean():.0f}s short | "
        f"B={df['sec_B'].mean():.0f}s reference | "
        f"C={df['sec_C'].mean():.0f}s held-out",
        fontsize=13, fontweight="bold",
    )

    out_name = f"Figure_HeldOut_{atlas_tag}.png"
    out = save_figure(fig, out_name)
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot held-out prediction validation figure"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/ds000243/results/held_out_validation_4s256parcels.csv"),
    )
    parser.add_argument("--fc-thresh", type=float, default=0.20)
    args = parser.parse_args()

    plot_held_out(args.csv, args.fc_thresh)
