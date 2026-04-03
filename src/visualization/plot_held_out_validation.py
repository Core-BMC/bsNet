"""Plot held-out prediction validation results (sliding-window design).

2-segment design: A (short) + B (full remaining reference).
Ground truth = r_FC(A, B).  Ceiling = split-half r(B1, B2).

4 panels:
  A: Bar — 4-metric summary with individual subject dots
  B: Scatter — ρ̂T(SW) vs ceiling r(B1,B2), coloured by sec_B
  C: Scatter — ρ̂T(SW) vs r_FC(A,B), identity + regression
  D: Strong-FC subset (|FC_B| >= fc_thresh)

Usage:
    python src/visualization/plot_held_out_validation.py \\
        --csv data/ds000243/results/held_out_validation_sw_4s256parcels.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import apply_bsnet_theme, save_figure

# ── Color scheme (Fig 3-7 standard) ──────────────────────────────────────────
C_REF   = "#95a5a6"   # Gray  — ceiling / split-half
C_RAW   = "#fdae61"   # Amber — raw FC
C_BSNET = "#4A90E2"   # Blue  — BS-NET (SW)
C_BP    = "#7BC8A4"   # Green — BS-NET (simple bootstrap)


def load_results(csv: Path) -> pd.DataFrame:
    """Load CSV and average across seeds per subject.

    Args:
        csv: Path to CSV from run_held_out_validation.py.

    Returns:
        Per-subject mean DataFrame.
    """
    df = pd.read_csv(csv)
    seed_cols = [c for c in df.columns if c != "sub_id"]
    agg = df.groupby("sub_id")[seed_cols].mean().reset_index()
    return agg


def plot_held_out(csv: Path, fc_thresh: float = 0.20) -> None:
    """Generate 4-panel held-out validation figure.

    Args:
        csv: Path to results CSV.
        fc_thresh: |FC| threshold for strong-connection subset label.
    """
    apply_bsnet_theme()
    df = load_results(csv)
    n  = len(df)

    if n == 0:
        print("No data found in CSV.")
        return

    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.38, wspace=0.28,
        left=0.07, right=0.97, top=0.91, bottom=0.08,
    )

    # ── Panel A: 4-Metric Bar Chart ───────────────────────────────────────────
    ax_A = fig.add_subplot(gs[0, 0])

    keys   = ["r_fc_AB", "rho_hat_T_sw", "rho_hat_T_bp", "r_fc_BB"]
    labels = [
        "Raw FC\nr(A, B)",
        "BS-NET\nρ̂T (SW)",
        "BS-NET\nρ̂T (BP)",
        "Ceiling\nr(B₁, B₂)",
    ]
    colors = [C_RAW, C_BSNET, C_BP, C_REF]
    means  = [df[k].mean() for k in keys]
    sds    = [df[k].std()  for k in keys]
    x      = np.arange(len(keys))

    bars = ax_A.bar(
        x, means, color=colors, width=0.55,
        edgecolor="#333", linewidth=0.8, zorder=2,
    )
    ax_A.errorbar(x, means, yerr=sds,
                  fmt="none", color="#333", capsize=5, lw=1.2, zorder=3)

    rng = np.random.default_rng(42)
    for xi, k in enumerate(keys):
        jitter = rng.uniform(-0.12, 0.12, size=n)
        ax_A.scatter(
            np.full(n, xi) + jitter, df[k],
            s=18, color="#333333", alpha=0.5, zorder=4,
        )

    for bar_, m in zip(bars, means):
        ax_A.text(
            bar_.get_x() + bar_.get_width() / 2, m + 0.012,
            f"{m:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Ceiling reference line
    ceil_mean = df["r_fc_BB"].mean()
    ax_A.axhline(ceil_mean, color=C_REF, lw=1.2, ls="--", alpha=0.7,
                 label=f"ceiling = {ceil_mean:.3f}")

    ax_A.set_xticks(x)
    ax_A.set_xticklabels(labels, fontsize=9.5)
    ax_A.set_ylabel("Spearman ρ  (FC similarity)", fontsize=11)
    ax_A.set_ylim(0, min(1.05, max(means) + max(sds) + 0.14))
    ax_A.set_title(
        f"A.  Four-Metric Summary  (N={n})\nmean ± SD across subjects",
        fontsize=11, fontweight="bold",
    )
    ax_A.legend(fontsize=9)

    # ── Panel B: ρ̂T(SW) vs Ceiling scatter ───────────────────────────────────
    ax_B = fig.add_subplot(gs[0, 1])

    sc = ax_B.scatter(
        df["r_fc_BB"], df["rho_hat_T_sw"],
        c=df["sec_B"], cmap="YlOrRd",
        s=60, edgecolors="#333", linewidths=0.5, zorder=3,
    )
    cb = plt.colorbar(sc, ax=ax_B, fraction=0.046, pad=0.04)
    cb.set_label("B duration (s)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    lims_B = [
        min(df["r_fc_BB"].min(), df["rho_hat_T_sw"].min()) - 0.05,
        max(df["r_fc_BB"].max(), df["rho_hat_T_sw"].max()) + 0.05,
    ]
    ax_B.plot(lims_B, lims_B, "k--", lw=1.2, alpha=0.5, label="identity")
    r_B, _ = pearsonr(df["r_fc_BB"], df["rho_hat_T_sw"])
    bias_B = (df["rho_hat_T_sw"] - df["r_fc_BB"]).mean()

    ax_B.set_xlim(lims_B)
    ax_B.set_ylim(lims_B)
    ax_B.set_xlabel("Ceiling  r(B₁, B₂)", fontsize=11)
    ax_B.set_ylabel("BS-NET  ρ̂T  (sliding window)", fontsize=11)
    ax_B.set_title(
        f"B.  ρ̂T(SW) vs. Within-B Ceiling\n"
        f"Pearson r = {r_B:.3f}  |  bias = {bias_B:+.3f}",
        fontsize=11, fontweight="bold",
    )
    ax_B.legend(fontsize=9)
    ax_B.text(
        0.04, 0.96,
        f"ρ̂T {'over' if bias_B > 0 else 'under'}estimates\nceiling by {abs(bias_B):.3f}",
        transform=ax_B.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(fc="white", alpha=0.75, ec="#cccccc", pad=3),
    )

    # ── Panel C: ρ̂T(SW) vs raw r_FC(A,B) ────────────────────────────────────
    ax_C = fig.add_subplot(gs[1, 0])

    ax_C.scatter(
        df["r_fc_AB"], df["rho_hat_T_sw"],
        c=C_BSNET, s=60, edgecolors="#333", linewidths=0.5, zorder=3,
        label="SW",
    )
    ax_C.scatter(
        df["r_fc_AB"], df["rho_hat_T_bp"],
        c=C_BP, s=40, marker="^", edgecolors="#333", linewidths=0.5,
        zorder=3, alpha=0.7, label="Bootstrap",
    )

    lims_C = [
        min(df["r_fc_AB"].min(), df["rho_hat_T_sw"].min()) - 0.04,
        max(df["r_fc_AB"].max(), df["rho_hat_T_sw"].max()) + 0.04,
    ]
    ax_C.plot(lims_C, lims_C, "k--", lw=1.2, alpha=0.4, label="identity")

    coefs = np.polyfit(df["r_fc_AB"], df["rho_hat_T_sw"], 1)
    xfit  = np.linspace(lims_C[0], lims_C[1], 100)
    ax_C.plot(xfit, np.polyval(coefs, xfit),
              color=C_BSNET, lw=1.8, alpha=0.7,
              label=f"SW fit (slope={coefs[0]:.2f})")

    r_C, _ = pearsonr(df["r_fc_AB"], df["rho_hat_T_sw"])
    gain    = (df["rho_hat_T_sw"] - df["r_fc_AB"]).mean()

    ax_C.set_xlim(lims_C)
    ax_C.set_ylim(lims_C)
    ax_C.set_xlabel("Raw FC  r(A, B)", fontsize=11)
    ax_C.set_ylabel("BS-NET  ρ̂T", fontsize=11)
    ax_C.set_title(
        f"C.  Correction Gain  (ρ̂T vs Raw)\n"
        f"Pearson r = {r_C:.3f}  |  mean gain = +{gain:.3f}",
        fontsize=11, fontweight="bold",
    )
    ax_C.legend(fontsize=9, loc="upper left")

    # ── Panel D: Strong-FC subset ─────────────────────────────────────────────
    ax_D = fig.add_subplot(gs[1, 1])

    d_str = df.dropna(subset=["r_fc_AB_strong"])
    n_str = len(d_str)

    keys_D   = ["r_fc_AB_strong", "rho_hat_T_sw", "rho_hat_T_bp", "r_fc_BB"]
    labels_D = [
        f"Raw FC\n(|FC_B|≥{fc_thresh})",
        "BS-NET\nρ̂T (SW)",
        "BS-NET\nρ̂T (BP)",
        "Ceiling\nr(B₁, B₂)",
    ]
    colors_D = [C_RAW, C_BSNET, C_BP, C_REF]
    means_D  = [d_str[k].mean() for k in keys_D]
    sds_D    = [d_str[k].std()  for k in keys_D]
    x_D      = np.arange(len(keys_D))

    bars_D = ax_D.bar(
        x_D, means_D, color=colors_D, width=0.55,
        edgecolor="#333", linewidth=0.8, zorder=2,
    )
    ax_D.errorbar(x_D, means_D, yerr=sds_D,
                  fmt="none", color="#333", capsize=5, lw=1.2, zorder=3)

    rng2 = np.random.default_rng(43)
    for xi, k in enumerate(keys_D):
        vals   = d_str[k].dropna()
        jitter = rng2.uniform(-0.12, 0.12, size=len(vals))
        ax_D.scatter(
            np.full(len(vals), xi) + jitter, vals,
            s=18, color="#333333", alpha=0.5, zorder=4,
        )

    for bar_, m in zip(bars_D, means_D):
        ax_D.text(
            bar_.get_x() + bar_.get_width() / 2, m + 0.012,
            f"{m:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    avg_n_strong = int(d_str["n_strong"].mean()) if "n_strong" in d_str.columns else "?"
    ax_D.set_xticks(x_D)
    ax_D.set_xticklabels(labels_D, fontsize=9.5)
    ax_D.set_ylabel("Spearman ρ  (FC similarity)", fontsize=11)
    ax_D.set_ylim(0, min(1.05, max(means_D) + max(sds_D) + 0.14))
    ax_D.set_title(
        f"D.  Strong-Connection Subset  (|FC_B|≥{fc_thresh})\n"
        f"N={n_str} subjects, mean {avg_n_strong} pairs/subject",
        fontsize=11, fontweight="bold",
    )
    ax_D.axhline(d_str["r_fc_BB"].mean(), color=C_REF, lw=1.2, ls="--", alpha=0.7)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    atlas_tag = csv.stem.replace("held_out_validation_sw_", "").replace("held_out_validation_", "")
    sec_A_mean = df["sec_A"].mean()
    sec_B_mean = df["sec_B"].mean()
    fig.suptitle(
        f"Held-out Prediction Validation (Sliding-Window) — ds000243 XCP-D ({atlas_tag})\n"
        f"2-segment design: A={sec_A_mean:.0f}s short scan  |  B={sec_B_mean:.0f}s reference\n"
        f"SW = sliding-window BS-NET   BP = simple bootstrap BS-NET",
        fontsize=12, fontweight="bold",
    )

    out_name = f"Figure_HeldOut_SW_{atlas_tag}.png"
    save_figure(fig, out_name)
    print(f"Saved: docs/figure/{out_name}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot held-out validation figure (sliding-window design)"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/ds000243/results/held_out_validation_sw_4s256parcels.csv"),
    )
    parser.add_argument("--fc-thresh", type=float, default=0.20)
    args = parser.parse_args()
    plot_held_out(args.csv, args.fc_thresh)
