"""Plot held-out prediction validation results (sliding-window design).

2-segment design: A (short) + B (full remaining reference).
Ground truth = r_FC(A, B).  Ceiling = split-half r(B1, B2).

4 panels:
  A: Bar — 4-metric summary with individual subject dots
  B: Scatter — rho_hat_T(SW) vs ceiling r(B1,B2), coloured by sec_B
  C: Scatter — rho_hat_T(SW) vs r_FC(A,B), identity + regression
  D: Strong-FC subset bar (|FC_B| >= fc_thresh)

Style: matches Figure 1 (FONT_PANEL / FONT_AXIS / FONT_TICK / CONDITION_PALETTE).

Usage:
    python src/visualization/plot_held_out_validation.py \\
        --csv data/ds000243/results/held_out_validation_sw_4s256parcels.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import (
    CONDITION_PALETTE,
    FONT_AXIS,
    FONT_PANEL,
    FONT_TICK,
    apply_bsnet_theme,
    save_figure,
)

warnings.filterwarnings("ignore")

# ── Colors (Fig 3-7 standard) ─────────────────────────────────────────────────
C_RAW   = CONDITION_PALETTE["raw"]        # Amber  #fdae61
C_BSNET = CONDITION_PALETTE["bsnet"]      # Blue   #4A90E2
C_REF   = CONDITION_PALETTE["reference"]  # Gray   #95a5a6
C_BP    = "#7BC8A4"                       # Green — simple bootstrap

# Darker dot variants for overlay jitter (bar background is light)
C_RAW_D   = "#c8700a"   # Dark amber
C_BSNET_D = "#1a5fa8"   # Dark blue
C_REF_D   = "#4a5a5b"   # Dark gray
C_BP_D    = "#2e8a5e"   # Dark green

DOT_COLORS = {
    C_RAW:   C_RAW_D,
    C_BSNET: C_BSNET_D,
    C_BP:    C_BP_D,
    C_REF:   C_REF_D,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(csv: Path) -> pd.DataFrame:
    """Load CSV and average across seeds per subject.

    Args:
        csv: Path to CSV from run_held_out_validation.py.

    Returns:
        Per-subject mean DataFrame.
    """
    df = pd.read_csv(csv)
    seed_cols = [c for c in df.columns if c != "sub_id"]
    return df.groupby("sub_id")[seed_cols].mean().reset_index()


# ── Panel helpers ─────────────────────────────────────────────────────────────

def _dot_jitter(
    ax: plt.Axes,
    x_center: float,
    values: pd.Series,
    color: str = "#333333",
    seed: int = 42,
) -> None:
    """Overlay individual subject dots with jitter.

    Args:
        ax: Target axes.
        x_center: Bar x-position.
        values: Per-subject values.
        color: Dot color (should match bar color).
        seed: Random seed for jitter.
    """
    rng = np.random.default_rng(seed)
    vals = values.dropna().values
    jitter = rng.uniform(-0.14, 0.14, size=len(vals))
    ax.scatter(
        np.full(len(vals), x_center) + jitter, vals,
        s=18, color=color, alpha=0.55,
        edgecolors="none", zorder=5,
    )


def _bar_label(
    ax: plt.Axes,
    bar: plt.Rectangle,
    mean: float,
    sd: float = 0.0,
    color: str = "#444444",
) -> None:
    """Annotate bar top with value, using element color + thin arrow.

    Text is placed at mean + sd + margin (above error bar cap),
    with a "-" arrowstyle line pointing down to the bar top.
    Matches Figure 1 Panel A/C annotation style.

    Args:
        ax: Target axes.
        bar: Bar rectangle.
        mean: Mean value (arrow tip position).
        sd: SD value — text is offset to mean + sd + 0.015.
        color: Text and arrow color (should match bar color).
    """
    x_bar = bar.get_x() + bar.get_width() / 2
    ax.annotate(
        f"{mean:.3f}",
        xy=(x_bar, mean),
        xytext=(x_bar, mean + sd + 0.015),
        ha="center", va="bottom",
        fontsize=8, color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.9),
    )


# ── Panel A: 4-metric bar chart ───────────────────────────────────────────────

def panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Four-metric summary bar chart with subject dots.

    Args:
        ax: Target axes.
        df: Per-subject aggregated DataFrame.
    """
    keys   = ["r_fc_AB", "rho_hat_T_sw", "rho_hat_T_bp", "r_fc_BB"]
    labels = ["Raw FC\nr(A, B)", "BS-NET\nrho_T (SW)", "BS-NET\nrho_T (BP)", "Ceiling\nr(B1, B2)"]
    colors = [C_RAW, C_BSNET, C_BP, C_REF]
    means  = [df[k].mean() for k in keys]
    sds    = [df[k].std()  for k in keys]
    x      = np.arange(len(keys))
    n      = len(df)

    bars = ax.bar(
        x, means, color=colors, width=0.55,
        edgecolor="white", linewidth=0.9, alpha=0.88, zorder=3,
    )
    ax.errorbar(x, means, yerr=sds,
                fmt="none", ecolor="#444444", elinewidth=1.3,
                capsize=4, zorder=4)

    for xi, (k, c) in enumerate(zip(keys, colors)):
        _dot_jitter(ax, xi, df[k], color=DOT_COLORS.get(c, c), seed=xi)

    for bar_, m, s, c in zip(bars, means, sds, colors):
        _bar_label(ax, bar_, m, sd=s, color=DOT_COLORS.get(c, c))

    # Ceiling reference line
    ceil_mean = df["r_fc_BB"].mean()
    ax.axhline(ceil_mean, color=C_REF, lw=1.5, ls="--", alpha=0.75,
               label=f"ceiling = {ceil_mean:.3f}", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_ylabel("Spearman r  (FC similarity)", **FONT_AXIS)
    ax.set_ylim(0, min(1.05, max(means) + max(sds) + 0.22))
    ax.set_title(
        f"A.  Four-Metric Summary  (N={n})\nmean +/- SD across subjects",
        **FONT_PANEL,
    )
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.88)


# ── Panel B: rho_T(SW) vs ceiling scatter ─────────────────────────────────────

def panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Scatter of BS-NET(SW) vs within-B ceiling, coloured by B duration.

    Args:
        ax: Target axes.
        df: Per-subject aggregated DataFrame.
    """
    sc = ax.scatter(
        df["r_fc_BB"], df["rho_hat_T_sw"],
        c=df["sec_B"], cmap="YlOrRd",
        s=55, edgecolors="#555555", linewidths=0.6, alpha=0.80, zorder=3,
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("B duration (s)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    lo = min(df["r_fc_BB"].min(), df["rho_hat_T_sw"].min()) - 0.06
    hi = max(df["r_fc_BB"].max(), df["rho_hat_T_sw"].max()) + 0.06
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.45, label="identity")

    r_val, _ = pearsonr(df["r_fc_BB"], df["rho_hat_T_sw"])
    bias      = (df["rho_hat_T_sw"] - df["r_fc_BB"]).mean()

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.set_xlabel("Ceiling  r(B1, B2)", **FONT_AXIS)
    ax.set_ylabel("BS-NET  rho_T  (sliding window)", **FONT_AXIS)
    ax.set_title(
        f"B.  rho_T(SW) vs. Within-B Ceiling\n"
        f"Pearson r = {r_val:.3f}  |  bias = {bias:+.3f}",
        **FONT_PANEL,
    )
    ax.text(
        0.04, 0.97,
        f"rho_T {'over' if bias > 0 else 'under'}estimates\nceiling by {abs(bias):.3f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.5, color="#444444",
        bbox=dict(fc="white", alpha=0.80, ec="#cccccc", pad=3),
    )
    ax.legend(fontsize=8.5, framealpha=0.88)


# ── Panel C: correction gain ───────────────────────────────────────────────────

def panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Scatter of rho_T vs raw r_FC(A,B) — shows correction gain.

    Args:
        ax: Target axes.
        df: Per-subject aggregated DataFrame.
    """
    ax.scatter(
        df["r_fc_AB"], df["rho_hat_T_sw"],
        c=C_BSNET_D, s=55, edgecolors="none", alpha=0.75,
        zorder=4, label="SW",
    )
    ax.scatter(
        df["r_fc_AB"], df["rho_hat_T_bp"],
        c=C_BP_D, s=38, marker="^", edgecolors="none",
        zorder=3, alpha=0.65, label="Bootstrap",
    )

    lo = min(df["r_fc_AB"].min(), df["rho_hat_T_sw"].min()) - 0.05
    hi = max(df["r_fc_AB"].max(), df["rho_hat_T_sw"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.45, label="identity")

    coefs = np.polyfit(df["r_fc_AB"], df["rho_hat_T_sw"], 1)
    xfit  = np.linspace(lo, hi, 100)
    ax.plot(xfit, np.polyval(coefs, xfit),
            color=C_BSNET, lw=1.8, alpha=0.75,
            label=f"SW fit (slope={coefs[0]:.2f})")

    r_val, _ = pearsonr(df["r_fc_AB"], df["rho_hat_T_sw"])
    gain      = (df["rho_hat_T_sw"] - df["r_fc_AB"]).mean()

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.set_xlabel("Raw FC  r(A, B)", **FONT_AXIS)
    ax.set_ylabel("BS-NET  rho_T", **FONT_AXIS)
    ax.set_title(
        f"C.  Correction Gain  (rho_T vs Raw)\n"
        f"Pearson r = {r_val:.3f}  |  mean gain = +{gain:.3f}",
        **FONT_PANEL,
    )
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.88)


# ── Panel D: strong-FC subset ─────────────────────────────────────────────────

def panel_d(ax: plt.Axes, df: pd.DataFrame, fc_thresh: float) -> None:
    """Strong-connection subset bar chart.

    Args:
        ax: Target axes.
        df: Per-subject aggregated DataFrame.
        fc_thresh: |FC| threshold used in experiment.
    """
    d_str  = df.dropna(subset=["r_fc_AB_strong"])
    n_str  = len(d_str)
    keys_D = ["r_fc_AB_strong", "rho_hat_T_sw", "rho_hat_T_bp", "r_fc_BB"]
    labels_D = [
        f"Raw FC\n(|FC_B|>={fc_thresh})",
        "BS-NET\nrho_T (SW)",
        "BS-NET\nrho_T (BP)",
        "Ceiling\nr(B1, B2)",
    ]
    colors_D = [C_RAW, C_BSNET, C_BP, C_REF]
    means_D  = [d_str[k].mean() for k in keys_D]
    sds_D    = [d_str[k].std()  for k in keys_D]
    x_D      = np.arange(len(keys_D))

    bars_D = ax.bar(
        x_D, means_D, color=colors_D, width=0.55,
        edgecolor="white", linewidth=0.9, alpha=0.88, zorder=3,
    )
    ax.errorbar(x_D, means_D, yerr=sds_D,
                fmt="none", ecolor="#444444", elinewidth=1.3,
                capsize=4, zorder=4)

    for xi, (k, c) in enumerate(zip(keys_D, colors_D)):
        _dot_jitter(ax, xi, d_str[k], color=DOT_COLORS.get(c, c), seed=xi + 10)

    for bar_, m, s, c in zip(bars_D, means_D, sds_D, colors_D):
        _bar_label(ax, bar_, m, sd=s, color=DOT_COLORS.get(c, c))

    ax.axhline(d_str["r_fc_BB"].mean(), color=C_REF, lw=1.5, ls="--",
               alpha=0.75, zorder=2)

    avg_n_strong = int(d_str["n_strong"].mean()) if "n_strong" in d_str.columns else "?"
    ax.set_xticks(x_D)
    ax.set_xticklabels(labels_D, fontsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_ylabel("Spearman r  (FC similarity)", **FONT_AXIS)
    ax.set_ylim(0, min(1.05, max(means_D) + max(sds_D) + 0.22))
    ax.set_title(
        f"D.  Strong-Connection Subset  (|FC_B|>={fc_thresh})\n"
        f"N={n_str} subjects, mean {avg_n_strong} pairs/subject",
        **FONT_PANEL,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_held_out(csv: Path, fc_thresh: float = 0.20) -> None:
    """Generate 4-panel held-out validation figure.

    Args:
        csv: Path to results CSV from run_held_out_validation.py.
        fc_thresh: |FC| threshold used for strong-connection subset.
    """
    apply_bsnet_theme()
    df = load_results(csv)
    n  = len(df)

    if n == 0:
        print("No data found in CSV.")
        return

    # ── Layout: 2×2, similar proportions to Figure 1 ─────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.42, wspace=0.32,
        left=0.07, right=0.97, top=0.90, bottom=0.08,
    )

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    panel_a(ax_A, df)
    panel_b(ax_B, df)
    panel_c(ax_C, df)
    panel_d(ax_D, df, fc_thresh)

    atlas_tag  = csv.stem.replace("held_out_validation_sw_", "").replace("held_out_validation_", "")
    sec_A_mean = df["sec_A"].mean()
    sec_B_mean = df["sec_B"].mean()

    fig.suptitle(
        f"Held-out Prediction Validation (Sliding-Window) — ds000243 XCP-D ({atlas_tag})\n"
        f"2-segment design:  A = {sec_A_mean:.0f}s short scan  |  "
        f"B = {sec_B_mean:.0f}s reference  "
        f"(SW = sliding-window BS-NET,  BP = simple bootstrap BS-NET)",
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
