"""Generate Figure 1: Duration sweep results from ds007535 (SpeechHemi).

4-panel figure using real rsfMRI data (N=12, task-residual FC, Schaefer 200):
  A - ρ̂T (BS-NET) and r_FC (Raw) vs. duration (mean ± SD across subjects×seeds)
  B - Improvement Δ = ρ̂T − r_FC vs. duration (bar chart, diminishing returns)
  C - 95% CI width decay vs. duration (uncertainty shrinkage)
  D - Per-subject ρ̂T trajectories (consistency across N=12)

Color scheme (CONDITION_PALETTE):
  Amber #fdae61 = Raw FC
  Blue  #4A90E2 = BS-NET (ρ̂T)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import (
    CONDITION_PALETTE,
    apply_bsnet_theme,
    save_figure,
)

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

CSV_PATH = Path("data/ds007535/results/ds007535_duration_sweep_schaefer200.csv")
OUTPUT_NAME = "Figure1_ds007535_DurationSweep.png"

C_RAW = CONDITION_PALETTE["raw"]    # Amber
C_BSNET = CONDITION_PALETTE["bsnet"]  # Blue

FONT_PANEL = dict(fontsize=13, fontweight="bold")
FONT_AXIS = dict(fontsize=11)
FONT_TICK = 10


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_and_aggregate(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sweep CSV and aggregate across seeds.

    Args:
        csv_path: Path to per-record CSV from run_duration_sweep.py.

    Returns:
        (df_raw, df_agg): raw DataFrame and per-duration aggregated DataFrame.
            df_agg columns: duration_sec, rfc_mean, rfc_sd, rho_mean, rho_sd,
                            delta_mean, delta_sd, ciw_mean, ciw_sd
    """
    df = pd.read_csv(csv_path)

    # Per-subject × per-seed × per-duration → aggregate across seeds first,
    # then across subjects (two-stage: matches paper statistics)
    # Stage 1: mean across seeds per (sub, duration)
    sub_dur = (
        df.groupby(["sub_id", "duration_sec"])
        .agg(
            r_fc_raw=("r_fc_raw", "mean"),
            rho_hat_T=("rho_hat_T", "mean"),
            ci_lower=("ci_lower", "mean"),
            ci_upper=("ci_upper", "mean"),
        )
        .reset_index()
    )
    sub_dur["ci_width"] = sub_dur["ci_upper"] - sub_dur["ci_lower"]
    sub_dur["improvement"] = sub_dur["rho_hat_T"] - sub_dur["r_fc_raw"]

    # Stage 2: mean ± SD across subjects per duration
    agg = (
        sub_dur.groupby("duration_sec")
        .agg(
            rfc_mean=("r_fc_raw", "mean"),
            rfc_sd=("r_fc_raw", "std"),
            rho_mean=("rho_hat_T", "mean"),
            rho_sd=("rho_hat_T", "std"),
            delta_mean=("improvement", "mean"),
            delta_sd=("improvement", "std"),
            ciw_mean=("ci_width", "mean"),
            ciw_sd=("ci_width", "std"),
        )
        .reset_index()
    )

    return sub_dur, agg


def _plot_panel_a(
    ax: plt.Axes,
    agg: pd.DataFrame,
) -> None:
    """Panel A: ρ̂T and r_FC vs. duration."""
    dur = agg["duration_sec"]

    # Raw FC (Amber)
    ax.plot(dur, agg["rfc_mean"], color=C_RAW, lw=2.5, marker="o",
            markersize=7, label=r"$r_{\mathrm{FC}}$ (Raw)", zorder=3)
    ax.fill_between(
        dur,
        agg["rfc_mean"] - agg["rfc_sd"],
        agg["rfc_mean"] + agg["rfc_sd"],
        color=C_RAW, alpha=0.20, zorder=2,
    )

    # BS-NET ρ̂T (Blue)
    ax.plot(dur, agg["rho_mean"], color=C_BSNET, lw=2.5, marker="s",
            markersize=7, label=r"$\hat{\rho}_T$ (BS-NET)", zorder=3)
    ax.fill_between(
        dur,
        agg["rho_mean"] - agg["rho_sd"],
        agg["rho_mean"] + agg["rho_sd"],
        color=C_BSNET, alpha=0.20, zorder=2,
    )

    # Vertical guide: 2-min mark
    ax.axvline(x=120, color="#888888", lw=1.5, ls="--", alpha=0.7,
               label="2 min (clinical target)")

    ax.set_xlabel("Scan Duration (s)", **FONT_AXIS)
    ax.set_ylabel("Correlation with Reference FC", **FONT_AXIS)
    ax.set_title("A.  Prediction Accuracy vs. Duration",
                 loc="left", **FONT_PANEL)
    ax.set_xlim(0, 480)
    ax.set_ylim(0.30, 1.02)
    ax.set_xticks([30, 60, 90, 120, 180, 240, 300, 450])
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=10, framealpha=0.9)


def _plot_panel_b(
    ax: plt.Axes,
    agg: pd.DataFrame,
) -> None:
    """Panel B: Improvement Δ = ρ̂T − r_FC vs. duration."""
    dur = agg["duration_sec"]
    delta = agg["delta_mean"]
    delta_sd = agg["delta_sd"]

    bars = ax.bar(
        dur, delta,
        width=22, color=C_BSNET, alpha=0.75, edgecolor="none",
        zorder=3,
    )
    ax.errorbar(
        dur, delta, yerr=delta_sd,
        fmt="none", ecolor="#333333", elinewidth=1.2, capsize=4,
        zorder=4,
    )
    ax.axhline(y=0, color="#888888", lw=1.2, ls="-", alpha=0.6)

    # Annotate bars with Δ values
    for bar, d in zip(bars, delta):
        if d >= 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                d + 0.005,
                f"+{d:.3f}",
                ha="center", va="bottom", fontsize=8.5, color="#333333",
            )

    ax.set_xlabel("Scan Duration (s)", **FONT_AXIS)
    ax.set_ylabel(r"$\hat{\rho}_T - r_{\mathrm{FC}}$ (Improvement)", **FONT_AXIS)
    ax.set_title("B.  BS-NET Improvement over Raw FC",
                 loc="left", **FONT_PANEL)
    ax.set_xlim(0, 480)
    ax.set_xticks([30, 60, 90, 120, 180, 240, 300, 450])
    ax.tick_params(labelsize=FONT_TICK)


def _plot_panel_c(
    ax: plt.Axes,
    agg: pd.DataFrame,
) -> None:
    """Panel C: 95% CI width decay."""
    dur = agg["duration_sec"]

    ax.plot(dur, agg["ciw_mean"], color=C_BSNET, lw=2.5, marker="D",
            markersize=7, label="Mean CI Width", zorder=3)
    ax.fill_between(
        dur,
        agg["ciw_mean"] - agg["ciw_sd"],
        agg["ciw_mean"] + agg["ciw_sd"],
        color=C_BSNET, alpha=0.20, zorder=2,
        label="± 1 SD",
    )

    ax.axvline(x=120, color="#888888", lw=1.5, ls="--", alpha=0.7)

    ax.set_xlabel("Scan Duration (s)", **FONT_AXIS)
    ax.set_ylabel("95% CI Width", **FONT_AXIS)
    ax.set_title("C.  Statistical Uncertainty Decay",
                 loc="left", **FONT_PANEL)
    ax.set_xlim(0, 480)
    ax.set_ylim(0, None)
    ax.set_xticks([30, 60, 90, 120, 180, 240, 300, 450])
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=10, framealpha=0.9)


def _plot_panel_d(
    ax: plt.Axes,
    sub_dur: pd.DataFrame,
    agg: pd.DataFrame,
) -> None:
    """Panel D: Per-subject ρ̂T trajectories (consistency across N=12)."""
    subjects = sub_dur["sub_id"].unique()
    n_sub = len(subjects)

    # Individual trajectories (thin, translucent blue)
    for sub in subjects:
        sub_df = sub_dur[sub_dur["sub_id"] == sub].sort_values("duration_sec")
        ax.plot(
            sub_df["duration_sec"], sub_df["rho_hat_T"],
            color=C_BSNET, lw=1.0, alpha=0.35, zorder=2,
        )

    # Group mean (bold blue)
    ax.plot(
        agg["duration_sec"], agg["rho_mean"],
        color=C_BSNET, lw=3.0, marker="s", markersize=8,
        label=f"Mean (N={n_sub})", zorder=4,
    )

    # Individual trajectories legend entry
    ax.plot([], [], color=C_BSNET, lw=1.0, alpha=0.45,
            label="Individual subjects")

    ax.axvline(x=120, color="#888888", lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("Scan Duration (s)", **FONT_AXIS)
    ax.set_ylabel(r"$\hat{\rho}_T$ (BS-NET)", **FONT_AXIS)
    ax.set_title(f"D.  Subject-Level Consistency (N={n_sub})",
                 loc="left", **FONT_PANEL)
    ax.set_xlim(0, 480)
    ax.set_ylim(0.40, 1.02)
    ax.set_xticks([30, 60, 90, 120, 180, 240, 300, 450])
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=10, framealpha=0.9)


# ── Main ──────────────────────────────────────────────────────────────────────


def plot_figure1() -> None:
    """Generate and save Figure 1 (ds007535 real-data duration sweep).

    Raises:
        FileNotFoundError: If sweep CSV is not found.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Sweep CSV not found: {CSV_PATH}\n"
            "Run: python src/scripts/run_duration_sweep.py "
            "--dataset ds007535 --atlas schaefer200 --n-seeds 10"
        )

    apply_bsnet_theme()
    sub_dur, agg = _load_and_aggregate(CSV_PATH)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10),
        constrained_layout=True,
    )
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    _plot_panel_a(ax1, agg)
    _plot_panel_b(ax2, agg)
    _plot_panel_c(ax3, agg)
    _plot_panel_d(ax4, sub_dur, agg)

    # Overall title
    n_sub = sub_dur["sub_id"].nunique()
    fig.suptitle(
        f"Figure 1 — Duration Sweep Validation (ds007535, N={n_sub}, "
        "Schaefer 200, Fisher z)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    save_figure(fig, OUTPUT_NAME)
    print(f"Saved: artifacts/figures/{OUTPUT_NAME}")


if __name__ == "__main__":
    plot_figure1()
