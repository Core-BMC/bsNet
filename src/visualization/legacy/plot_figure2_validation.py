"""Generate Figure 2: BS-NET validation on ds007535 (SpeechHemi) real data.

Dataset: N=30, task-residual FC, 6 atlases, 2-min short scan (120 s),
         full BS-NET pipeline (block-bootstrap + SB prophecy + Fisher-z correction).

Data source: data/ds007535/results/ds007535_duration_sweep_{atlas}.csv
  Columns: sub_id, duration_sec, seed, r_fc_raw, rho_hat_T, ci_lower, ci_upper
  No recomputation — CSVs already contain full-pipeline results.

Layout: 2×2
  A — Scatter: r_fc_raw vs ρ̂T (seed-averaged, 30 pts/atlas × 6 atlas)
  B — KDE: r_fc_raw vs ρ̂T distribution per atlas (paired overlay)
  C — Violin: Δ = ρ̂T − r_fc_raw by atlas (improvement distribution)
  D — Bar: mean Δ ± SD per atlas + overall mean line
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import (
    ATLAS_META,
    CONDITION_PALETTE,
    FIGSIZE,
    FONT_AXIS,
    FONT_PANEL,
    FONT_TICK,
    apply_bsnet_theme,
    save_figure,
)

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("data/ds007535/results")
OUTPUT_NAME = "Figure2_Validation_ds007535.png"
SHORT_SEC   = 120          # 2-min short scan
ATLASES     = list(ATLAS_META.keys())

C_RAW   = CONDITION_PALETTE["raw"]    # Amber
C_BSNET = CONDITION_PALETTE["bsnet"]  # Blue


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_subject_level(short_sec: int = SHORT_SEC) -> pd.DataFrame:
    """Load all atlases, filter to short_sec, seed-average per (sub_id, atlas).

    Returns:
        DataFrame with columns: sub_id, atlas, r_fc, rho_hat_T, ci_lower,
        ci_upper, improvement, label (atlas display name).
    """
    rows: list[pd.DataFrame] = []
    for atlas in ATLASES:
        path = RESULTS_DIR / f"ds007535_duration_sweep_{atlas}.csv"
        if not path.exists():
            print(f"  [WARN] missing: {path.name}")
            continue
        df = pd.read_csv(path)
        df = df[df["duration_sec"] == short_sec]
        agg = (
            df.groupby("sub_id", as_index=False)
            .agg(
                r_fc=("r_fc_raw", "mean"),
                rho_hat_T=("rho_hat_T", "mean"),
                ci_lower=("ci_lower", "mean"),
                ci_upper=("ci_upper", "mean"),
                improvement=("improvement", "mean"),
            )
        )
        agg["atlas"] = atlas
        agg["label"] = ATLAS_META[atlas]["label"]
        rows.append(agg)

    if not rows:
        raise FileNotFoundError(f"No CSVs found in {RESULTS_DIR}")
    return pd.concat(rows, ignore_index=True)


# ── Panel helpers ──────────────────────────────────────────────────────────────


def _plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel A: r_fc vs ρ̂T scatter — 30 subjects × 6 atlases.

    Each atlas: seed-averaged subject points (colored dots).
    Identity line shows zero-improvement boundary.
    """
    lo = min(df["r_fc"].min(), df["rho_hat_T"].min()) - 0.03
    hi = max(df["r_fc"].max(), df["rho_hat_T"].max()) + 0.03

    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1.2, ls="--",
            zorder=1, label="y = x")

    for atlas in ATLASES:
        sub = df[df["atlas"] == atlas]
        if sub.empty:
            continue
        m = ATLAS_META[atlas]
        ax.scatter(
            sub["r_fc"], sub["rho_hat_T"],
            color=m["color"], s=38, alpha=0.75,
            linewidths=0.5, edgecolors=m["color"],
            zorder=3, label=m["label"],
        )

    ax.axvline(df["r_fc"].mean(),     color=C_RAW,   lw=0.9, ls=":", alpha=0.6)
    ax.axhline(df["rho_hat_T"].mean(), color=C_BSNET, lw=0.9, ls=":", alpha=0.6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$r_{\mathrm{FC}}$ (Raw, 2 min)", **FONT_AXIS)
    ax.set_ylabel(r"$\hat{\rho}_T$ (bs-Net, 2 min)", **FONT_AXIS)
    ax.set_title("A", loc="left", **FONT_PANEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)


def _plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel B: Paired KDE — r_fc vs ρ̂T per atlas.

    Solid = r_fc (Raw), dashed = ρ̂T (bs-Net). Schaefer200 opaque; others faded.
    """
    PRIMARY = "schaefer200"
    x_grid  = np.linspace(
        min(df["r_fc"].min(), df["rho_hat_T"].min()) - 0.05,
        max(df["r_fc"].max(), df["rho_hat_T"].max()) + 0.05,
        300,
    )

    for atlas in ATLASES:
        sub = df[df["atlas"] == atlas]
        if len(sub) < 4:
            continue
        m       = ATLAS_META[atlas]
        color   = m["color"]
        is_main = atlas == PRIMARY
        lw      = 2.2 if is_main else 0.9
        alpha_l = 0.92 if is_main else 0.28

        kde_raw   = gaussian_kde(sub["r_fc"])
        kde_bsnet = gaussian_kde(sub["rho_hat_T"])

        ax.plot(x_grid, kde_raw(x_grid),   color=color, lw=lw, ls="-",
                alpha=alpha_l, zorder=3 if is_main else 2)
        ax.plot(x_grid, kde_bsnet(x_grid), color=color, lw=lw, ls="--",
                alpha=alpha_l, zorder=3 if is_main else 2)

        if is_main:
            ax.fill_between(x_grid, kde_raw(x_grid),   alpha=0.08, color=C_RAW)
            ax.fill_between(x_grid, kde_bsnet(x_grid), alpha=0.08, color=C_BSNET)

    leg_handles = [
        Line2D([0], [0], color="#555555", lw=2, ls="-",
               label=r"$r_{\mathrm{FC}}$ (Raw)"),
        Line2D([0], [0], color="#555555", lw=2, ls="--",
               label=r"$\hat{\rho}_T$ (bs-Net)"),
    ]
    ax.legend(handles=leg_handles, fontsize=8.5, framealpha=0.9)
    ax.set_xlabel("Pearson r", **FONT_AXIS)
    ax.set_ylabel("Density", **FONT_AXIS)
    ax.set_title("B", loc="left", **FONT_PANEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_ylim(bottom=0)


def _plot_panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel C: Improvement violin Δ = ρ̂T − r_fc by atlas.

    6 violins colored by atlas. Schaefer200 opaque; others faded.
    Subject dots + median bar overlaid. Horizontal line at Δ=0.
    """
    PRIMARY   = "schaefer200"
    positions = np.arange(len(ATLASES))

    ax.axhline(0, color="#888888", lw=1.0, ls="--", alpha=0.7, zorder=1)

    rng = np.random.default_rng(42)
    for ai, atlas in enumerate(ATLASES):
        sub = df[df["atlas"] == atlas]["improvement"].dropna().values
        if len(sub) < 3:
            continue
        m       = ATLAS_META[atlas]
        color   = m["color"]
        is_main = atlas == PRIMARY
        alpha_v = 0.70 if is_main else 0.22

        parts = ax.violinplot(
            sub, positions=[ai], widths=0.6,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(alpha_v)
            pc.set_edgecolor("none")

        jitter = rng.uniform(-0.12, 0.12, len(sub))
        ax.scatter(
            ai + jitter, sub,
            color=color,
            s=22 if is_main else 8,
            alpha=0.85 if is_main else 0.22,
            linewidths=0, zorder=4,
        )

        med = float(np.median(sub))
        ax.plot(
            [ai - 0.22, ai + 0.22], [med, med],
            color=color,
            lw=2.2 if is_main else 0.9,
            alpha=0.95 if is_main else 0.35,
            zorder=5,
        )

    labels = [ATLAS_META[a]["label"] for a in ATLASES]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=FONT_TICK - 0.5)
    ax.set_ylabel(r"$\hat{\rho}_T - r_{\mathrm{FC}}$ (Δ)", **FONT_AXIS)
    ax.set_title("C", loc="left", **FONT_PANEL)
    ax.tick_params(labelsize=FONT_TICK)


def _plot_panel_d(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel D: Mean Δ ± SD per atlas bar chart + overall mean line.

    Schaefer200 fully opaque; others faded.
    """
    PRIMARY   = "schaefer200"
    positions = np.arange(len(ATLASES))

    for ai, atlas in enumerate(ATLASES):
        sub     = df[df["atlas"] == atlas]["improvement"].dropna()
        m_val   = float(sub.mean())
        s_val   = float(sub.std())
        color   = ATLAS_META[atlas]["color"]
        is_main = atlas == PRIMARY
        alpha   = 0.85 if is_main else 0.28

        ax.bar(ai, m_val, color=color, alpha=alpha,
               edgecolor="none", width=0.6, zorder=2)
        ax.errorbar(ai, m_val, yerr=s_val,
                    color=color, alpha=min(alpha + 0.3, 1.0),
                    capsize=4, lw=1.5, zorder=3)
        ax.text(
            ai, m_val + s_val + 0.003,
            f"{m_val:.3f}",
            ha="center", va="bottom",
            fontsize=7.5, color=color,
            alpha=min(alpha + 0.3, 1.0),
        )

    overall_mean = float(df["improvement"].mean())
    ax.axhline(overall_mean, color="#333333", lw=1.4, ls=":",
               zorder=4, label=f"Overall mean = {overall_mean:.3f}")

    labels = [ATLAS_META[a]["label"] for a in ATLASES]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=FONT_TICK - 0.5)
    ax.set_ylabel(r"Mean $\Delta$ ± SD", **FONT_AXIS)
    ax.set_title("D", loc="left", **FONT_PANEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_ylim(0, 0.2)
    ax.legend(fontsize=8, framealpha=0.9)


# ── Main ──────────────────────────────────────────────────────────────────────


def plot_figure2(short_sec: int = SHORT_SEC) -> None:
    """Generate and save Figure 2: real-data validation (ds007535, 6 atlases).

    Loads per-record CSVs (full BS-NET pipeline, fisher_z correction).
    Seed-averages per subject, then plots 4-panel validation figure.

    Args:
        short_sec: Short-scan duration to analyse (default 120 = 2 min).
    """
    print(f"Loading ds007535 results (duration={short_sec}s, 6 atlases) …")
    df = _load_subject_level(short_sec)

    N_sub = df["sub_id"].nunique()
    N_atl = df["atlas"].nunique()
    print(f"  {N_sub} subjects × {N_atl} atlases = {len(df)} data points")

    print()
    for atlas in ATLASES:
        sub = df[df["atlas"] == atlas]
        if sub.empty:
            continue
        print(
            f"  {ATLAS_META[atlas]['label']:20s}  "
            f"r_fc={sub['r_fc'].mean():.3f}±{sub['r_fc'].std():.3f}  "
            f"ρ̂T={sub['rho_hat_T'].mean():.3f}±{sub['rho_hat_T'].std():.3f}  "
            f"Δ={sub['improvement'].mean():.3f}±{sub['improvement'].std():.3f}"
        )

    apply_bsnet_theme()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE.get("2x2", (14, 11)))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    _plot_panel_a(ax_a, df)
    _plot_panel_b(ax_b, df)
    _plot_panel_c(ax_c, df)
    _plot_panel_d(ax_d, df)

    fig.suptitle(
        f"BS-NET Validation — ds007535 (SpeechHemi)  "
        f"N={N_sub}, {N_atl} atlases, {short_sec}s short scan",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_NAME)
    print(f"\nSaved: {OUTPUT_NAME}")


if __name__ == "__main__":
    plot_figure2()
