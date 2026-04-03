"""Figure 1: Duration Sweep + FC Stratification — ds000243 XCP-D.

4-panel figure (1×4):

  Panel A : ρ̂T (BS-NET), r_FC (Raw, all pairs), r_FC (Strong |FC|>0.20)
            vs duration — mean ± subject SD
  Panel B : Improvement Δ = ρ̂T − r_FC vs duration — bar chart
  Panel C : 95% CI width decay vs duration
  Panel D : Quartile stratification at 2-min (Q1–Q4 connection strength)
            with ρ̂T and overall r_FC reference lines

Color schema (unified with Fig 3–7):
  Amber  #fdae61  — Raw FC (overall)
  Blue   #4A90E2  — BS-NET ρ̂T
  Gray   #95a5a6  — Reference / weak connections
  Amber dotted    — Strong FC (|FC|>0.20), no correction

Usage:
    cd /path/to/bsNet
    python src/visualization/plot_figure1_ds000243_xcpd.py [--no-strong]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR  = Path("data/ds000243/results")
CACHE_DIR    = Path("data/ds000243/timeseries_cache_xcpd")
OUTPUT_NAME  = "Figure1_ds000243_xcpd_DurationSweep.png"
DATASET      = "ds000243_xcpd"
ATLAS        = "4s256parcels"

TR           = 2.5
MIN_REF_SEC  = 600.0
STRONG_THRESH = 0.20      # |ref FC| threshold for "strong connections"
STRATIFY_SEEDS = [42, 43, 44]   # seeds for strong-FC computation (fast)

C_RAW    = CONDITION_PALETTE["raw"]        # Amber  #fdae61
C_BSNET  = CONDITION_PALETTE["bsnet"]      # Blue   #4A90E2
C_REF    = CONDITION_PALETTE["reference"]  # Gray   #95a5a6
C_STRONG = "#c87137"                       # Dark amber — strong FC line

XTICKS       = [30, 60, 90, 120, 180, 240, 300, 450]
OVERCORRECT  = 450
TWO_MIN_SEC  = 120
Q_LABELS     = ["Q1\n(Weakest)", "Q2\n(Weak)", "Q3\n(Strong)", "Q4\n(Strongest)"]
Q_COLORS     = [C_REF, C_REF, C_BSNET, C_BSNET]


# ── Data loading ──────────────────────────────────────────────────────────────
def _load_aggregated() -> pd.DataFrame:
    path = RESULTS_DIR / f"{DATASET}_duration_sweep_{ATLAS}_aggregated.csv"
    return pd.read_csv(path).sort_values("duration_sec")


def _load_per_record() -> pd.DataFrame:
    path = RESULTS_DIR / f"{DATASET}_duration_sweep_{ATLAS}.csv"
    return pd.read_csv(path)


def _subject_sd(df_rec: pd.DataFrame) -> pd.DataFrame:
    sub_means = (
        df_rec.groupby(["sub_id", "duration_sec"])[["rho_hat_T", "r_fc_raw"]]
        .mean().reset_index()
    )
    return (
        sub_means.groupby("duration_sec")[["rho_hat_T", "r_fc_raw"]]
        .std().reset_index()
        .rename(columns={"rho_hat_T": "rho_sub_sd", "r_fc_raw": "r_fc_sub_sd"})
    )


def _load_timeseries(n_max: int = 0) -> list[dict]:
    """Load .npy timeseries for subjects with total_sec >= MIN_REF_SEC."""
    cache = CACHE_DIR / ATLAS
    files = sorted(cache.glob("*.npy"))
    if n_max > 0:
        files = files[:n_max]
    subjects = []
    for fp in files:
        ts = np.load(fp).astype(np.float64)
        total_sec = ts.shape[0] * TR
        if total_sec < MIN_REF_SEC:
            continue
        subjects.append({"sub_id": fp.stem.split("_")[0], "ts": ts})
    return subjects


def _pearson_fc(ts: np.ndarray) -> np.ndarray:
    fc = np.corrcoef(ts.T)
    fc = np.nan_to_num(fc, nan=0.0)
    i, j = np.triu_indices_from(fc, k=1)
    return fc[i, j]


# ── Strong-FC sweep (Panel A 3rd line) ───────────────────────────────────────
def _compute_strong_fc_sweep(
    subjects: list[dict],
    durations: list[int],
    threshold: float,
    seeds: list[int],
) -> pd.DataFrame:
    """Compute r_FC restricted to |ref FC|>threshold across durations.

    Returns DataFrame with columns: duration_sec, r_strong_mean, r_strong_sd.
    """
    dur_data: dict[int, list[float]] = {d: [] for d in durations}

    for sub in subjects:
        ts = sub["ts"]
        fc_ref = _pearson_fc(ts)
        mask = np.abs(fc_ref) > threshold
        if mask.sum() < 50:
            continue

        for seed in seeds:
            rng = np.random.default_rng(seed)
            for dur in durations:
                short_vols = int(dur / TR)
                max_start = ts.shape[0] - short_vols
                if max_start < 0:
                    continue
                start = int(rng.integers(0, max_start + 1))
                fc_short = _pearson_fc(ts[start: start + short_vols])
                r, _ = spearmanr(fc_short[mask], fc_ref[mask])
                if not np.isnan(r):
                    dur_data[dur].append(r)

    rows = []
    for dur in durations:
        vals = dur_data[dur]
        if vals:
            rows.append({
                "duration_sec": dur,
                "r_strong_mean": float(np.mean(vals)),
                "r_strong_sd":   float(np.std(vals)),
            })
    return pd.DataFrame(rows).sort_values("duration_sec")


# ── Quartile data (Panel D) ───────────────────────────────────────────────────
def _compute_quartile_at_dur(
    subjects: list[dict],
    short_sec: int,
    seeds: list[int],
    n_quartiles: int = 4,
) -> tuple[list[float], list[float]]:
    """Compute per-quartile r_FC at a fixed duration.

    Returns (means, sds) lists of length n_quartiles.
    """
    short_vols = int(short_sec / TR)
    q_corrs: dict[int, list[float]] = {q: [] for q in range(n_quartiles)}

    for sub in subjects:
        ts = sub["ts"]
        fc_ref = _pearson_fc(ts)
        abs_ref = np.abs(fc_ref)
        q_bounds = np.quantile(abs_ref, np.linspace(0, 1, n_quartiles + 1))

        for seed in seeds:
            rng = np.random.default_rng(seed)
            max_start = ts.shape[0] - short_vols
            if max_start < 0:
                continue
            start = int(rng.integers(0, max_start + 1))
            fc_short = _pearson_fc(ts[start: start + short_vols])

            for q in range(n_quartiles):
                lo, hi = q_bounds[q], q_bounds[q + 1]
                mask = (abs_ref >= lo) & (abs_ref <= hi if q == n_quartiles - 1
                                          else abs_ref < hi)
                if mask.sum() < 10:
                    continue
                r, _ = spearmanr(fc_short[mask], fc_ref[mask])
                if not np.isnan(r):
                    q_corrs[q].append(r)

    means = [float(np.mean(q_corrs[q])) if q_corrs[q] else np.nan
             for q in range(n_quartiles)]
    sds   = [float(np.std(q_corrs[q])) if q_corrs[q] else np.nan
             for q in range(n_quartiles)]
    return means, sds


# ── Shared axis helpers ───────────────────────────────────────────────────────
def _set_xticks(ax: plt.Axes) -> None:
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(x) for x in XTICKS], fontsize=FONT_TICK)
    ax.set_xlim(20, 470)


def _mark_2min(ax: plt.Axes, y_text: float) -> None:
    ax.axvline(TWO_MIN_SEC, color="#888888", lw=1.2, ls="--", alpha=0.65, zorder=1)
    ax.text(TWO_MIN_SEC + 5, y_text, "2 min",
            ha="left", va="top", fontsize=8, color="#666666")


# ── Panel A ───────────────────────────────────────────────────────────────────
def panel_a(
    ax: plt.Axes,
    agg: pd.DataFrame,
    sub_sd: pd.DataFrame,
    strong_df: pd.DataFrame | None,
) -> None:
    merged = agg.merge(sub_sd, on="duration_sec", how="left")
    dur      = merged["duration_sec"].values
    rho_mean = merged["rho_hat_T_mean"].values
    rho_sd   = merged["rho_sub_sd"].values
    rfc_mean = merged["r_fc_mean"].values
    rfc_sd   = merged["r_fc_sub_sd"].values
    oc       = dur == OVERCORRECT

    # ── BS-NET ρ̂T (blue solid)
    ax.fill_between(dur, rho_mean - rho_sd, rho_mean + rho_sd,
                    color=C_BSNET, alpha=0.13, zorder=2)
    ax.plot(dur[~oc], rho_mean[~oc], color=C_BSNET, lw=2.5,
            marker="o", ms=7, label="BS-NET (ρ̂T)", zorder=4)
    ax.plot(dur[oc], rho_mean[oc], color=C_BSNET, lw=0, marker="o", ms=7,
            mfc="white", mec=C_BSNET, mew=2.0, zorder=4)

    # ── Raw FC all pairs (amber dashed)
    ax.fill_between(dur, rfc_mean - rfc_sd, rfc_mean + rfc_sd,
                    color=C_RAW, alpha=0.13, zorder=2)
    ax.plot(dur[~oc], rfc_mean[~oc], color=C_RAW, lw=2.0,
            marker="s", ms=6, ls="--", label="Raw FC (all pairs)", zorder=3)
    ax.plot(dur[oc], rfc_mean[oc], color=C_RAW, lw=0, marker="s", ms=6,
            mfc="white", mec=C_RAW, mew=2.0, ls="none", zorder=3)

    # ── Strong FC only (dark amber dotted)
    if strong_df is not None and not strong_df.empty:
        sd = strong_df["duration_sec"].values
        sm = strong_df["r_strong_mean"].values
        ss = strong_df["r_strong_sd"].values
        soc = sd == OVERCORRECT
        ax.fill_between(sd, sm - ss, sm + ss,
                        color=C_STRONG, alpha=0.10, zorder=2)
        ax.plot(sd[~soc], sm[~soc], color=C_STRONG, lw=2.0, marker="^", ms=6,
                ls=":", label=f"Raw FC (|FC|>{STRONG_THRESH}, no correction)",
                zorder=3)
        if soc.any():
            ax.plot(sd[soc], sm[soc], color=C_STRONG, lw=0, marker="^", ms=6,
                    mfc="white", mec=C_STRONG, mew=2.0, zorder=3)

    ax.set_ylim(0.35, 1.02)
    _mark_2min(ax, 1.00)
    _set_xticks(ax)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_xlabel("Scan duration (s)", **FONT_AXIS)
    ax.set_ylabel("FC reliability", **FONT_AXIS)
    ax.set_title("A. Reliability vs. Duration\n(4S256Parcels, N=49)", **FONT_PANEL)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.88)

    # 2-min annotation
    idx = np.where(dur == TWO_MIN_SEC)[0]
    if len(idx):
        i = idx[0]
        ax.annotate(f"ρ̂T={rho_mean[i]:.3f}",
                    xy=(TWO_MIN_SEC, rho_mean[i]),
                    xytext=(145, rho_mean[i] - 0.075),
                    fontsize=8, color=C_BSNET,
                    arrowprops=dict(arrowstyle="-", color=C_BSNET, lw=0.9))
        ax.annotate(f"r={rfc_mean[i]:.3f}",
                    xy=(TWO_MIN_SEC, rfc_mean[i]),
                    xytext=(145, rfc_mean[i] - 0.075),
                    fontsize=8, color=C_RAW,
                    arrowprops=dict(arrowstyle="-", color=C_RAW, lw=0.9))


# ── Panel B ───────────────────────────────────────────────────────────────────
def panel_b(ax: plt.Axes, agg: pd.DataFrame) -> None:
    dur = agg["duration_sec"].values
    imp = agg["improvement_mean"].values
    isd = agg["improvement_sd"].values

    colors = [C_BSNET if d < OVERCORRECT else "#bbccdd" for d in dur]
    bars = ax.bar(dur, imp, width=18, color=colors, alpha=0.85,
                  edgecolor="white", lw=0.8, zorder=3)
    ax.errorbar(dur, imp, yerr=isd, fmt="none",
                ecolor="#555555", elinewidth=1.2, capsize=3, zorder=4)
    ax.axhline(0, color="#444444", lw=1.0, zorder=2)

    for bar, val, d in zip(bars, imp, dur):
        va    = "top" if val < 0 else "bottom"
        off   = -0.005 if val < 0 else 0.005
        color = "#999999" if d == OVERCORRECT else "#333333"
        label = f"{val:+.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, val + off, label,
                ha="center", va=va, fontsize=7.5, color=color)

    ax.axvline(TWO_MIN_SEC, color="#888888", lw=1.2, ls="--", alpha=0.65, zorder=1)
    ax.text(455, min(imp) - 0.004, "OC", ha="center", va="top",
            fontsize=7.5, color="#aaaaaa", style="italic")

    _set_xticks(ax)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Scan duration (s)", **FONT_AXIS)
    ax.set_ylabel("Δ (ρ̂T − r_FC)", **FONT_AXIS)
    ax.set_title("B. Improvement Δ vs. Duration", **FONT_PANEL)


# ── Panel C ───────────────────────────────────────────────────────────────────
def panel_c(ax: plt.Axes, agg: pd.DataFrame) -> None:
    dur = agg["duration_sec"].values
    ciw = agg["ci_width_mean"].values
    cis = agg["ci_width_sd"].values
    oc  = dur == OVERCORRECT

    ax.fill_between(dur, ciw - cis, ciw + cis,
                    color=C_BSNET, alpha=0.13, zorder=2)
    ax.plot(dur[~oc], ciw[~oc], color=C_BSNET, lw=2.5,
            marker="o", ms=7, zorder=4)
    ax.plot(dur[oc], ciw[oc], color=C_BSNET, lw=0, marker="o", ms=7,
            mfc="white", mec=C_BSNET, mew=2.0, zorder=4)
    ax.axvline(TWO_MIN_SEC, color="#888888", lw=1.2, ls="--", alpha=0.65, zorder=1)

    idx = np.where(dur == TWO_MIN_SEC)[0]
    if len(idx):
        i = idx[0]
        ax.annotate(f"CI={ciw[i]:.3f}",
                    xy=(TWO_MIN_SEC, ciw[i]),
                    xytext=(160, ciw[i] + 0.022),
                    fontsize=8.5, color=C_BSNET,
                    arrowprops=dict(arrowstyle="-", color=C_BSNET, lw=0.9))

    _set_xticks(ax)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_xlabel("Scan duration (s)", **FONT_AXIS)
    ax.set_ylabel("95% CI width", **FONT_AXIS)
    ax.set_title("C. Estimation Uncertainty vs. Duration", **FONT_PANEL)


# ── Panel D ───────────────────────────────────────────────────────────────────
def panel_d(
    ax: plt.Axes,
    q_means: list[float],
    q_sds: list[float],
    rho_ref: float,
    rfc_ref: float,
) -> None:
    """Quartile stratification bar chart at 2-min scan.

    Args:
        ax: Axes.
        q_means: Mean r per quartile (Q1–Q4).
        q_sds: SD r per quartile.
        rho_ref: BS-NET ρ̂T at 120s (reference line).
        rfc_ref: Raw r_FC (all pairs) at 120s (reference line).
    """
    x = np.arange(len(Q_LABELS))

    bars = ax.bar(x, q_means, color=Q_COLORS, alpha=0.82,
                  width=0.55, edgecolor="white", lw=0.8, zorder=3)
    ax.errorbar(x, q_means, yerr=q_sds, fmt="none",
                ecolor="#444444", elinewidth=1.3, capsize=4, zorder=4)

    # value labels
    for bar, m, s in zip(bars, q_means, q_sds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.012,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")

    # Reference lines
    ax.axhline(rho_ref, color=C_BSNET, lw=1.8, ls="--", zorder=2,
               label=f"BS-NET ρ̂T = {rho_ref:.3f}")
    ax.axhline(rfc_ref, color=C_RAW, lw=1.5, ls=":", zorder=2,
               label=f"Raw FC (all) = {rfc_ref:.3f}")

    # Shade region: Q4 vs ρ̂T gap annotation
    if not np.isnan(q_means[3]):
        ax.annotate("",
                    xy=(3, q_means[3]),
                    xytext=(3, rho_ref),
                    arrowprops=dict(arrowstyle="<->", color="#666666", lw=1.2))
        ax.text(3.32, (q_means[3] + rho_ref) / 2,
                f"Δ={q_means[3]-rho_ref:+.3f}",
                ha="left", va="center", fontsize=7.5, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(Q_LABELS, fontsize=FONT_TICK)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_xlabel("|Reference FC| strength quartile", **FONT_AXIS)
    ax.set_ylabel("Spearman r (2-min vs. reference FC)", **FONT_AXIS)
    ax.set_title(
        "D. Reliability by Connection Strength\n(at 2-min scan, N=49)",
        **FONT_PANEL,
    )
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.88)

    # Zero-inflation annotation
    ax.text(0, q_means[0] - q_sds[0] - 0.04,
            f"35.9% near-zero\n(|FC|<0.10)",
            ha="center", va="top", fontsize=7.5, color="#888888",
            style="italic")


# ── Main ──────────────────────────────────────────────────────────────────────
def plot_figure1(compute_strong: bool = True) -> None:
    apply_bsnet_theme()

    print("Loading CSVs...")
    agg    = _load_aggregated()
    rec    = _load_per_record()
    sub_sd = _subject_sd(rec)

    # Reference values at 2-min for Panel D
    row_2min = agg[agg["duration_sec"] == TWO_MIN_SEC]
    rho_2min = float(row_2min["rho_hat_T_mean"].iloc[0])
    rfc_2min = float(row_2min["r_fc_mean"].iloc[0])

    strong_df = None
    q_means, q_sds = [], []

    if compute_strong:
        print("Loading timeseries for stratification analysis...")
        subjects = _load_timeseries()
        n_sub = len(subjects)
        print(f"  {n_sub} subjects loaded (≥{MIN_REF_SEC:.0f}s)")

        print("Computing strong-FC sweep (Panel A 3rd line)...")
        strong_df = _compute_strong_fc_sweep(
            subjects, XTICKS, STRONG_THRESH, STRATIFY_SEEDS
        )

        print("Computing quartile stratification at 2-min (Panel D)...")
        q_means, q_sds = _compute_quartile_at_dur(
            subjects, TWO_MIN_SEC, STRATIFY_SEEDS
        )
    else:
        n_sub = int(agg["n_subjects"].iloc[0])

    n_sub_agg = int(agg["n_subjects"].iloc[0])
    n_seeds   = int(agg["n_seeds"].iloc[0])

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 5.4))
    gs  = gridspec.GridSpec(
        1, 4, figure=fig,
        wspace=0.38,
        left=0.05, right=0.97,
        top=0.88, bottom=0.14,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    fig.suptitle(
        f"Figure 1 — Duration Sweep & FC Stratification: "
        f"BS-NET on ds000243 XCP-D "
        f"(4S256Parcels, N={n_sub_agg}, {n_seeds} seeds)",
        fontsize=12.5, fontweight="bold",
    )

    panel_a(axes[0], agg, sub_sd, strong_df)
    panel_b(axes[1], agg)
    panel_c(axes[2], agg)

    if q_means:
        panel_d(axes[3], q_means, q_sds, rho_2min, rfc_2min)
    else:
        axes[3].text(0.5, 0.5, "Run with --strong\nto compute Panel D",
                     ha="center", va="center", transform=axes[3].transAxes,
                     fontsize=11, color="#aaaaaa")
        axes[3].set_title("D. Reliability by Connection Strength", **FONT_PANEL)

    out = save_figure(fig, OUTPUT_NAME)
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-strong", action="store_true",
                        help="Skip strong-FC computation (faster, Panel A 3rd line & Panel D omitted)")
    args = parser.parse_args()

    plot_figure1(compute_strong=not args.no_strong)
