"""Generate Figure 1: Duration sweep results from ds007535 (SpeechHemi).

7-panel figure using real rsfMRI data (N=30, task-residual FC, multi-atlas):

  Panels A–C: Primary atlas (Schaefer 200) — detailed accuracy, improvement, CI
  Panels D–F: Across-atlas — comparison, robustness, per-subject consistency
  Panel G  : Exemplar FC scatter — short-scan vs. reference (real data, sub-01)

  A - ρ̂T (bs-Net) and r_FC (Raw) vs. duration — mean ± subject SD (Schaefer 200)
  B - Improvement Δ = ρ̂T − r_FC vs. duration — bar chart (Schaefer 200)
  C - 95% CI width decay vs. duration (Schaefer 200)
  D - Multi-atlas ρ̂T comparison — mean lines, all 6 atlases
  E - Atlas robustness at 2 min — ρ̂T bar chart across atlases
  F - Per-subject ρ̂T trajectories — all 6 atlases (thin individual + bold mean)
  G - FC pair scatter (19,900 pairs): FC_short vs. FC_reference, sub-01, 2-min
      Replaces synthetic coherence visualization (Figure1_Combined Panel D)

Subject-level SD (Panels A, C, F):
  Two-stage aggregation from per-record CSV:
    Stage 1: per (sub_id, duration_sec) → mean across seeds
    Stage 2: per duration_sec → mean ± SD across subjects

Layout: 3×3 GridSpec — Row 0-1: A–F, Row 2: G (full-width)

Color scheme:
  Amber  #fdae61 = Raw FC
  Blue   #4A90E2 = bs-Net (primary / Schaefer 200)
  Atlas palette: 6 distinct colors for multi-atlas panels
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.bootstrap import block_bootstrap_indices, estimate_optimal_block_length
from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.data.data_loader import get_fc_matrix
from src.visualization.style import (
    ATLAS_META,
    BAR_STYLE,
    CONDITION_PALETTE,
    FIGSIZE,
    FONT_AXIS,
    FONT_PANEL,
    FONT_TICK,
    apply_bsnet_theme,
    save_figure,
)

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("data/ds007535/results")
CACHE_DIR = Path("data/ds007535/timeseries_cache")
OUTPUT_NAME = "Figure1_ds007535_DurationSweep.png"
PRIMARY_ATLAS = "schaefer200"
TWO_MIN_SEC = 120
EXEMPLAR_SUB = "sub-01"   # representative subject (r_fc≈median, ρ̂T=0.838)

C_RAW   = CONDITION_PALETTE["raw"]    # Amber
C_BSNET = CONDITION_PALETTE["bsnet"]  # Blue
# ATLAS_META, FONT_PANEL, FONT_AXIS, FONT_TICK, BAR_STYLE, FIGSIZE → from style.py
XTICKS = [30, 60, 90, 120, 180, 240, 300, 450]


# ── Data loading & aggregation ────────────────────────────────────────────────


def _load_per_record(atlas: str) -> pd.DataFrame:
    """Load per-record (raw) sweep CSV for one atlas.

    Args:
        atlas: Atlas name key.

    Returns:
        DataFrame with sub_id, duration_sec, seed, rho_hat_T, r_fc_raw,
        ci_lower, ci_upper.
    """
    path = RESULTS_DIR / f"ds007535_duration_sweep_{atlas}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Per-record CSV not found: {path}")
    return pd.read_csv(path)


def _compute_agg(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-stage aggregation giving proper subject-level SD.

    Stage 1: per (sub_id, duration_sec) → mean across seeds
    Stage 2: per duration_sec → mean ± SD across subjects

    Args:
        df_raw: Per-record DataFrame from _load_per_record().

    Returns:
        (sub_dur, agg):
          sub_dur — subject-level means per (sub_id, duration_sec)
          agg     — across-subject statistics per duration_sec
    """
    # Stage 1: seed-average per subject
    sub_dur = (
        df_raw.groupby(["sub_id", "duration_sec"])
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

    # Stage 2: subject-mean ± subject-SD per duration
    agg = (
        sub_dur.groupby("duration_sec")
        .agg(
            r_fc_mean=("r_fc_raw", "mean"),
            r_fc_sd=("r_fc_raw", "std"),
            rho_hat_T_mean=("rho_hat_T", "mean"),
            rho_hat_T_sd=("rho_hat_T", "std"),
            improvement_mean=("improvement", "mean"),
            improvement_sd=("improvement", "std"),
            ci_width_mean=("ci_width", "mean"),
            ci_width_sd=("ci_width", "std"),
        )
        .reset_index()
    )
    return sub_dur, agg


# ── Panel helpers ─────────────────────────────────────────────────────────────


def _plot_panel_a(
    ax: plt.Axes,
    atlas_sub_durs: dict[str, pd.DataFrame],
    atlas_aggs: dict[str, pd.DataFrame],
) -> None:
    """Panel A: ρ̂T + r_FC vs. duration — all 6 atlases, Schaefer200 prominent.

    r_FC (Amber) shown only for primary atlas.
    ρ̂T shown for all atlases using atlas colors; non-primary heavily faded.
    """
    PRIMARY = "schaefer200"

    from matplotlib.lines import Line2D  # local import to avoid top-level clutter

    # ── Raw FC: primary atlas only (Amber, prominent) ──
    agg_p = atlas_aggs[PRIMARY]
    dur_p = agg_p["duration_sec"]
    ax.fill_between(dur_p,
                    agg_p["r_fc_mean"] - agg_p["r_fc_sd"],
                    agg_p["r_fc_mean"] + agg_p["r_fc_sd"],
                    color=C_RAW, alpha=0.20, zorder=2)
    ax.plot(dur_p, agg_p["r_fc_mean"], color=C_RAW, lw=2.5, marker="o",
            markersize=6, zorder=4)

    # ── bs-Net ρ̂T: all 6 atlases (no auto-labels; legend built manually) ──
    for atlas_key, agg in atlas_aggs.items():
        m       = ATLAS_META[atlas_key]
        color   = m["color"]
        is_main = atlas_key == PRIMARY

        line_alpha = 0.92 if is_main else 0.25
        fill_alpha = 0.20 if is_main else 0.05
        lw         = 2.5  if is_main else 1.0
        ms         = 6.0  if is_main else 2.5
        zline      = 5    if is_main else 2

        dur = agg["duration_sec"]
        ax.fill_between(dur,
                        agg["rho_hat_T_mean"] - agg["rho_hat_T_sd"],
                        agg["rho_hat_T_mean"] + agg["rho_hat_T_sd"],
                        color=color, alpha=fill_alpha, zorder=zline - 1)
        ax.plot(dur, agg["rho_hat_T_mean"],
                color=color, lw=lw, ls=m["ls"],
                marker=m["marker"], markersize=ms,
                alpha=line_alpha, zorder=zline)

    ax.axvline(x=TWO_MIN_SEC, color="#888888", lw=1.5, ls="--", alpha=0.7)

    # ── Custom legend: concept labels + atlas entries (Schaefer200 faded, right below ρ̂T) ──
    m_p   = ATLAS_META[PRIMARY]
    OTHER = [k for k in atlas_aggs if k != PRIMARY]
    legend_handles = [
        Line2D([0], [0], color=C_RAW, lw=2.5, marker="o", markersize=6,
               label=r"$r_{\mathrm{FC}}$ (Raw)"),
        Line2D([0], [0], color=m_p["color"], lw=2.5, marker=m_p["marker"],
               markersize=6, alpha=0.92,
               label=r"$\hat{\rho}_T$ (bs-Net)"),
        # Schaefer200 — faded, identical style to other atlas entries
        Line2D([0], [0], color=m_p["color"], lw=1.0, ls=m_p["ls"],
               marker=m_p["marker"], markersize=2.5, alpha=0.25,
               label=m_p["label"]),
        *[
            Line2D([0], [0], color=ATLAS_META[k]["color"], lw=1.0,
                   ls=ATLAS_META[k]["ls"], marker=ATLAS_META[k]["marker"],
                   markersize=2.5, alpha=0.25, label=ATLAS_META[k]["label"])
            for k in OTHER
        ],
    ]
    ax.legend(handles=legend_handles, fontsize=9.0, framealpha=0.9, ncol=2)

    ax.set_xlabel("Scan Duration (s)", **FONT_AXIS)
    ax.set_ylabel("Correlation with Reference FC", **FONT_AXIS)
    ax.set_title("A", loc="left", **FONT_PANEL)
    ax.set_xlim(0, 480)
    ax.set_ylim(0.20, 1.05)
    ax.set_xticks(XTICKS)
    ax.tick_params(labelsize=FONT_TICK)


def _plot_panel_b(
    ax: plt.Axes,
    atlas_sub_durs: dict[str, pd.DataFrame],
) -> None:
    """Panel B: Improvement Δ = ρ̂T − r_FC — grouped box plots, all atlases.

    Each duration has 6 atlas box plots (offset within group).
    Medians connected by lines per atlas. Semi-transparent boxes.

    Args:
        ax: Axes to plot on.
        atlas_sub_durs: Dict of seed-averaged subject DataFrames per atlas.
    """
    durations = sorted(
        next(iter(atlas_sub_durs.values()))["duration_sec"].unique()
    )
    n_dur    = len(durations)
    n_atlas  = len(atlas_sub_durs)
    x_idx    = np.arange(n_dur)                          # 0…7
    offsets  = np.linspace(-0.38, 0.38, n_atlas)
    box_w    = 0.10

    ax.axhline(y=0, color="#888888", lw=1.0, ls="-", alpha=0.5, zorder=1)

    PRIMARY = "schaefer200"

    for ai, (atlas_key, sub_dur) in enumerate(atlas_sub_durs.items()):
        m        = ATLAS_META[atlas_key]
        color    = m["color"]
        is_main  = atlas_key == PRIMARY

        # Schaefer200: full opacity; others: heavily faded
        box_alpha  = 0.55 if is_main else 0.12
        line_alpha = 0.90 if is_main else 0.30
        lw_scale   = m["lw"] if is_main else max(m["lw"] * 0.6, 0.8)
        ms_scale   = 4.5    if is_main else 2.5

        sub_dur = sub_dur.copy()
        sub_dur["improvement"] = sub_dur["rho_hat_T"] - sub_dur["r_fc_raw"]

        medians, positions = [], []
        for di, dur in enumerate(durations):
            vals = sub_dur[sub_dur["duration_sec"] == dur]["improvement"].dropna().values
            if len(vals) == 0:
                continue
            pos = x_idx[di] + offsets[ai]
            positions.append(pos)
            medians.append(float(np.median(vals)))

            ax.boxplot(
                vals,
                positions=[pos],
                widths=box_w,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color, alpha=box_alpha,
                              linewidth=0.7 if is_main else 0.4,
                              edgecolor=color),
                medianprops=dict(color=color,
                                 linewidth=1.8 if is_main else 0.8,
                                 alpha=line_alpha),
                whiskerprops=dict(color=color,
                                  linewidth=0.7 if is_main else 0.3,
                                  alpha=line_alpha * 0.8),
                capprops=dict(color=color,
                              linewidth=0.7 if is_main else 0.3,
                              alpha=line_alpha * 0.8),
                zorder=4 if is_main else 2,
            )

        # Median connection line
        if positions:
            ax.plot(positions, medians,
                    color=color, lw=lw_scale, ls=m["ls"],
                    marker=m["marker"], markersize=ms_scale,
                    alpha=line_alpha, label=m["label"],
                    zorder=5 if is_main else 2)

    ax.set_xticks(x_idx)
    ax.set_xticklabels([str(d) for d in durations], fontsize=FONT_TICK)
    ax.set_xlabel("Scan Duration (s)", **FONT_AXIS)
    ax.set_ylabel(r"$\hat{\rho}_T - r_{\mathrm{FC}}$ (Δ)", **FONT_AXIS)
    ax.set_title("B", loc="left", **FONT_PANEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)


def _plot_panel_c(
    ax: plt.Axes,
    atlas_aggs: dict[str, pd.DataFrame],
) -> None:
    """Panel C: 95% CI width decay — all 6 atlases, Schaefer200 prominent.

    Each atlas: mean CI width line + ±1 SD fill. Non-primary heavily faded.
    """
    PRIMARY = "schaefer200"

    for atlas_key, agg in atlas_aggs.items():
        m       = ATLAS_META[atlas_key]
        color   = m["color"]
        is_main = atlas_key == PRIMARY

        line_alpha = 0.92 if is_main else 0.25
        fill_alpha = 0.20 if is_main else 0.05
        lw         = 2.5  if is_main else 1.0
        ms         = 6.0  if is_main else 2.5
        zline      = 4    if is_main else 2
        label      = m["label"]

        dur = agg["duration_sec"]
        ax.fill_between(dur,
                        agg["ci_width_mean"] - agg["ci_width_sd"],
                        agg["ci_width_mean"] + agg["ci_width_sd"],
                        color=color, alpha=fill_alpha, zorder=zline - 1)
        ax.plot(dur, agg["ci_width_mean"],
                color=color, lw=lw, ls=m["ls"],
                marker=m["marker"], markersize=ms,
                alpha=line_alpha, label=label, zorder=zline)

    ax.axvline(x=TWO_MIN_SEC, color="#888888", lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("Scan Duration (s)", **FONT_AXIS)
    ax.set_ylabel("95% CI Width", **FONT_AXIS)
    ax.set_title("C", loc="left", **FONT_PANEL)
    ax.set_xlim(0, 480)
    ax.set_ylim(0, None)
    ax.set_xticks(XTICKS)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=9.5, framealpha=0.9, ncol=2)


def _plot_panel_d(
    ax: plt.Axes,
    atlas_aggs: dict[str, pd.DataFrame],
) -> None:
    """Panel D: Multi-atlas ρ̂T mean vs. duration (all 6 atlases)."""
    for key, agg in atlas_aggs.items():
        m = ATLAS_META[key]
        ax.plot(agg["duration_sec"], agg["rho_hat_T_mean"],
                color=m["color"], lw=m["lw"], ls=m["ls"],
                marker=m["marker"], markersize=5,
                label=m["label"], zorder=3)

    ax.axvline(x=TWO_MIN_SEC, color="#888888", lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("Scan Duration (s)", **FONT_AXIS)
    ax.set_ylabel(r"$\hat{\rho}_T$ (bs-Net)", **FONT_AXIS)
    # ax.set_title("D.  Atlas Robustness — ρ̂T vs. Duration\n"
    #              "(N=30, mean across subjects)", loc="left", **FONT_PANEL)
    ax.set_title("D", loc="left", **FONT_PANEL)
    ax.set_xlim(0, 480)
    ax.set_ylim(0.60, 0.96)
    ax.set_xticks(XTICKS)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=8.5, framealpha=0.9, ncol=2)


def _plot_panel_e(
    ax: plt.Axes,
    atlas_aggs: dict[str, pd.DataFrame],
    target_sec: int = TWO_MIN_SEC,
) -> None:
    """Panel E: Atlas comparison at 2 min — paired ρ̂T + r_FC bars.

    Args:
        ax: Axes to plot on.
        atlas_aggs: Dict of aggregated DataFrames per atlas.
        target_sec: Duration to compare (default: 120s = 2 min).
    """
    atlas_keys = list(atlas_aggs.keys())
    labels = [ATLAS_META[k]["label"] for k in atlas_keys]
    colors = [ATLAS_META[k]["color"] for k in atlas_keys]

    rho_vals, r_fc_vals = [], []
    for key in atlas_keys:
        row = atlas_aggs[key][atlas_aggs[key]["duration_sec"] == target_sec]
        rho_vals.append(float(row["rho_hat_T_mean"].iloc[0]))
        r_fc_vals.append(float(row["r_fc_mean"].iloc[0]))

    x = np.arange(len(atlas_keys))
    bw = 0.35

    # Raw bars — per-atlas color, solid
    for i, (val, col) in enumerate(zip(r_fc_vals, colors)):
        ax.bar(x[i] - bw / 2, val, width=bw,
               color=col, zorder=3, **BAR_STYLE["raw"])

    # bs-Net bars — per-atlas color, diagonal stripe
    for i, (val, col) in enumerate(zip(rho_vals, colors)):
        ax.bar(x[i] + bw / 2, val, width=bw,
               color=col, zorder=3, **BAR_STYLE["bsnet"])

    # Legend: 패턴으로만 구분 (solid vs stripe), 회색 proxy patch
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#888888", edgecolor="#444444", alpha=0.85,
              label=r"$r_{\mathrm{FC}}$ (Raw)"),
        Patch(facecolor="#888888", edgecolor="#444444", alpha=0.75,
              hatch="////", label=r"$\hat{\rho}_T$ (bs-Net)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.9)

    # Δ annotations
    for i, (rho, rfc) in enumerate(zip(rho_vals, r_fc_vals)):
        delta = rho - rfc
        ax.annotate(f"+{delta:.3f}",
                    xy=(x[i] + bw / 2, rho + 0.003),
                    ha="center", va="bottom", fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Correlation with Reference FC", **FONT_AXIS)
    # ax.set_title(f"E.  Atlas Comparison at {target_sec}s (2 min)\n"
    #              "(N=30)", loc="left", **FONT_PANEL)
    ax.set_title("E", loc="left", **FONT_PANEL)
    ax.set_ylim(0.60, 0.92)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=FONT_TICK)


def _plot_panel_f(
    ax: plt.Axes,
    atlas_sub_durs: dict[str, pd.DataFrame],
    atlas_aggs: dict[str, pd.DataFrame],
) -> None:
    """Panel F: Per-subject ρ̂T trajectories for ALL 6 atlases.

    Each atlas is plotted with its own color:
      - Thin translucent lines for each individual subject
      - Bold line for group mean

    Args:
        ax: Axes to plot on.
        atlas_sub_durs: Dict of subject-level DataFrames per atlas.
        atlas_aggs: Dict of aggregated DataFrames per atlas.
    """
    for key in atlas_sub_durs:
        m = ATLAS_META[key]
        sub_dur = atlas_sub_durs[key]
        agg = atlas_aggs[key]
        subjects = sub_dur["sub_id"].unique()

        # Individual subject trajectories (thin, very translucent)
        for sub in subjects:
            sdf = sub_dur[sub_dur["sub_id"] == sub].sort_values("duration_sec")
            ax.plot(sdf["duration_sec"], sdf["rho_hat_T"],
                    color=m["color"], lw=0.7, alpha=0.18, zorder=2)

        # Group mean (bold, with marker)
        ax.plot(agg["duration_sec"], agg["rho_hat_T_mean"],
                color=m["color"], lw=m["lw"], ls=m["ls"],
                marker=m["marker"], markersize=5,
                label=m["label"], zorder=4)

    ax.axvline(x=TWO_MIN_SEC, color="#888888", lw=1.5, ls="--", alpha=0.7)
    n_sub = next(iter(atlas_sub_durs.values()))["sub_id"].nunique()
    n_atlas = len(atlas_sub_durs)
    ax.set_xlabel("Scan Duration (s)", **FONT_AXIS)
    ax.set_ylabel(r"$\hat{\rho}_T$ (bs-Net)", **FONT_AXIS)
    # ax.set_title(f"F.  Subject-Level Consistency — All Atlases\n"
    #              f"({n_atlas} atlases, N={n_sub}/atlas, thin=individual)",
    #              loc="left", **FONT_PANEL)
    ax.set_title("F", loc="left", **FONT_PANEL)
    ax.set_xlim(0, 480)
    ax.set_ylim(0.55, 1.02)
    ax.set_xticks(XTICKS)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=8.5, framealpha=0.9, ncol=2)


# ── Panel G: cross-subject reliability matrices + exemplar scatter ───────────


def _corr_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-column Pearson r between two [N × M] matrices.

    Args:
        A: Array of shape (N, M).
        B: Array of shape (N, M).

    Returns:
        Array of shape (M,) with Pearson r for each column.
    """
    A_c = A - A.mean(axis=0)
    B_c = B - B.mean(axis=0)
    num  = (A_c * B_c).sum(axis=0)
    den  = np.sqrt((A_c ** 2).sum(axis=0) * (B_c ** 2).sum(axis=0))
    return np.where(den > 1e-12, num / den, 0.0)


def _load_reliability_matrices(
    atlas: str,
    short_sec: int,
    tr: float = 2.0,
    correction_method: str = "fisher_z",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute cross-subject reliability matrices for Panel G1 and G2.

    For each connection (i, j), the cross-subject Pearson r is computed between:
      R_short[i,j]  = r over subjects( fc_short[:, i,j], fc_ref[:, i,j] )
                      → "short-long" reliability (raw short scan)
      R_bsnet[i,j]  = r over subjects( fc_bsnet[:, i,j], fc_ref[:, i,j] )
                      → "bs-Net-long" reliability (full pipeline corrected)

    Full BS-NET pipeline per subject:
      1. Block-bootstrap resampling → bootstrap-averaged FC matrix
      2. run_bootstrap_prediction() → rho_hat_T (scalar, global correction)
      3. Apply z-space additive shift per connection:
           z_shift = arctanh(rho_hat_T) - arctanh(r_fc)
           fc_bsnet[i,j] = tanh( arctanh(fc_bootstrap[i,j]) + z_shift )
         This extends the Fisher-z correction (used in the pipeline) to the
         per-connection level — same z-space shift, applied element-wise.

    Args:
        atlas: Atlas name key.
        short_sec: Short-scan duration in seconds.
        tr: Repetition time in seconds.
        correction_method: Attenuation correction method (default "fisher_z",
            consistent with project-wide recommendation).

    Returns:
        (R_short, R_bsnet, fc_ref_mean, meta)

    Cache:
        Result is saved to data/ds007535/results/rel_cache_{atlas}_{short_sec}s_{correction_method}.npz
        and reloaded on subsequent calls (skip recomputation unless file is missing).
        Delete the .npz to force recomputation.
    """
    npz_path = (
        RESULTS_DIR
        / f"rel_cache_{atlas}_{short_sec}s_{correction_method}.npz"
    )
    if npz_path.exists():
        print(f"Loading cached reliability matrices: {npz_path.name}")
        data = np.load(npz_path, allow_pickle=True)
        meta = data["meta"].item()
        return data["R_short"], data["R_bsnet"], data["fc_ref_mean"], meta

    print(f"Computing reliability matrices (first run — will cache to {npz_path.name}) …")
    cache_path = CACHE_DIR / atlas
    ts_files   = sorted(cache_path.glob(f"*_{atlas}.npy"))
    short_vols = int(short_sec / tr)
    config     = BSNetConfig()

    fc_short_list: list[np.ndarray] = []
    fc_bsnet_list: list[np.ndarray] = []
    fc_ref_list:   list[np.ndarray] = []

    for sub_i, ts_path in enumerate(ts_files):
        ts    = np.load(ts_path).astype(np.float64)
        valid = np.std(ts, axis=0) > 1e-8
        ts    = ts[:, valid]
        if ts.shape[0] < short_vols + 1:
            continue

        ts_short  = ts[:short_vols, :]
        n_rois_s  = ts_short.shape[1]

        fc_short_vec = get_fc_matrix(ts_short, vectorized=True,  use_shrinkage=True)
        fc_ref_vec   = get_fc_matrix(ts,       vectorized=True,  use_shrinkage=True)
        fc_short_mat = get_fc_matrix(ts_short, vectorized=False, use_shrinkage=True)
        fc_ref_mat   = get_fc_matrix(ts,       vectorized=False, use_shrinkage=True)

        # ── Step 1: bootstrap-averaged FC matrix ──────────────────────────
        blk   = max(3, int(estimate_optimal_block_length(ts_short)))
        n_blk = max(1, short_vols // blk)
        rng   = np.random.default_rng(42 + sub_i)
        fc_bs_sum = np.zeros((n_rois_s, n_rois_s))
        for _ in range(config.n_bootstraps):
            idx  = block_bootstrap_indices(short_vols, blk, n_blk)
            ts_b = ts_short[idx[:short_vols], :]
            fc_bs_sum += get_fc_matrix(ts_b, vectorized=False, use_shrinkage=True)
        fc_bootstrap_mat = fc_bs_sum / config.n_bootstraps

        # ── Step 2: full pipeline → scalar rho_hat_T ──────────────────────
        result = run_bootstrap_prediction(
            ts_short, fc_ref_vec, config=config,
            correction_method=correction_method,
        )
        rho_hat_T = result.rho_hat_T
        r_fc      = float(np.corrcoef(fc_short_vec, fc_ref_vec)[0, 1])

        # ── Step 3: Fisher-z additive shift per connection ─────────────────
        # z_shift from pipeline's global correction → applied element-wise
        _clip = lambda x: np.clip(x, -0.9999, 0.9999)
        z_shift  = np.arctanh(_clip(rho_hat_T)) - np.arctanh(_clip(r_fc))
        z_bs_mat = np.arctanh(_clip(fc_bootstrap_mat))
        fc_bsnet_mat = np.tanh(z_bs_mat + z_shift)

        fc_short_list.append(fc_short_mat)
        fc_bsnet_list.append(fc_bsnet_mat)
        fc_ref_list.append(fc_ref_mat)

    if len(fc_short_list) < 3:
        raise ValueError(
            f"Too few valid subjects ({len(fc_short_list)}) in {cache_path}"
        )

    # Align n_rois (take minimum across subjects)
    n_rois = min(m.shape[0] for m in fc_short_list)
    def _trim(mats: list[np.ndarray]) -> np.ndarray:
        return np.stack([m[:n_rois, :n_rois] for m in mats])   # [N, r, r]

    fc_short_all = _trim(fc_short_list)   # [N, r, r]
    fc_bsnet_all = _trim(fc_bsnet_list)
    fc_ref_all   = _trim(fc_ref_list)
    N = fc_short_all.shape[0]

    # Vectorised cross-subject r per connection
    sh_flat  = fc_short_all.reshape(N, -1)
    bn_flat  = fc_bsnet_all.reshape(N, -1)
    ref_flat = fc_ref_all.reshape(N, -1)

    R_short_flat = _corr_vectorized(sh_flat,  ref_flat)
    R_bsnet_flat = _corr_vectorized(bn_flat,  ref_flat)

    R_short     = R_short_flat.reshape(n_rois, n_rois)
    R_bsnet     = R_bsnet_flat.reshape(n_rois, n_rois)
    fc_ref_mean = fc_ref_all.mean(axis=0)

    meta = {
        "n_subs":            N,
        "n_rois":            n_rois,
        "short_sec":         short_sec,
        "n_bootstraps":      config.n_bootstraps,
        "correction_method": correction_method,
    }

    # Save cache
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        R_short=R_short,
        R_bsnet=R_bsnet,
        fc_ref_mean=fc_ref_mean,
        meta=np.array(meta, dtype=object),
    )
    print(f"Cached → {npz_path.name}")

    return R_short, R_bsnet, fc_ref_mean, meta


def _load_exemplar_fc(
    atlas: str,
    sub_id: str,
    short_sec: int,
    tr: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load timeseries and compute short/reference FC for one subject.

    Args:
        atlas: Atlas name key.
        sub_id: Subject ID (e.g. "sub-01").
        short_sec: Short-scan duration in seconds.
        tr: Repetition time in seconds.

    Returns:
        (fc_short_vec, fc_ref_vec, meta) where meta contains n_pairs, r_fc,
        short_vols, ref_vols.
    """
    ts_path = CACHE_DIR / atlas / f"{sub_id}_{atlas}.npy"
    ts = np.load(ts_path).astype(np.float64)

    # Remove zero-variance ROIs
    valid = np.std(ts, axis=0) > 1e-8
    ts = ts[:, valid]

    ref_vols = ts.shape[0]
    short_vols = int(short_sec / tr)

    ts_short = ts[:short_vols, :]
    ts_ref = ts

    fc_short_vec = get_fc_matrix(ts_short, vectorized=True,  use_shrinkage=True)
    fc_ref_vec   = get_fc_matrix(ts_ref,   vectorized=True,  use_shrinkage=True)
    fc_short_mat = get_fc_matrix(ts_short, vectorized=False, use_shrinkage=True)
    fc_ref_mat   = get_fc_matrix(ts_ref,   vectorized=False, use_shrinkage=True)

    # Bootstrap-averaged FC matrix (bs-Net stabilized estimate)
    block_size = max(3, int(estimate_optimal_block_length(ts_short)))
    n_blocks   = max(1, short_vols // block_size)
    rng        = np.random.default_rng(42)
    fc_bs_sum  = np.zeros((ts.shape[1], ts.shape[1]))
    n_bs       = 100
    for _ in range(n_bs):
        idx  = block_bootstrap_indices(short_vols, block_size, n_blocks)
        ts_b = ts_short[idx[:short_vols], :]
        fc_bs_sum += get_fc_matrix(ts_b, vectorized=False, use_shrinkage=True)
    fc_bootstrap_mat = fc_bs_sum / n_bs

    r_fc = float(np.corrcoef(fc_short_vec, fc_ref_vec)[0, 1])

    return fc_short_vec, fc_ref_vec, fc_short_mat, fc_ref_mat, fc_bootstrap_mat, {
        "n_pairs": len(fc_ref_vec),
        "r_fc": r_fc,
        "short_vols": short_vols,
        "short_sec": short_sec,
        "ref_vols": ref_vols,
        "ref_sec": ref_vols * tr,
        "n_rois": ts.shape[1],
        "n_bootstraps": n_bs,
    }


def _diagonal_split_matrix(
    mat_upper: np.ndarray,
    mat_lower: np.ndarray,
) -> np.ndarray:
    """Combine two symmetric FC matrices into one with diagonal split.

    Upper triangle (i < j): mat_upper  (e.g. short-scan FC)
    Lower triangle (i > j): mat_lower  (e.g. reference FC)
    Diagonal (i == j):      NaN → rendered as white gap

    Args:
        mat_upper: Matrix for upper triangle.
        mat_lower: Matrix for lower triangle.

    Returns:
        Combined matrix (n × n) with NaN on diagonal.
    """
    n = mat_upper.shape[0]
    combined = np.full((n, n), np.nan)
    rows_u, cols_u = np.triu_indices(n, k=1)
    rows_l, cols_l = np.tril_indices(n, k=-1)
    combined[rows_u, cols_u] = mat_upper[rows_u, cols_u]
    combined[rows_l, cols_l] = mat_lower[rows_l, cols_l]
    return combined


def _draw_split_heatmap(
    ax: plt.Axes,
    mat_upper: np.ndarray,
    mat_lower: np.ndarray,
    label_upper: str,
    label_lower: str,
    title: str,
    vmin: float = -0.6,
    vmax: float = 0.8,
) -> None:
    """Draw a diagonal-split FC matrix heatmap.

    Upper triangle (above diagonal): mat_upper
    Lower triangle (below diagonal): mat_lower
    Labels are placed inside their respective triangle regions.

    Args:
        ax: Target axes.
        mat_upper: FC matrix for upper triangle.
        mat_lower: FC matrix for lower triangle.
        label_upper: Label text placed inside upper-triangle region.
        label_lower: Label text placed inside lower-triangle region.
        title: Panel title (short, e.g. "G1").
        vmin: Colormap minimum.
        vmax: Colormap maximum.
    """
    combined = _diagonal_split_matrix(mat_upper, mat_lower)
    n = combined.shape[0]

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="white")   # NaN diagonal → white

    im = ax.imshow(combined, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.82, label="Pearson r", pad=0.02)

    # Diagonal dividing line
    ax.plot([0, n - 1], [0, n - 1], color="#222222", lw=1.2, zorder=10)

    # Labels positioned inside each triangle in data coordinates
    # Upper triangle: top-right area  (x=75%, y=25% of n)
    # Lower triangle: bottom-left area (x=25%, y=75% of n)
    ax.text(n * 0.72, n * 0.22, label_upper,
            fontsize=8.5, va="center", ha="center", color="#111111",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.82, ec="none"),
            zorder=11)
    ax.text(n * 0.28, n * 0.78, label_lower,
            fontsize=8.5, va="center", ha="center", color="#111111",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.82, ec="none"),
            zorder=11)

    tick_pos = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xlabel("ROIs (Schaefer 200)", **FONT_AXIS)
    ax.set_ylabel("ROIs (Schaefer 200)", **FONT_AXIS)
    ax.set_title(title, loc="left", **FONT_PANEL)
    ax.tick_params(labelsize=FONT_TICK)


def _plot_panel_g(
    ax_g1: plt.Axes,
    ax_g2: plt.Axes,
    ax_g3: plt.Axes,
    R_short: np.ndarray,
    R_bsnet: np.ndarray,
    fc_ref_mean: np.ndarray,
    rel_meta: dict,
    df_record: pd.DataFrame,
    exemplar_sub: str,
) -> None:
    """Panel G (3 sub-axes): cross-subject reliability matrices + subject scatter.

    G1: Diagonal split — upper: R_short (cross-subject r, short→ref per conn.)
                         lower: fc_ref_mean
    G2: Diagonal split — upper: R_bsnet  (cross-subject r, bs-Net→ref per conn.)
                         lower: fc_ref_mean
    G3: Subject-level scatter — x=r_fc, y=ρ̂T at 2-min (N subjects).
        exemplar_sub highlighted; others faded.
        Coordinates from per-record CSV (seed-averaged per subject).

    Args:
        ax_g1, ax_g2: Axes for diagonal-split heatmaps.
        ax_g3:        Axes for subject-level scatter.
        R_short:      Cross-subject reliability matrix, short-long (n × n).
        R_bsnet:      Cross-subject reliability matrix, bs-Net-long (n × n).
        fc_ref_mean:  Mean reference FC matrix across subjects (n × n).
        rel_meta:     Metadata from _load_reliability_matrices().
        df_record:    Per-record CSV (all subjects, all durations, all seeds).
        exemplar_sub: Subject ID to highlight in G3.
    """
    N = rel_meta["n_subs"]

    # ── G1: Short-long reliability vs mean Reference ──────────────────────
    _draw_split_heatmap(
        ax_g1,
        mat_upper=R_short,
        mat_lower=fc_ref_mean,
        label_upper=f"r(Short, Ref)\nN={N} subjects",
        label_lower=f"Reference FC\n(mean, N={N})",
        title="G1",
        vmin=-0.6,
        vmax=0.8,
    )

    # ── G2: bs-Net-long reliability vs mean Reference ─────────────────────
    _draw_split_heatmap(
        ax_g2,
        mat_upper=R_bsnet,
        mat_lower=fc_ref_mean,
        label_upper=f"r(bs-Net, Ref)\nN={N} subjects",
        label_lower=f"Reference FC\n(mean, N={N})",
        title="G2",
        vmin=-0.6,
        vmax=0.8,
    )

    # ── G3: Subject-level summary scatter ────────────────────────────────
    # Pre-compute: seed-average r_fc / rho_hat_T per subject at 2-min
    sub_pts = (
        df_record[df_record["duration_sec"] == TWO_MIN_SEC]
        .groupby("sub_id", as_index=False)
        .agg(r_fc=("r_fc_raw", "mean"), rho_hat_T=("rho_hat_T", "mean"))
    )

    others    = sub_pts[sub_pts["sub_id"] != exemplar_sub]
    highlight = sub_pts[sub_pts["sub_id"] == exemplar_sub]

    # Faded background: all other subjects
    ax_g3.scatter(
        others["r_fc"], others["rho_hat_T"],
        color=C_BSNET, s=40, alpha=0.22, linewidths=0.6,
        edgecolors=C_BSNET, zorder=2,
        label=f"Other subjects (N={len(others)})",
    )
    # Highlighted: exemplar subject
    if len(highlight):
        ax_g3.scatter(
            highlight["r_fc"], highlight["rho_hat_T"],
            color=C_BSNET, s=120, alpha=1.0, linewidths=1.4,
            edgecolors="#1a5fa8", zorder=5,
            label=exemplar_sub,
        )

    # Identity line
    all_vals = pd.concat([sub_pts["r_fc"], sub_pts["rho_hat_T"]])
    lo = float(all_vals.min()) - 0.04
    hi = float(all_vals.max()) + 0.04
    ax_g3.plot([lo, hi], [lo, hi], color="#888888", lw=1.2, ls="--",
               zorder=1, 
               #label="y = x"
            )

    # Mean crosshairs + annotation
    mean_r   = float(sub_pts["r_fc"].mean())
    mean_rho = float(sub_pts["rho_hat_T"].mean())
    ax_g3.axhline(mean_rho, color=C_BSNET, lw=0.8, ls=":", alpha=0.55, zorder=3)
    ax_g3.axvline(mean_r,   color=C_RAW,   lw=0.8, ls=":", alpha=0.55, zorder=3)
    ax_g3.text(
        0.97, 0.05,
        f"mean $r_{{\\mathrm{{FC}}}}$ = {mean_r:.3f}\n"
        f"mean $\\hat{{\\rho}}_T$ = {mean_rho:.3f}",
        transform=ax_g3.transAxes, 
        fontsize=8,
        va="bottom", ha="right", family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.88, ec="#cccccc"),
    )

    ax_g3.set_xlim(lo, hi)
    ax_g3.set_ylim(lo, hi)
    ax_g3.set_aspect("equal", adjustable="box")
    ax_g3.set_xlabel(r"$r_{\mathrm{FC}}$ (Raw, 2 min)", **FONT_AXIS)
    ax_g3.set_ylabel(r"$\hat{\rho}_T$ (bs-Net, 2 min)", **FONT_AXIS)
    ax_g3.set_title("G3", loc="left", **FONT_PANEL)
    ax_g3.tick_params(labelsize=FONT_TICK)
    ax_g3.legend(fontsize=8, framealpha=0.9, loc="upper left")


# ── Main ──────────────────────────────────────────────────────────────────────


def plot_figure1(
    primary_atlas: str = PRIMARY_ATLAS,
    atlases: list[str] | None = None,
    exemplar_sub: str = EXEMPLAR_SUB,
    exemplar_short_sec: int = TWO_MIN_SEC,
) -> None:
    """Generate and save Figure 1 (ds007535, 7-panel, multi-atlas).

    Panels A–C use primary_atlas (Schaefer 200) with subject-level SD.
    Panels D–F compare all atlases.
    Panel G: exemplar FC scatter (real data, sub-01 at 2-min).

    Layout: 3×3 GridSpec
      Row 0: A  B  C
      Row 1: D  E  F
      Row 2: G  (full-width)

    Args:
        primary_atlas: Atlas used for Panels A–C and G.
        atlases: Atlas list for multi-atlas panels (D–F).
        exemplar_sub: Subject ID for Panel G scatter.
        exemplar_short_sec: Short-scan duration for Panel G (seconds).

    Raises:
        FileNotFoundError: If any required CSV or timeseries cache is missing.
    """
    if atlases is None:
        atlases = list(ATLAS_META.keys())

    # Load & aggregate all atlases from per-record CSVs (proper subject-level SD)
    atlas_records: dict[str, pd.DataFrame] = {
        k: _load_per_record(k) for k in atlases
    }
    atlas_computed: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        k: _compute_agg(df) for k, df in atlas_records.items()
    }

    atlas_sub_durs = {k: v[0] for k, v in atlas_computed.items()}
    atlas_aggs = {k: v[1] for k, v in atlas_computed.items()}

    primary_sub_dur = atlas_sub_durs[primary_atlas]
    primary_agg = atlas_aggs[primary_atlas]

    # Cross-subject reliability matrices for G1/G2 (cached after first run)
    R_short, R_bsnet, fc_ref_mean, rel_meta = _load_reliability_matrices(
        primary_atlas, exemplar_short_sec
    )
    print(
        f"  N={rel_meta['n_subs']} subjects, "
        f"n_rois={rel_meta['n_rois']}, "
        f"short={rel_meta['short_sec']}s, "
        f"method={rel_meta['correction_method']}"
    )

    df_primary_record = atlas_records[primary_atlas]

    apply_bsnet_theme()

    # ── 3×3 GridSpec layout ──────────────────────────────────────────────────
    # Row 0: A  B  C  (Schaefer200 primary)
    # Row 1: D  E  F  (multi-atlas)
    # Row 2: G1 G2 G3 (exemplar FC heatmaps + scatter)
    fig = plt.figure(figsize=FIGSIZE["3x3"])
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        height_ratios=[1, 1, 1.1],
        hspace=0.50,
        wspace=0.34,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    ax_g1 = fig.add_subplot(gs[2, 0])   # FC_short heatmap
    ax_g2 = fig.add_subplot(gs[2, 1])   # FC_ref heatmap
    ax_g3 = fig.add_subplot(gs[2, 2])   # scatter

    _plot_panel_a(ax_a, atlas_sub_durs, atlas_aggs)
    _plot_panel_b(ax_b, atlas_sub_durs)
    _plot_panel_c(ax_c, atlas_aggs)
    _plot_panel_d(ax_d, atlas_aggs)
    _plot_panel_e(ax_e, atlas_aggs)
    _plot_panel_f(ax_f, atlas_sub_durs, atlas_aggs)
    _plot_panel_g(
        ax_g1, ax_g2, ax_g3,
        R_short, R_bsnet, fc_ref_mean,
        rel_meta, df_primary_record, exemplar_sub,
    )

    save_figure(fig, OUTPUT_NAME)
    print(f"Saved: artifacts/reports/{OUTPUT_NAME}  |  docs/figure/{OUTPUT_NAME}")


if __name__ == "__main__":
    plot_figure1()
