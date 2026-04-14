"""FC Intuition Figure: Making BS-NET's ρ̂T improvement tangible.

Storyline:
  Row 1 — "2min FC is noisy → 4min is better → Reference is the target"
    A: FC(2 min)       — the problem
    B: FC(4 min)       — BS-NET equivalent level
    C: FC(reference)   — the target
  Row 2 — Quantitative evidence for the A→B→C narrative
    D: Dual scatter (2min & 4min vs reference) — attenuation reduction
    E: Duration sweep + equivalent scan duration arrow — why 4 min
    F: Network-level bars (2min / 4min / ref) — per-network quantification

Key message: BS-NET makes a 2-minute scan work like a 4-minute scan (ρ̂T=0.825 ≈ r_FC@240s).

Data: ds000243 XCP-D timeseries (4S256Parcels → Schaefer200 cortical subset)

Usage:
    python src/visualization/plot_fc_intuition.py --n-subjects 52 --n-jobs 8
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from src.data.data_loader import get_fc_matrix
from src.visualization.plot_network_visualization import (
    ATLAS_TXT,
    N_CORTICAL,
    NETWORK_COLORS,
    get_network_boundaries,
    parse_schaefer_networks,
    sort_rois_by_network,
)
from src.visualization.style import (
    CONDITION_PALETTE,
    FONT,
    apply_bsnet_theme,
    save_figure,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

NETWORK_NAMES_FULL = list(NETWORK_COLORS.keys())


# ── Data loading ──────────────────────────────────────────────────────────────


def load_three_condition_data(
    ts_dir: Path,
    n_subjects: int,
    short_samples: int,
    equiv_samples: int,
    n_jobs: int = 1,
) -> dict:
    """Load timeseries, compute 3 FC conditions: 2min, 4min(equiv), reference.

    Args:
        ts_dir: Directory with .npy files.
        n_subjects: Max subjects.
        short_samples: TRs for short scan (e.g. 48 = 2min).
        equiv_samples: TRs for equivalent duration (e.g. 96 = 4min).
        n_jobs: Parallel workers.

    Returns:
        dict with fc_short, fc_equiv, fc_ref stacks + metadata.
    """
    import glob

    from joblib import Parallel, delayed

    files = sorted(glob.glob(str(ts_dir / "sub-*_4s256parcels.npy")))[:n_subjects]
    if not files:
        raise FileNotFoundError(f"No timeseries in {ts_dir}")

    def _process(fp):
        ts = np.load(fp).astype(np.float64)[:, :N_CORTICAL]

        # Short scan (2 min)
        ts_short = ts[:short_samples]
        # Equivalent scan (4 min)
        ts_equiv = ts[:equiv_samples]
        # Reference: remaining timepoints AFTER equiv window (non-overlapping)
        ts_ref = ts[equiv_samples:]

        fc_short = get_fc_matrix(ts_short, vectorized=False, use_shrinkage=False)
        fc_equiv = get_fc_matrix(ts_equiv, vectorized=False, use_shrinkage=False)
        fc_ref = get_fc_matrix(ts_ref, vectorized=False, use_shrinkage=True)

        return fc_short, fc_equiv, fc_ref

    print(f"  Loading {len(files)} subjects (short={short_samples}, equiv={equiv_samples} TRs)")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process)(fp) for fp in files
    )

    fc_short_stack = np.array([r[0] for r in results])
    fc_equiv_stack = np.array([r[1] for r in results])
    fc_ref_stack = np.array([r[2] for r in results])

    # Per-subject r_FC and group similarity
    triu = np.triu_indices(N_CORTICAL, k=1)
    grp_short = fc_short_stack.mean(axis=0)
    grp_ref = fc_ref_stack.mean(axis=0)

    r_fc_short = np.array([
        np.corrcoef(fc_short_stack[i][triu], fc_ref_stack[i][triu])[0, 1]
        for i in range(len(results))
    ])
    r_fc_equiv = np.array([
        np.corrcoef(fc_equiv_stack[i][triu], fc_ref_stack[i][triu])[0, 1]
        for i in range(len(results))
    ])
    sim_ref = np.array([
        np.corrcoef(fc_ref_stack[i][triu], grp_ref[triu])[0, 1]
        for i in range(len(results))
    ])

    # Select most representative subject:
    # Minimize deviation from group mean + maximize group FC similarity
    mean_rs = r_fc_short.mean()
    mean_re = r_fc_equiv.mean()
    mean_dr = (r_fc_equiv - r_fc_short).mean()
    rep_scores = np.array([
        (abs(r_fc_short[i] - mean_rs) + abs(r_fc_equiv[i] - mean_re)
         + abs((r_fc_equiv[i] - r_fc_short[i]) - mean_dr)) / 3
        - sim_ref[i]
        for i in range(len(results))
    ])
    best_idx = int(np.argmin(rep_scores))
    pid = Path(files[best_idx]).name.split("_")[0]

    print(f"  Representative: {pid} (sim_ref={sim_ref[best_idx]:.3f}, "
          f"r_short={r_fc_short[best_idx]:.3f}, r_equiv={r_fc_equiv[best_idx]:.3f})")
    print(f"  Mean r_FC: short={mean_rs:.3f}±{r_fc_short.std():.3f}, "
          f"equiv={mean_re:.3f}±{r_fc_equiv.std():.3f}")

    return {
        "fc_short": fc_short_stack,
        "fc_equiv": fc_equiv_stack,
        "fc_ref": fc_ref_stack,
        "r_fc_short": r_fc_short,
        "r_fc_equiv": r_fc_equiv,
        "best_idx": best_idx,
        "pid": pid,
        "files": files,
    }


# ── Panel functions ───────────────────────────────────────────────────────────


def panel_fc_matrix(
    ax: plt.Axes,
    fc: np.ndarray,
    sort_order: np.ndarray,
    boundaries: list[int],
    net_indices: list[int],
    title: str = "",
    show_cbar: bool = False,
) -> None:
    """Plot a single RSN-sorted FC matrix."""
    vmin, vmax = -0.8, 0.8
    fc_sorted = fc[np.ix_(sort_order, sort_order)]
    im = ax.imshow(fc_sorted, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="none")
    for b in boundaries:
        ax.axhline(y=b - 0.5, color="k", linewidth=1.0, alpha=0.6)
        ax.axvline(x=b - 0.5, color="k", linewidth=1.0, alpha=0.6)
    ax.set_title(title, fontsize=FONT["title"], fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # Network labels on left
    net_sorted = np.array(net_indices)[sort_order]
    net_colors = list(NETWORK_COLORS.values())
    for idx, name in enumerate(NETWORK_NAMES_FULL):
        mask = net_sorted == idx
        positions = np.where(mask)[0]
        if len(positions) > 0:
            mid = positions[len(positions) // 2]
            ax.text(-3, mid, name[:3], fontsize=6, ha="right", va="center",
                    color=net_colors[idx], fontweight="bold")

    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("FC (Pearson r)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)


def panel_dual_scatter(
    ax: plt.Axes,
    fc_short: np.ndarray,
    fc_equiv: np.ndarray,
    fc_ref: np.ndarray,
    net_indices: list[int],
    short_sec: float,
    equiv_sec: float,
    label: str = "D",
) -> None:
    """Dual scatter: 2min vs ref AND 4min vs ref, showing attenuation reduction."""
    triu = np.triu_indices(N_CORTICAL, k=1)
    short_vals = fc_short[triu]
    equiv_vals = fc_equiv[triu]
    ref_vals = fc_ref[triu]

    # Subsample for visual clarity (19900 × 2 = too dense)
    rng = np.random.RandomState(42)
    n_edges = len(short_vals)
    idx_sub = rng.choice(n_edges, size=min(5000, n_edges), replace=False)

    # 2min scatter (amber, background)
    ax.scatter(
        short_vals[idx_sub], ref_vals[idx_sub],
        c=CONDITION_PALETTE["raw"], s=3, alpha=0.15, rasterized=True, zorder=1,
        label=None,
    )
    # 4min scatter (blue, foreground)
    ax.scatter(
        equiv_vals[idx_sub], ref_vals[idx_sub],
        c=CONDITION_PALETTE["bsnet"], s=3, alpha=0.15, rasterized=True, zorder=2,
        label=None,
    )

    # Diagonal
    lims = [-0.5, 0.9]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4, zorder=0)

    # Regression lines
    from numpy.polynomial.polynomial import polyfit

    coeffs_s = polyfit(short_vals, ref_vals, 1)
    coeffs_e = polyfit(equiv_vals, ref_vals, 1)
    x_fit = np.linspace(-0.5, 0.9, 100)

    ax.plot(x_fit, coeffs_s[0] + coeffs_s[1] * x_fit,
            color=CONDITION_PALETTE["raw"], linewidth=2.5, alpha=0.9, zorder=3,
            label=f"{short_sec/60:.0f} min  (slope={coeffs_s[1]:.2f})")
    ax.plot(x_fit, coeffs_e[0] + coeffs_e[1] * x_fit,
            color=CONDITION_PALETTE["bsnet"], linewidth=2.5, alpha=0.9, zorder=4,
            label=f"{equiv_sec/60:.0f} min  (slope={coeffs_e[1]:.2f})")

    # Correlation annotations
    r_short = np.corrcoef(short_vals, ref_vals)[0, 1]
    r_equiv = np.corrcoef(equiv_vals, ref_vals)[0, 1]
    ax.text(
        0.05, 0.95,
        f"$r_{{FC}}$({short_sec/60:.0f}min) = {r_short:.3f}\n"
        f"$r_{{FC}}$({equiv_sec/60:.0f}min) = {r_equiv:.3f}\n"
        f"Δ$r_{{FC}}$ = +{r_equiv - r_short:.3f}",
        transform=ax.transAxes, fontsize=9.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("FC$_{short}$", fontsize=FONT["axis_label"])
    ax.set_ylabel("FC$_{reference}$", fontsize=FONT["axis_label"])
    ax.set_title(f"{label}. Edge-level Attenuation",
                 fontsize=FONT["title"], fontweight="bold")
    ax.set_xlim(-0.5, 0.9)
    ax.set_ylim(-0.5, 0.9)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85,
              markerscale=0, handlelength=2)
    ax.grid(True, alpha=0.2)


def panel_duration_arrow(
    ax: plt.Axes,
    sweep_csv: Path,
    sweep_per_subject_csv: Path | None = None,
    short_sec: float = 120.0,
    equiv_sec: float = 240.0,
    label: str = "E",
) -> None:
    """Duration sweep curve with equivalent scan duration arrow and ±SD shading."""
    import pandas as pd

    df = pd.read_csv(sweep_csv)
    durations = df["duration_sec"].values
    r_fc_vals = df["r_fc_mean"].values
    rho_hat_vals = df["rho_hat_T_mean"].values

    # Compute subject-level SD if per-subject CSV available
    r_fc_sd = np.zeros_like(r_fc_vals)
    rho_hat_sd = np.zeros_like(rho_hat_vals)
    if sweep_per_subject_csv is not None and sweep_per_subject_csv.exists():
        df_sub = pd.read_csv(sweep_per_subject_csv)
        subj = df_sub.groupby(["sub_id", "duration_sec"]).agg(
            r_fc=("r_fc_raw", "mean"),
            rho_hat=("rho_hat_T", "mean"),
        ).reset_index()
        for idx, dur in enumerate(durations):
            d = subj[subj["duration_sec"] == dur]
            if len(d) > 1:
                r_fc_sd[idx] = d["r_fc"].std()
                rho_hat_sd[idx] = d["rho_hat"].std()

    # Find crossover point where ρ̂T drops below r_FC
    diff = rho_hat_vals - r_fc_vals
    cross_idx = None
    for ci in range(len(diff) - 1):
        if diff[ci] >= 0 and diff[ci + 1] < 0:
            # Linear interpolation for precise crossover
            frac = diff[ci] / (diff[ci] - diff[ci + 1])
            cross_dur = durations[ci] + frac * (durations[ci + 1] - durations[ci])
            cross_idx = ci + 1
            break

    # Split into valid (solid) and diminishing-returns (dashed) segments
    if cross_idx is not None:
        valid_mask = np.arange(len(durations)) < cross_idx
        tail_mask = np.arange(len(durations)) >= (cross_idx - 1)  # overlap by 1 for continuity
    else:
        valid_mask = np.ones(len(durations), dtype=bool)
        tail_mask = np.zeros(len(durations), dtype=bool)

    # r_FC curve: always solid (it's observed data)
    ax.plot(durations, r_fc_vals, "o-",
            color=CONDITION_PALETTE["raw"], linewidth=2.5, markersize=7,
            label="$r_{FC}$ (observed)", zorder=2)
    if r_fc_sd.any():
        ax.fill_between(durations, r_fc_vals - r_fc_sd, r_fc_vals + r_fc_sd,
                         color=CONDITION_PALETTE["raw"], alpha=0.15, zorder=1)

    # ρ̂T curve: solid where valid, dashed in diminishing-returns zone
    ax.plot(durations[valid_mask], rho_hat_vals[valid_mask], "s-",
            color=CONDITION_PALETTE["bsnet"], linewidth=2.5, markersize=7,
            label="$\\hat{\\rho}_T$ (BS-NET)", zorder=2)
    if tail_mask.any():
        ax.plot(durations[tail_mask], rho_hat_vals[tail_mask], "s--",
                color=CONDITION_PALETTE["bsnet"], linewidth=2.0, markersize=6,
                alpha=0.5, zorder=2)
    if rho_hat_sd.any():
        ax.fill_between(durations[valid_mask],
                         (rho_hat_vals - rho_hat_sd)[valid_mask],
                         (rho_hat_vals + rho_hat_sd)[valid_mask],
                         color=CONDITION_PALETTE["bsnet"], alpha=0.15, zorder=1)
        if tail_mask.any():
            ax.fill_between(durations[tail_mask],
                             (rho_hat_vals - rho_hat_sd)[tail_mask],
                             (rho_hat_vals + rho_hat_sd)[tail_mask],
                             color=CONDITION_PALETTE["bsnet"], alpha=0.06, zorder=1)

    # Key point: 120s
    idx_120 = np.argmin(np.abs(durations - short_sec))
    r_fc_120 = r_fc_vals[idx_120]
    rho_hat_120 = rho_hat_vals[idx_120]

    # Equivalent duration via interpolation
    f_interp = interpolate.interp1d(r_fc_vals, durations, kind="linear",
                                     fill_value="extrapolate")
    equiv_duration = float(f_interp(rho_hat_120))

    # Highlight 120s points
    ax.plot(short_sec, r_fc_120, "o", color=CONDITION_PALETTE["raw"],
            markersize=13, markeredgecolor="black", markeredgewidth=2, zorder=5)
    ax.plot(short_sec, rho_hat_120, "s", color=CONDITION_PALETTE["bsnet"],
            markersize=13, markeredgecolor="black", markeredgewidth=2, zorder=5)

    # Vertical arrow: r_FC → ρ̂T
    ax.annotate(
        "", xy=(short_sec, rho_hat_120), xytext=(short_sec, r_fc_120),
        arrowprops=dict(arrowstyle="-|>", color="#d7191c", lw=2.5,
                        mutation_scale=15),
        zorder=4,
    )
    ax.text(
        short_sec + 10, (r_fc_120 + rho_hat_120) / 2,
        f"Δ = +{rho_hat_120 - r_fc_120:.3f}",
        fontsize=9.5, color="#d7191c", fontweight="bold", va="center",
    )

    # Horizontal arrow: 120s → equiv duration
    ax.annotate(
        "", xy=(equiv_duration, rho_hat_120), xytext=(short_sec, rho_hat_120),
        arrowprops=dict(arrowstyle="-|>", color=CONDITION_PALETTE["bsnet"],
                        lw=2.5, linestyle="--", mutation_scale=15),
        zorder=4,
    )

    # Drop-down line from equiv point
    ax.plot([equiv_duration, equiv_duration], [0.38, rho_hat_120],
            ":", color=CONDITION_PALETTE["bsnet"], linewidth=1.5, alpha=0.5)

    # Shaded region between curves at 120s
    ax.fill_between(
        [short_sec - 5, short_sec + 5],
        [r_fc_120, r_fc_120], [rho_hat_120, rho_hat_120],
        color="#d7191c", alpha=0.08,
    )

    # Labels
    equiv_min = equiv_duration / 60
    ax.text(
        equiv_duration, 0.40,
        f"≈ {equiv_min:.0f} min",
        fontsize=12, color=CONDITION_PALETTE["bsnet"], fontweight="bold",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F0FE",
                  edgecolor=CONDITION_PALETTE["bsnet"], alpha=0.9),
    )
    ax.text(
        short_sec, 0.40,
        "2 min",
        fontsize=12, color=CONDITION_PALETTE["raw"], fontweight="bold",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0",
                  edgecolor=CONDITION_PALETTE["raw"], alpha=0.9),
    )

    ax.set_xlabel("Scan Duration (sec)", fontsize=FONT["axis_label"])
    ax.set_ylabel("Correlation with Reference FC", fontsize=FONT["axis_label"])
    ax.set_title(f"{label}. Equivalent Scan Duration",
                 fontsize=FONT["title"], fontweight="bold")
    ax.set_ylim(0.38, 1.0)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax.grid(True, alpha=0.2)


def panel_paired_bsnet(
    ax: plt.Axes,
    sweep_per_subject_csv: Path,
    short_sec: float = 120.0,
    label: str = "F",
) -> None:
    """Paired subject-level: r_FC(2min) vs ρ̂T(BS-NET at 2min)."""
    import pandas as pd

    df = pd.read_csv(sweep_per_subject_csv)
    d = df[df["duration_sec"] == short_sec].groupby("sub_id").agg(
        r_fc=("r_fc_raw", "mean"),
        rho_hat=("rho_hat_T", "mean"),
    ).reset_index()

    r_fc_vals = d["r_fc"].values
    rho_hat_vals = d["rho_hat"].values
    n_subs = len(d)
    improved = rho_hat_vals > r_fc_vals
    n_improved = improved.sum()
    pct_improved = 100 * n_improved / n_subs

    # Paired lines
    for i in range(n_subs):
        color = "#2ecc71" if improved[i] else "#e74c3c"
        ax.plot([0, 1], [r_fc_vals[i], rho_hat_vals[i]],
                color=color, linewidth=0.8, alpha=0.4, zorder=1)

    # Dots with jitter
    jitter = 0.03
    rng = np.random.RandomState(42)
    jx_0 = rng.uniform(-jitter, jitter, n_subs)
    jx_1 = rng.uniform(-jitter, jitter, n_subs)

    ax.scatter(jx_0, r_fc_vals, c=CONDITION_PALETTE["raw"],
               s=30, alpha=0.7, zorder=2, edgecolors="white", linewidths=0.3)
    ax.scatter(1 + jx_1, rho_hat_vals, c=CONDITION_PALETTE["bsnet"],
               s=30, alpha=0.7, zorder=2, edgecolors="white", linewidths=0.3)

    # Group means ± SD
    for x_pos, vals in [(0, r_fc_vals), (1, rho_hat_vals)]:
        mean_v = vals.mean()
        sd_v = vals.std()
        ax.plot(x_pos, mean_v, "D", color="#d7191c", markersize=10,
                markeredgecolor="black", markeredgewidth=1.5, zorder=5)
        ax.errorbar(x_pos, mean_v, yerr=sd_v, color="black",
                    linewidth=2.0, capsize=6, capthick=2.0, zorder=4)

    # Annotations
    delta_mean = (rho_hat_vals - r_fc_vals).mean()
    ax.text(
        0.5, 0.97,
        f"Δ = +{delta_mean:.3f}\n"
        f"{n_improved}/{n_subs} improved ({pct_improved:.0f}%)",
        transform=ax.transAxes, fontsize=10, ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    short_min = short_sec / 60
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"$r_{{FC}}$ ({short_min:.0f} min)", "$\\hat{{\\rho}}_T$ (BS-NET)"],
        fontsize=10, fontweight="bold",
    )
    ax.set_ylabel("Similarity to Reference FC", fontsize=FONT["axis_label"])
    ax.set_title(f"{label}. BS-NET Correction (N={n_subs})",
                 fontsize=FONT["title"], fontweight="bold")
    ax.set_xlim(-0.4, 1.4)
    ax.grid(True, axis="y", alpha=0.2)

    # Footnote: explain N difference if less than expected
    if n_subs < 52:
        ax.text(
            0.5, 0.02,
            f"* {52 - n_subs} subjects excluded\n  (total scan < 600 s)",
            transform=ax.transAxes, fontsize=7, ha="center", va="bottom",
            color="#666666", fontstyle="italic",
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="FC Intuition Figure")
    parser.add_argument("--n-subjects", type=int, default=52)
    parser.add_argument("--short-sec", type=float, default=120.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    print("=== FC Intuition Figure v2 (ds000243) ===")

    ts_dir = Path("data/ds000243/timeseries_cache_xcpd/4s256parcels")
    sweep_csv = Path("data/ds000243/results/"
                     "ds000243_xcpd_duration_sweep_4s256parcels_aggregated.csv")
    tr = 2.5
    short_samples = int(args.short_sec / tr)  # 48 TRs

    # Compute equivalent duration from sweep data
    import pandas as pd
    df_sweep = pd.read_csv(sweep_csv)
    idx_120 = np.argmin(np.abs(df_sweep["duration_sec"].values - args.short_sec))
    rho_hat_120 = df_sweep["rho_hat_T_mean"].values[idx_120]
    f_interp = interpolate.interp1d(
        df_sweep["r_fc_mean"].values, df_sweep["duration_sec"].values,
        kind="linear", fill_value="extrapolate",
    )
    equiv_sec = float(f_interp(rho_hat_120))
    equiv_samples = int(equiv_sec / tr)
    print(f"  ρ̂T at {args.short_sec:.0f}s = {rho_hat_120:.4f} → equiv {equiv_sec:.0f}s "
          f"({equiv_samples} TRs)")

    # 1. Parse atlas
    print("Parsing Schaefer atlas...")
    net_indices = parse_schaefer_networks(ATLAS_TXT, N_CORTICAL)
    sort_order = sort_rois_by_network(net_indices)
    boundaries = get_network_boundaries(np.array(net_indices)[sort_order].tolist())

    # 2. Load 3-condition data
    data = load_three_condition_data(
        ts_dir, args.n_subjects, short_samples, equiv_samples, args.n_jobs,
    )
    bi = data["best_idx"]

    # 3. Group averages
    fc_short_grp = data["fc_short"].mean(axis=0)
    fc_equiv_grp = data["fc_equiv"].mean(axis=0)
    fc_ref_grp = data["fc_ref"].mean(axis=0)

    # 4. Plot — 3 rows × 3 cols
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    apply_bsnet_theme()
    fig = plt.figure(figsize=(24, 22))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.30, wspace=0.32,
                           height_ratios=[1, 1, 1])

    short_min = args.short_sec / 60
    equiv_min = equiv_sec / 60
    ref_min = (463 - equiv_samples) * tr / 60
    n_sub = len(data["files"])
    pid = data["pid"]

    # ── Row 1 (A–C): Individual FC matrices ──
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_fc_matrix(
        ax_a, data["fc_short"][bi], sort_order, boundaries, net_indices,
        title=f"A. Raw FC ({short_min:.0f} min)  [{pid}]",
    )
    panel_fc_matrix(
        ax_b, data["fc_equiv"][bi], sort_order, boundaries, net_indices,
        title=f"B. FC at BS-NET Equiv. ({equiv_min:.0f} min)  [{pid}]",
    )
    panel_fc_matrix(
        ax_c, data["fc_ref"][bi], sort_order, boundaries, net_indices,
        title=f"C. Reference FC ({ref_min:.0f} min)  [{pid}]",
    )

    # Shared colorbar for Row 1
    norm = mcolors.Normalize(vmin=-0.8, vmax=0.8)
    sm = cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    cbar1 = fig.colorbar(sm, ax=[ax_a, ax_b, ax_c], shrink=0.6, pad=0.02,
                         location="right", label="FC (Pearson r)")
    cbar1.ax.tick_params(labelsize=8)

    # ── Row 2 (D–F): Group FC matrices ──
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    panel_fc_matrix(
        ax_d, fc_short_grp, sort_order, boundaries, net_indices,
        title=f"D. Raw FC ({short_min:.0f} min)  [N={n_sub}]",
    )
    panel_fc_matrix(
        ax_e, fc_equiv_grp, sort_order, boundaries, net_indices,
        title=f"E. FC at BS-NET Equiv. ({equiv_min:.0f} min)  [N={n_sub}]",
    )
    panel_fc_matrix(
        ax_f, fc_ref_grp, sort_order, boundaries, net_indices,
        title=f"F. Reference FC ({ref_min:.0f} min)  [N={n_sub}]",
    )

    # Shared colorbar for Row 2
    sm2 = cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=[ax_d, ax_e, ax_f], shrink=0.6, pad=0.02,
                         location="right", label="FC (Pearson r)")
    cbar2.ax.tick_params(labelsize=8)

    # ── Row 3 (G–I): Quantitative evidence ──
    ax_g = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[2, 1])
    ax_i = fig.add_subplot(gs[2, 2])

    panel_dual_scatter(
        ax_g,
        fc_short=data["fc_short"][bi],
        fc_equiv=data["fc_equiv"][bi],
        fc_ref=data["fc_ref"][bi],
        net_indices=net_indices,
        short_sec=args.short_sec,
        equiv_sec=equiv_sec,
        label="G",
    )

    sweep_per_sub_csv = Path("data/ds000243/results/"
                             "ds000243_xcpd_duration_sweep_4s256parcels.csv")
    panel_duration_arrow(
        ax_h, sweep_csv,
        sweep_per_subject_csv=sweep_per_sub_csv,
        short_sec=args.short_sec,
        equiv_sec=equiv_sec,
        label="H",
    )

    panel_paired_bsnet(
        ax_i,
        sweep_per_subject_csv=sweep_per_sub_csv,
        short_sec=args.short_sec,
        label="I",
    )

    save_figure(fig, "Fig1_FC_Intuition.png")
    print(f"\nFigure saved: docs/figure/Figure_FC_Intuition.png")

    # Summary
    triu = np.triu_indices(N_CORTICAL, k=1)
    r_grp_short = np.corrcoef(fc_short_grp[triu], fc_ref_grp[triu])[0, 1]
    r_grp_equiv = np.corrcoef(fc_equiv_grp[triu], fc_ref_grp[triu])[0, 1]
    r_ind_s = data["r_fc_short"][bi]
    r_ind_e = data["r_fc_equiv"][bi]
    print(f"\n--- Summary ---")
    print(f"  Equiv duration:  {equiv_sec:.0f}s ({equiv_min:.1f} min)")
    print(f"  Representative:  {pid} (r_short={r_ind_s:.3f}, r_equiv={r_ind_e:.3f}, Δ={r_ind_e-r_ind_s:.3f})")
    print(f"  Group r_FC:      short={r_grp_short:.4f}, equiv={r_grp_equiv:.4f}")
    print(f"  Indiv mean:      short={data['r_fc_short'].mean():.3f}±{data['r_fc_short'].std():.3f}, "
          f"equiv={data['r_fc_equiv'].mean():.3f}±{data['r_fc_equiv'].std():.3f}")
    print(f"  Δr_FC (mean):    +{(data['r_fc_equiv'] - data['r_fc_short']).mean():.4f}")


if __name__ == "__main__":
    main()
