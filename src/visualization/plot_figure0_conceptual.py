"""Figure 0: Conceptual Framework — BS-NET.

Schematic introduction figure using nilearn brain visualizations.

Layout (2 rows):
  Row 1: [A-long: connectome stable] [A-short: connectome noisy] [B: BOLD multi-freq + brain inset]
  Row 2: [C-fix: phase space fixed]  [C-sw: phase space sliding] [D: pipeline] [E: use cases]

Requires nilearn + local Schaefer atlas at atlas/schaefer_2018/.

Usage:
    cd /path/to/bsNet
    python src/visualization/plot_figure0_conceptual.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.visualization.style import (
    FONT_AXIS, FONT_PANEL, FONT_TICK,
    apply_bsnet_theme, save_figure,
)

warnings.filterwarnings("ignore")

# ── Colors ────────────────────────────────────────────────────────────────────
C_BSNET  = "#4A90E2"
C_RAW    = "#fdae61"
C_REF    = "#95a5a6"
C_DARK   = "#2c3e50"
C_GOLD   = "#c8a951"
C_GREEN  = "#2e8a5e"

RNG = np.random.default_rng(42)

ATLAS_NII = Path("atlas/schaefer_2018/"
                 "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")

NETWORK_COLS = [C_BSNET, C_GREEN, C_GOLD, "#9b59b6", "#e74c3c"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_coords() -> np.ndarray:
    from nilearn import plotting
    return plotting.find_parcellation_cut_coords(str(ATLAS_NII))


def _make_adj(n: int, density: float, noise: float,
              seed: int = 0) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    base = rng.standard_normal((n, n))
    base = (base + base.T) / 2
    base += noise * rng.standard_normal((n, n))
    np.fill_diagonal(base, 0)
    thr = np.quantile(np.abs(base), 1 - density)
    return np.where(np.abs(base) >= thr, base, 0.0)


def _bold(t: np.ndarray,
          freqs: list[tuple[float, float, float]]) -> np.ndarray:
    sig = np.zeros_like(t)
    for f, a, phi in freqs:
        sig += a * np.sin(2 * np.pi * f * t + phi)
    return sig


def _phase_xy(sig: np.ndarray, lag: int = 6):
    return sig[:-lag], sig[lag:]


# ── Panel A ───────────────────────────────────────────────────────────────────

def panel_a(ax_long: plt.Axes, ax_short: plt.Axes,
            coords: np.ndarray) -> None:
    """Long (stable) vs Short (noisy) connectome on glass brain."""
    from nilearn import plotting

    n = 50
    c = coords[:n]

    # Long: clean positive-dominant edges → blue nodes, thin edges
    adj_long  = _make_adj(n, density=0.05, noise=0.05, seed=1)
    # Short: noisy, mixed-sign edges → amber nodes, thicker chaotic edges
    adj_short = _make_adj(n, density=0.05, noise=2.50, seed=2)

    specs = [
        (ax_long,  adj_long,
         "Long scan (~15 min)\nStable FC",
         C_BSNET, C_BSNET,
         {"alpha": 0.15, "linewidth": 0.15}),
        (ax_short, adj_short,
         "Short scan (< 5 min)\nNoisy FC",
         C_RAW, C_RAW,
         {"alpha": 0.10, "linewidth": 0.30}),
    ]

    for ax, adj, title, ncol, ecol, ekw in specs:
        plotting.plot_connectome(
            adj, c,
            node_size=10,
            node_color=ncol,
            edge_threshold="90%",
            display_mode="l",
            colorbar=False,
            axes=ax,
            edge_kwargs=ekw,
        )
        ax.set_title(title, fontsize=9.5, color=ecol,
                     fontweight="bold", pad=5)
        for sp in ax.spines.values():
            sp.set_color(ecol); sp.set_linewidth(2.2)


# ── Panel B ───────────────────────────────────────────────────────────────────

def panel_b(ax: plt.Axes, coords: np.ndarray) -> None:
    """BOLD multi-frequency decomposition + glass-brain inset."""
    from nilearn import plotting

    t = np.linspace(0, 120, 600)
    networks = [
        ("DMN",    0.013, 1.0, 0.0,  C_BSNET),
        ("DAN",    0.022, 0.7, 1.1,  C_GREEN),
        ("SN",     0.031, 0.5, 2.3,  C_GOLD),
        ("Visual", 0.018, 0.4, 0.7,  "#9b59b6"),
        ("SMN",    0.027, 0.35, 1.8, "#e74c3c"),
    ]

    bold = np.zeros_like(t)
    for _, f, a, phi, _ in networks:
        bold += a * np.sin(2 * np.pi * f * t + phi)
    bold += RNG.standard_normal(len(t)) * 0.12

    # Observed BOLD (thick dark)
    ax.plot(t, bold, color=C_DARK, lw=2.0, alpha=0.90, zorder=5,
            label="BOLD (observed)")

    # Component sinusoids (thin dashed, colour-coded)
    for name, f, a, phi, col in networks:
        comp = a * np.sin(2 * np.pi * f * t + phi)
        ax.plot(t, comp, color=col, lw=1.1, alpha=0.55, ls="--", zorder=3)
        ax.text(122, comp[-1], name, color=col, fontsize=7.5, va="center")

    ax.set_xlim(0, 120)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_xlabel("Time (s)", **FONT_AXIS)
    ax.set_ylabel("Signal (a.u.)", **FONT_AXIS)
    ax.text(4, ax.get_ylim()[0] * 0.78,
            "0.01 – 0.1 Hz\n(multi-network)",
            fontsize=7.5, color="#555", style="italic",
            bbox=dict(fc="white", alpha=0.85, ec="#ccc", pad=2))
    ax.set_title("B.  BOLD = Multi-frequency Superposition\n"
                 "→  Direct prediction fails (non-stationary)",
                 **FONT_PANEL)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.88)

    # ── Glass-brain inset (upper-right corner) ────────────────────────────────
    iax = inset_axes(ax, width="18%", height="24%",
                     loc="upper right",
                     bbox_to_anchor=(0.62, 0.52, 0.38, 0.48),
                     bbox_transform=ax.transAxes, borderpad=0)

    n = min(40, len(coords))
    c = coords[:n]
    adj = _make_adj(n, density=0.12, noise=0.15, seed=3)
    node_cols = [NETWORK_COLS[i % len(NETWORK_COLS)] for i in range(n)]

    plotting.plot_connectome(
        adj, c,
        node_size=8,
        node_color=node_cols,
        edge_threshold="92%",
        display_mode="l",
        colorbar=False,
        axes=iax,
        edge_kwargs={"alpha": 0.13, "linewidth": 0.13},
    )
    iax.set_title("Network\noverlap", fontsize=7, color=C_DARK, pad=2)


# ── Panel C ───────────────────────────────────────────────────────────────────

def panel_c(ax_fix: plt.Axes, ax_sw: plt.Axes) -> None:
    """Fixed vs sliding window phase-space coverage (attractor-like)."""
    t = np.linspace(0, 300, 3000)

    # Multiple incommensurate frequencies → quasi-periodic attractor (Lissajous-like)
    sig = (1.0 * np.sin(2 * np.pi * 0.013 * t + 0.0)
         + 0.7 * np.sin(2 * np.pi * 0.022 * t + 1.1)
         + 0.5 * np.sin(2 * np.pi * 0.031 * t + 2.3)
         + 0.3 * np.sin(2 * np.pi * 0.008 * t + 0.5)
         + 0.2 * np.sin(2 * np.pi * 0.047 * t + 1.7))
    sig += RNG.standard_normal(len(t)) * 0.03

    x_full, y_full = _phase_xy(sig, lag=12)

    # Fixed: single 60s window = 600 samples
    n_win = 600
    x_fix, y_fix = x_full[:n_win], y_full[:n_win]

    # Sliding: 60s windows, 15s step
    step  = 150
    xs_sw, ys_sw = [], []
    for s in range(0, len(x_full) - n_win, step):
        xs_sw.append(x_full[s:s + n_win])
        ys_sw.append(y_full[s:s + n_win])

    for ax, title, col, xpts, ypts in [
        (ax_fix, "Fixed Window\n(sparse phase coverage)",
         C_RAW, [x_fix], [y_fix]),
        (ax_sw, "Sliding Windows\n(comprehensive coverage)",
         C_BSNET, xs_sw, ys_sw),
    ]:
        # Full manifold (gray background)
        ax.scatter(x_full, y_full, s=1.5, color=C_REF,
                   alpha=0.10, zorder=1, edgecolors="none")
        for xp, yp in zip(xpts, ypts):
            ax.scatter(xp, yp, s=4, color=col,
                       alpha=0.30, zorder=3, edgecolors="none")

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("x(t)", **FONT_AXIS)
        ax.set_ylabel("x(t+τ)", **FONT_AXIS)
        ax.set_title(title, **FONT_PANEL)

        cov = sum(len(xp) for xp in xpts) / len(x_full) * 100
        ax.text(0.05, 0.05,
                f"{len(xpts)} window{'s' if len(xpts) > 1 else ''}\n"
                f"{cov:.0f}% coverage",
                transform=ax.transAxes, fontsize=8, color=col,
                fontweight="bold",
                bbox=dict(fc="white", alpha=0.82, ec=col, pad=2))

        for sp in ax.spines.values():
            sp.set_color(col); sp.set_linewidth(1.8)


# ── Panel D ───────────────────────────────────────────────────────────────────

def panel_d(ax: plt.Axes) -> None:
    """BS-NET pipeline flow with formula."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.5)
    ax.axis("off")

    steps = [
        (1.0, 2.3, "Short\nBOLD\n(2 min)",              C_RAW,   ),
        (3.0, 2.3, "Sliding\nWindow\nBootstrap",         C_BSNET, ),
        (5.0, 2.3, "Split-half\nReliability\n(ρ_xx')",  C_BSNET, ),
        (7.0, 2.3, "Attenuation\nCorrection\n+ SB",     C_GREEN, ),
        (9.0, 2.3, "Restored\nFC\n(ρ̂T)",               C_GOLD,  ),
    ]
    bw, bh = 1.45, 1.05

    for x, y, label, col in steps:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.09",
            facecolor=col, edgecolor="white", linewidth=1.5, zorder=3,
        ))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold", zorder=4)

    # Arrows
    for xa in [1.78, 3.78, 5.78, 7.78]:
        ax.annotate("", xy=(xa + 0.44, 2.3), xytext=(xa, 2.3),
                    arrowprops=dict(arrowstyle="->", color=C_DARK,
                                   lw=1.6, mutation_scale=14), zorder=5)

    # Equation box
    ax.text(5.0, 0.82,
            r"$\hat{\rho}_T = \tanh\!\left("
            r"\dfrac{\mathrm{atanh}(r_\mathrm{obs})}{"
            r"\sqrt{\rho_{xx'} \cdot r_\mathrm{SB}}}"
            r"\right)$",
            ha="center", va="center", fontsize=11,
            color=C_DARK,
            bbox=dict(fc="#f0f7ff", ec=C_BSNET, pad=6, alpha=0.95))

    ax.text(5.0, 4.15, "D.  BS-NET Pipeline",
            ha="center", **FONT_PANEL)


# ── Panel E ───────────────────────────────────────────────────────────────────

def panel_e(ax: plt.Axes) -> None:
    """Three downstream use cases (clean boxes, no inset connectomes)."""
    ax.set_xlim(0, 6); ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.text(3.0, 4.15, "E.  Restored FC → Applications",
            ha="center", **FONT_PANEL)

    cases = [
        (1.0, C_BSNET,
         "① Individual\nBiomarker",
         "FC-behavior\ncorrelation"),
        (3.0, C_GREEN,
         "② Group\nComparison",
         "Patient vs.\nControl"),
        (5.0, C_GOLD,
         "③ Multi-site\nHarmonization",
         "Standardize\nscan length"),
    ]
    bw, bh = 1.58, 1.55
    for x, col, title, sub in cases:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - bw / 2, 1.8), bw, bh,
            boxstyle="round,pad=0.10",
            facecolor=col, edgecolor="white",
            linewidth=1.3, alpha=0.90, zorder=3,
        ))
        ax.text(x, 1.8 + bh * 0.68, title,
                ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", zorder=4)
        ax.text(x, 1.8 + bh * 0.24, sub,
                ha="center", va="center",
                fontsize=8, color="white", alpha=0.92, zorder=4)

    # Input → output label
    ax.text(3.0, 1.45,
            "Short scan BOLD  →  BS-NET  →  Recovered FC",
            ha="center", fontsize=8.5, color=C_DARK, fontweight="bold",
            bbox=dict(fc="white", ec="#cccccc", pad=3, alpha=0.92))
    ax.annotate("", xy=(3.0, 1.78), xytext=(3.0, 1.56),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK,
                                lw=1.5, mutation_scale=13), zorder=5)


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_figure0() -> None:
    """Generate full conceptual framework figure."""
    apply_bsnet_theme()

    print("Loading atlas coordinates...")
    coords = _get_coords()
    print(f"  {len(coords)} ROI coordinates loaded")

    fig = plt.figure(figsize=(22, 12))
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1, 1.05],
        hspace=0.46,
        left=0.03, right=0.96, top=0.88, bottom=0.05,
    )

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    row1 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0],
        width_ratios=[1, 1, 2.6],
        wspace=0.22,
    )
    ax_long  = fig.add_subplot(row1[0, 0])
    ax_short = fig.add_subplot(row1[0, 1])
    ax_b     = fig.add_subplot(row1[0, 2])

    ax_long.text(-0.18, 1.10, "A.  Short vs. Long Scan FC",
                 transform=ax_long.transAxes,
                 fontsize=FONT_PANEL["fontsize"],
                 fontweight="bold", va="bottom")

    # ── Row 2 ─────────────────────────────────────────────────────────────────
    row2 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1],
        width_ratios=[1, 1, 2.0, 1.7],
        wspace=0.30,
    )
    ax_cfix = fig.add_subplot(row2[0, 0])
    ax_csw  = fig.add_subplot(row2[0, 1])
    ax_d    = fig.add_subplot(row2[0, 2])
    ax_e    = fig.add_subplot(row2[0, 3])

    ax_cfix.text(-0.18, 1.10, "C.  Phase-Space Coverage",
                 transform=ax_cfix.transAxes,
                 fontsize=FONT_PANEL["fontsize"],
                 fontweight="bold", va="bottom")

    # ── Draw ──────────────────────────────────────────────────────────────────
    print("Drawing Panel A (connectomes)...")
    panel_a(ax_long, ax_short, coords)

    print("Drawing Panel B (BOLD + brain inset)...")
    panel_b(ax_b, coords)

    print("Drawing Panel C (phase space)...")
    panel_c(ax_cfix, ax_csw)

    print("Drawing Panel D (pipeline)...")
    panel_d(ax_d)

    print("Drawing Panel E (use cases)...")
    panel_e(ax_e)

    # Row divider
    fig.add_artist(plt.Line2D(
        [0.02, 0.98], [0.505, 0.505],
        transform=fig.transFigure,
        color="#cccccc", lw=1.0, ls="--", zorder=0,
    ))

    fig.suptitle(
        "Figure 1 — Conceptual Framework: "
        "BS-NET recovers long-scan FC from 2-minute resting-state fMRI",
        fontsize=13, fontweight="bold", y=0.975,
    )

    print("Saving...")
    save_figure(fig, "Figure0_Conceptual_Framework.png")
    print("Saved → docs/figure/Figure0_Conceptual_Framework.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_figure0()
