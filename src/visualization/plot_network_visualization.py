"""Generate Network Visualization Figure: FC Matrix + Glass-Brain + Chord Diagram.

Three-panel (or 3×3) visualization comparing Raw FC, LW FC, and Reference FC
using ds000243 (4S256Parcels, Schaefer200 cortical subset).

Panels:
  Row 1: ROI-sorted FC matrices (Raw / BS-NET / Reference) — heatmap with RSN boundaries
  Row 2: Glass-brain connectomes (Raw / BS-NET / Reference) — top edges on MNI glass brain
  Row 3: Chord diagrams (Raw / BS-NET / Reference) — 7 Yeo RSN circular connectivity

Data: ds000243 XCP-D timeseries (4s256parcels), N=52 subjects
Atlas: Schaefer 200 (cortical only, first 200 of 256 ROIs)

Usage:
    python src/visualization/plot_network_visualization.py
    python src/visualization/plot_network_visualization.py --n-subjects 10 --short-sec 120
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

# Allow direct script execution
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.core.config import NETWORK_NAMES
from src.data.data_loader import get_fc_matrix
from src.visualization.style import (
    CONDITION_PALETTE,
    FONT,
    apply_bsnet_theme,
    save_figure,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Atlas constants ──────────────────────────────────────────────────────────

ATLAS_NII = Path("atlas/schaefer_2018/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz")
ATLAS_TXT = Path("atlas/schaefer_2018/Schaefer2018_200Parcels_7Networks_order.txt")
N_CORTICAL = 200  # Use only Schaefer200 cortical ROIs from 256-ROI 4S parcellation

# Yeo 7-network short labels (order matches Schaefer naming convention)
YEO7_SHORT = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]

# Network colors (Yeo 7-network standard)
NETWORK_COLORS: dict[str, str] = {
    "Visual": "#781286",
    "Somatomotor": "#4682B4",
    "Dorsal Attn": "#00760E",
    "Ventral Attn": "#C43AFA",
    "Limbic": "#7AB648",
    "Control": "#E69422",
    "Default Mode": "#CD3E4E",
}


# ── Helper: Parse Schaefer labels → network assignment ───────────────────────


def parse_schaefer_networks(txt_path: Path, n_rois: int = 200) -> list[int]:
    """Parse Schaefer atlas text file to extract network index per ROI.

    Args:
        txt_path: Path to Schaefer order .txt file.
        n_rois: Number of cortical ROIs.

    Returns:
        List of network indices (0–6) for each ROI.
    """
    net_indices: list[int] = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            name = parts[1]  # e.g., "7Networks_LH_Vis_1"
            for idx, short in enumerate(YEO7_SHORT):
                if f"_{short}_" in name or name.endswith(f"_{short}"):
                    net_indices.append(idx)
                    break
            if len(net_indices) >= n_rois:
                break
    return net_indices


def get_network_boundaries(net_indices: list[int]) -> list[int]:
    """Get boundary positions between networks for matrix visualization."""
    boundaries: list[int] = []
    for i in range(1, len(net_indices)):
        if net_indices[i] != net_indices[i - 1]:
            boundaries.append(i)
    return boundaries


def sort_rois_by_network(net_indices: list[int]) -> np.ndarray:
    """Return ROI sort order grouped by network."""
    return np.argsort(net_indices)


# ── Helper: Get atlas coordinates ────────────────────────────────────────────


def get_schaefer_coords(atlas_nii: Path) -> np.ndarray:
    """Extract ROI centroid coordinates from Schaefer NIfTI atlas.

    Returns:
        ndarray of shape (200, 3) — MNI coordinates.
    """
    from nilearn.plotting import find_parcellation_cut_coords

    coords = find_parcellation_cut_coords(str(atlas_nii))
    return coords[:N_CORTICAL]


# ── Helper: Compute network-level FC (7×7) ──────────────────────────────────


def compute_network_fc(
    fc_matrix: np.ndarray, net_indices: list[int], n_networks: int = 7
) -> np.ndarray:
    """Average FC within/between 7 Yeo networks.

    Args:
        fc_matrix: (n_rois, n_rois) correlation matrix.
        net_indices: Network assignment per ROI.
        n_networks: Number of networks.

    Returns:
        (n_networks, n_networks) mean FC matrix.
    """
    net_fc = np.zeros((n_networks, n_networks))
    net_arr = np.array(net_indices)
    for i in range(n_networks):
        for j in range(i, n_networks):
            mask_i = net_arr == i
            mask_j = net_arr == j
            block = fc_matrix[np.ix_(mask_i, mask_j)]
            if i == j:
                # Within-network: exclude diagonal
                vals = block[np.triu_indices_from(block, k=1)]
            else:
                vals = block.flatten()
            net_fc[i, j] = np.mean(vals) if len(vals) > 0 else 0.0
            net_fc[j, i] = net_fc[i, j]
    return net_fc


# ── Data loading and FC computation ─────────────────────────────────────────


def _process_single_subject(
    fp: str,
    short_samples: int,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process one subject: compute Raw FC, LW-shrinkage FC, and Reference FC.

    Designed for joblib parallel execution — no shared state.

    Args:
        fp: Path to .npy timeseries file.
        short_samples: Number of TRs for short scan window.
        offset: Starting TR index for the short scan window.

    Returns:
        fc_raw: Pearson correlation FC (no shrinkage) from short scan.
        fc_lw: Ledoit-Wolf shrinkage FC from short scan (BS-NET component).
        fc_ref: LW-shrinkage FC from remaining timepoints (reference).
    """
    ts = np.load(fp).astype(np.float64)[:, :N_CORTICAL]
    t_total = ts.shape[0]

    var = ts.var(axis=0)
    valid = var > 0
    if valid.sum() < N_CORTICAL:
        logger.warning(f"  {Path(fp).name}: {N_CORTICAL - valid.sum()} zero-var ROIs")

    # Short scan: ts[offset : offset + short_samples]
    end = min(offset + short_samples, t_total)
    ts_short = ts[offset:end, :]

    # Reference: everything EXCEPT the short scan window
    ts_long = np.concatenate([ts[:offset, :], ts[end:, :]], axis=0)
    if ts_long.shape[0] < short_samples:
        logger.warning(f"  {Path(fp).name}: reference too short ({ts_long.shape[0]} TRs)")

    fc_raw = get_fc_matrix(ts_short, vectorized=False, use_shrinkage=False)
    fc_lw = get_fc_matrix(ts_short, vectorized=False, use_shrinkage=True)
    fc_ref = get_fc_matrix(ts_long, vectorized=False, use_shrinkage=True)

    return fc_raw, fc_lw, fc_ref


def load_and_compute_fc(
    ts_dir: Path,
    n_subjects: int = 52,
    short_samples: int = 48,
    offset: int = 0,
    n_jobs: int = 1,
) -> tuple[dict, dict, str, dict]:
    """Load ds000243 timeseries and compute group-average FC matrices.

    Three FC conditions:
      - Raw: Pearson correlation, no regularization (noisy baseline)
      - LW:  Ledoit-Wolf shrinkage (BS-NET core component, regularized)
      - Reference: LW shrinkage on remaining timepoints (target)

    Args:
        ts_dir: Directory with sub-*_4s256parcels.npy files.
        n_subjects: Max subjects to use.
        short_samples: Number of timepoints for short scan (48 = 120s at TR=2.5).
        offset: Starting TR index for the short scan window.
        n_jobs: Number of parallel workers (default 1 = serial).

    Returns:
        (group, individual, pid, stacks) tuple.
    """
    import glob

    from joblib import Parallel, delayed

    files = sorted(glob.glob(str(ts_dir / "sub-*_4s256parcels.npy")))[:n_subjects]
    if not files:
        raise FileNotFoundError(f"No timeseries files in {ts_dir}")

    print(f"  Loading {len(files)} subjects (offset={offset} TRs, n_jobs={n_jobs})")

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_single_subject)(fp, short_samples, offset)
        for fp in files
    )

    fc_raw_stack = np.array([r[0] for r in results])
    fc_lw_stack = np.array([r[1] for r in results])
    fc_ref_stack = np.array([r[2] for r in results])

    # Select representative single subject (median raw-vs-reference similarity)
    triu = np.triu_indices(fc_raw_stack.shape[1], k=1)
    r_fc_per_sub = np.array([
        np.corrcoef(fc_raw_stack[i][triu], fc_ref_stack[i][triu])[0, 1]
        for i in range(len(results))
    ])
    median_idx = int(np.argsort(r_fc_per_sub)[len(r_fc_per_sub) // 2])

    # Extract PID from filename (e.g., "sub-01_4s256parcels.npy" → "sub-01")
    pid = Path(files[median_idx]).name.split("_")[0]
    print(f"  Representative subject: {pid} (index {median_idx}, "
          f"r_fc={r_fc_per_sub[median_idx]:.3f}, median of {len(results)})")

    print("  Computing group-average FC matrices...")
    group = {
        "raw": fc_raw_stack.mean(axis=0),
        "bsnet": fc_lw_stack.mean(axis=0),
        "reference": fc_ref_stack.mean(axis=0),
    }
    individual = {
        "raw": fc_raw_stack[median_idx],
        "bsnet": fc_lw_stack[median_idx],
        "reference": fc_ref_stack[median_idx],
    }
    stacks = {
        "raw": fc_raw_stack,
        "bsnet": fc_lw_stack,
        "reference": fc_ref_stack,
        "files": files,
    }
    return group, individual, pid, stacks


# ── Panel A: ROI-sorted FC matrices ─────────────────────────────────────────


def plot_fc_matrices(
    axes: list[plt.Axes],
    fc_dict: dict[str, np.ndarray],
    net_indices: list[int],
    sort_order: np.ndarray,
    boundaries: list[int],
    row_label: str = "",
    pid: str = "",
) -> None:
    """Plot 3 RSN-sorted FC matrices (Raw, BS-NET, Reference)."""
    parts = [p for p in [row_label, pid] if p]
    suffix = f"  [{', '.join(parts)}]" if parts else ""
    titles = [
        f"Raw FC (2 min){suffix}",
        f"LW FC (2 min){suffix}",
        f"Reference FC (full){suffix}",
    ]
    keys = ["raw", "bsnet", "reference"]
    colors = [
        CONDITION_PALETTE["raw"],
        CONDITION_PALETTE["bsnet"],
        CONDITION_PALETTE["reference"],
    ]

    vmin, vmax = -0.8, 0.8

    for ax, key, title, _color in zip(axes, keys, titles, colors):
        fc_sorted = fc_dict[key][np.ix_(sort_order, sort_order)]
        im = ax.imshow(
            fc_sorted, cmap="RdBu_r", vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="none",
        )
        # Network boundaries
        for b in boundaries:
            ax.axhline(y=b - 0.5, color="k", linewidth=1.0, alpha=0.6)
            ax.axvline(x=b - 0.5, color="k", linewidth=1.0, alpha=0.6)

        ax.set_title(title, fontsize=FONT["title"], fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Network label bar on left
        net_sorted = np.array(net_indices)[sort_order]
        for idx, name in enumerate(NETWORK_NAMES):
            mask = net_sorted == idx
            positions = np.where(mask)[0]
            if len(positions) > 0:
                mid = positions[len(positions) // 2]
                ax.text(
                    -3, mid, name[:3], fontsize=6, ha="right", va="center",
                    color=list(NETWORK_COLORS.values())[idx], fontweight="bold",
                )

    # Shared colorbar
    cbar = plt.colorbar(im, ax=axes, shrink=0.6, pad=0.02, label="FC (Pearson r)")
    cbar.ax.tick_params(labelsize=8)


# ── Panel B: Glass-brain connectomes ─────────────────────────────────────────


def _build_hemi_network_data(
    fc: np.ndarray, coords: np.ndarray, net_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build 14-node hemi-network FC matrix and centroids.

    Splits each of 7 Yeo networks into left (x<0) and right (x>0) hemisphere,
    giving 14 nodes with natural spatial separation in axial view.

    Returns:
        hemi_fc: (14, 14) mean FC between hemi-network pairs.
        hemi_coords: (14, 3) MNI centroid per hemi-network.
        hemi_colors: list of 14 hex colors (same color for L/R of a network).
    """
    n_nets = len(NETWORK_NAMES)
    net_colors = list(NETWORK_COLORS.values())
    n_hemi = n_nets * 2  # 14

    # Assign each ROI to a hemi-network index (0..13)
    # Even indices = left, odd = right for each network
    hemi_labels = np.zeros(len(net_indices), dtype=int)
    for roi_idx in range(len(net_indices)):
        ni = net_indices[roi_idx]
        is_right = 1 if coords[roi_idx, 0] > 0 else 0
        hemi_labels[roi_idx] = ni * 2 + is_right

    # Compute centroids
    hemi_coords = np.zeros((n_hemi, 3))
    for hi in range(n_hemi):
        mask = hemi_labels == hi
        if mask.sum() > 0:
            hemi_coords[hi] = coords[mask].mean(axis=0)

    # Compute 14×14 hemi-network FC
    hemi_fc = np.zeros((n_hemi, n_hemi))
    for i in range(n_hemi):
        for j in range(i, n_hemi):
            mask_i = hemi_labels == i
            mask_j = hemi_labels == j
            if mask_i.sum() == 0 or mask_j.sum() == 0:
                continue
            block = fc[np.ix_(mask_i, mask_j)]
            if i == j:
                tri = np.triu_indices_from(block, k=1)
                vals = block[tri]
            else:
                vals = block.flatten()
            if len(vals) > 0:
                hemi_fc[i, j] = hemi_fc[j, i] = vals.mean()

    # Colors: same network color for L and R
    hemi_colors = []
    for ni in range(n_nets):
        hemi_colors.extend([net_colors[ni], net_colors[ni]])

    return hemi_fc, hemi_coords, hemi_colors


def plot_glass_brains(
    axes: list[plt.Axes],
    fc_dict: dict[str, np.ndarray],
    coords: np.ndarray,
    net_indices: list[int],
    top_pct: float = 5.0,
    pid: str = "",
) -> None:
    """Plot 3 glass-brain connectomes at hemi-network level (14 nodes).

    Splits 7 Yeo networks into left/right hemisphere (14 nodes total) to
    provide natural spatial separation in axial view. Edges represent mean
    FC between hemi-network pairs; within-hemi-network edges excluded.
    """
    from nilearn import plotting

    pid_tag = f"  [{pid}]" if pid else ""
    titles = [
        f"Raw FC (2 min){pid_tag}",
        f"LW FC (2 min){pid_tag}",
        f"Reference FC (full){pid_tag}",
    ]
    keys = ["raw", "bsnet", "reference"]

    for ax, key, title in zip(axes, keys, titles):
        fc = fc_dict[key].copy()

        hemi_fc, hemi_coords, hemi_colors = _build_hemi_network_data(
            fc, coords, net_indices,
        )
        np.fill_diagonal(hemi_fc, 0)

        # Threshold: keep top_pct% strongest edges
        n_hemi = hemi_fc.shape[0]
        triu = np.triu_indices(n_hemi, k=1)
        vals = np.abs(hemi_fc[triu])
        if vals.max() > 0:
            threshold = np.percentile(vals, 100 - top_pct * 3)  # ~15% for 14 nodes
            hemi_fc[np.abs(hemi_fc) < threshold] = 0

        # Dynamic vmin/vmax from surviving edges
        surviving_nz = hemi_fc[triu][hemi_fc[triu] != 0]
        vmax = np.max(np.abs(surviving_nz)) if len(surviving_nz) > 0 else 0.5

        plotting.plot_connectome(
            hemi_fc, hemi_coords,
            node_color=hemi_colors,
            node_size=80,
            edge_cmap="RdBu_r",
            edge_vmin=-vmax, edge_vmax=vmax,
            edge_kwargs={"linewidth": 2.0},
            display_mode="z",
            axes=ax,
            annotate=False,
            colorbar=False,
        )
        ax.set_title(title, fontsize=FONT["title"], fontweight="bold")


# ── Panel C: Chord diagrams ─────────────────────────────────────────────────


def plot_chord_diagrams(
    axes: list[plt.Axes],
    fc_dict: dict[str, np.ndarray],
    net_indices: list[int],
) -> None:
    """Plot 3 chord diagrams showing 7-network FC strength."""
    titles = ["Raw FC (2 min)", "LW FC (2 min)", "Reference FC (full)"]
    keys = ["raw", "bsnet", "reference"]
    n_nets = len(NETWORK_NAMES)

    for ax, key, title in zip(axes, keys, titles):
        net_fc = compute_network_fc(fc_dict[key], net_indices, n_nets)

        _draw_chord(ax, net_fc, NETWORK_NAMES, list(NETWORK_COLORS.values()))
        ax.set_title(title, fontsize=FONT["title"], fontweight="bold")


def _draw_chord(
    ax: plt.Axes,
    net_fc: np.ndarray,
    names: list[str],
    colors: list[str],
    min_width: float = 0.5,
    max_width: float = 8.0,
) -> None:
    """Draw a chord diagram on a polar-like axes.

    Args:
        ax: Matplotlib Axes.
        net_fc: (n, n) network-level FC matrix.
        names: Network labels.
        colors: Network colors.
        min_width: Min edge width.
        max_width: Max edge width.
    """
    n = len(names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Draw nodes
    radius = 1.0
    node_x = radius * np.cos(angles)
    node_y = radius * np.sin(angles)

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges (between-network)
    triu = np.triu_indices(n, k=1)
    fc_vals = net_fc[triu]
    fc_abs = np.abs(fc_vals)
    if fc_abs.max() > 0:
        widths = min_width + (fc_abs - fc_abs.min()) / (fc_abs.max() - fc_abs.min()) * (max_width - min_width)
    else:
        widths = np.full_like(fc_abs, min_width)

    for idx, (i, j) in enumerate(zip(triu[0], triu[1])):
        val = fc_vals[idx]
        w = widths[idx]
        color = "#d7191c" if val > 0 else "#2c7bb6"
        alpha = 0.15 + 0.55 * (fc_abs[idx] / fc_abs.max() if fc_abs.max() > 0 else 0)

        # Curved edge via quadratic Bezier
        mid_x = 0.3 * (node_x[i] + node_x[j])
        mid_y = 0.3 * (node_y[i] + node_y[j])

        from matplotlib.path import Path as MplPath

        verts = [
            (node_x[i], node_y[i]),
            (mid_x, mid_y),
            (node_x[j], node_y[j]),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
        path = MplPath(verts, codes)
        patch = mpatches.PathPatch(
            path, facecolor="none", edgecolor=color,
            linewidth=w, alpha=alpha,
        )
        ax.add_patch(patch)

    # Draw within-network arcs (self-loops as arcs)
    for i in range(n):
        val = net_fc[i, i]
        arc_radius = 0.12
        arc_x = node_x[i] * 1.15
        arc_y = node_y[i] * 1.15
        arc = mpatches.Circle(
            (arc_x, arc_y), arc_radius,
            fill=True, facecolor=colors[i], edgecolor="white",
            alpha=0.6, linewidth=0.5,
        )
        ax.add_patch(arc)

    # Draw node circles and labels
    for i in range(n):
        circle = plt.Circle(
            (node_x[i], node_y[i]), 0.08,
            color=colors[i], ec="white", lw=1.5, zorder=5,
        )
        ax.add_patch(circle)

        # Label
        label_r = 1.35
        lx = label_r * np.cos(angles[i])
        ly = label_r * np.sin(angles[i])
        ha = "left" if np.cos(angles[i]) > 0.1 else ("right" if np.cos(angles[i]) < -0.1 else "center")
        ax.text(
            lx, ly, names[i], fontsize=7, ha=ha, va="center",
            fontweight="bold", color=colors[i],
        )


# ── Main figure assembly ─────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Network Visualization (ds000243)")
    parser.add_argument("--n-subjects", type=int, default=52)
    parser.add_argument("--short-sec", type=float, default=120.0)
    parser.add_argument("--top-pct", type=float, default=5.0, help="Top %% edges for glass-brain")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for subject loop")
    parser.add_argument(
        "--offsets", type=str, default="0,60,120,180",
        help="Comma-separated offset seconds for multi-offset robustness test",
    )
    args = parser.parse_args()

    print("=== Network Visualization (ds000243, 4S256) ===")

    ts_dir = Path("data/ds000243/timeseries_cache_xcpd/4s256parcels")
    tr = 2.5
    short_samples = int(args.short_sec / tr)
    offset_secs = [float(x) for x in args.offsets.split(",")]
    offset_trs = [int(s / tr) for s in offset_secs]

    # 1. Parse atlas
    print("Parsing Schaefer atlas...")
    net_indices = parse_schaefer_networks(ATLAS_TXT, N_CORTICAL)
    sort_order = sort_rois_by_network(net_indices)
    boundaries = get_network_boundaries(np.array(net_indices)[sort_order].tolist())

    # 2. Multi-offset FC computation
    all_offset_stacks = {}
    fc_group = fc_indiv = rep_pid = None

    for oi, (off_sec, off_tr) in enumerate(zip(offset_secs, offset_trs)):
        print(f"\n--- Offset {off_sec:.0f}s (TR {off_tr}) ---")
        grp, ind, pid, stk = load_and_compute_fc(
            ts_dir, n_subjects=args.n_subjects,
            short_samples=short_samples, offset=off_tr,
            n_jobs=args.n_jobs,
        )
        all_offset_stacks[off_sec] = stk

        # Use first offset for the main figure
        if oi == 0:
            fc_group, fc_indiv, rep_pid = grp, ind, pid

    # 3. Get atlas coordinates
    print("\nExtracting atlas coordinates...")
    coords = get_schaefer_coords(ATLAS_NII)

    # 4. Plot — 4-row layout (based on offset=0)
    apply_bsnet_theme()
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.28, wspace=0.15,
        height_ratios=[1, 1, 0.85, 1],
    )

    # Row 1 (A–C): Individual FC matrices
    ax_ind_mat = [fig.add_subplot(gs[0, i]) for i in range(3)]
    plot_fc_matrices(
        ax_ind_mat, fc_indiv, net_indices, sort_order, boundaries,
        row_label="Individual", pid=rep_pid,
    )

    # Row 2 (D–F): Group-average FC matrices
    ax_grp_mat = [fig.add_subplot(gs[1, i]) for i in range(3)]
    plot_fc_matrices(
        ax_grp_mat, fc_group, net_indices, sort_order, boundaries,
        row_label=f"Group (N={args.n_subjects})",
    )

    # Row 3 (G–I): Individual glass-brain connectomes
    ax_ind_brain = [fig.add_subplot(gs[2, i]) for i in range(3)]
    plot_glass_brains(
        ax_ind_brain, fc_indiv, coords, net_indices,
        top_pct=args.top_pct, pid=rep_pid,
    )

    # Row 4 (J–L): Group-average chord diagrams
    ax_grp_chord = [fig.add_subplot(gs[3, i]) for i in range(3)]
    plot_chord_diagrams(ax_grp_chord, fc_group, net_indices)

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, label=n)
        for n, c in NETWORK_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=7, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    save_figure(fig, "Figure_NetworkVisualization.png")
    print("\nFigure saved: Figure_NetworkVisualization.png")

    # 5. Reviewer statistics table (per-offset + summary)
    generate_reviewer_table(
        all_offset_stacks, net_indices, out_dir=Path("artifacts/reports"),
    )


def generate_reviewer_table(
    all_offset_stacks: dict[float, dict],
    net_indices: list[int],
    out_dir: Path = Path("artifacts/reports"),
) -> None:
    """Generate reviewer-ready CSV tables with multi-offset FC quality metrics.

    Produces three CSV files:
      1. Per-subject per-offset: PID, offset, r_fc(raw→ref), r_fc(lw→ref), Δr
      2. Per-offset summary: offset, mean r_fc, Δr, % improved
      3. Per-network-pair (offset=0 only): network pair, mean FC, Δ

    These tables demonstrate robustness of LW-shrinkage improvement across
    different temporal segments of the scan.
    """
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    n_nets = len(NETWORK_NAMES)

    # ── Table 1: Per-subject × per-offset ───────────────────────────────────
    all_rows = []
    offset_summaries = []

    for off_sec in sorted(all_offset_stacks.keys()):
        stk = all_offset_stacks[off_sec]
        fc_raw = stk["raw"]
        fc_lw = stk["bsnet"]
        fc_ref = stk["reference"]
        files = stk["files"]
        n_sub = fc_raw.shape[0]
        triu = np.triu_indices(fc_raw.shape[1], k=1)

        r_raws, r_lws, deltas = [], [], []
        for i in range(n_sub):
            pid = Path(files[i]).name.split("_")[0]
            r_raw = np.corrcoef(fc_raw[i][triu], fc_ref[i][triu])[0, 1]
            r_lw = np.corrcoef(fc_lw[i][triu], fc_ref[i][triu])[0, 1]
            delta = r_lw - r_raw
            r_raws.append(r_raw)
            r_lws.append(r_lw)
            deltas.append(delta)
            all_rows.append({
                "offset_sec": int(off_sec),
                "subject": pid,
                "r_fc_raw_vs_ref": round(r_raw, 4),
                "r_fc_lw_vs_ref": round(r_lw, 4),
                "delta_r": round(delta, 4),
                "improved": delta > 0,
            })

        r_raws = np.array(r_raws)
        r_lws = np.array(r_lws)
        deltas = np.array(deltas)
        n_improved = int((deltas > 0).sum())

        offset_summaries.append({
            "offset_sec": int(off_sec),
            "n_subjects": n_sub,
            "mean_r_fc_raw": round(float(r_raws.mean()), 4),
            "std_r_fc_raw": round(float(r_raws.std()), 4),
            "mean_r_fc_lw": round(float(r_lws.mean()), 4),
            "std_r_fc_lw": round(float(r_lws.std()), 4),
            "mean_delta_r": round(float(deltas.mean()), 4),
            "std_delta_r": round(float(deltas.std()), 4),
            "n_improved": n_improved,
            "pct_improved": round(100 * n_improved / n_sub, 1),
        })

    # Save per-subject table
    csv1 = out_dir / "NetworkViz_per_subject_multioffset.csv"
    with open(csv1, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Table 2: Per-offset summary ─────────────────────────────────────────
    csv2 = out_dir / "NetworkViz_offset_summary.csv"
    with open(csv2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=offset_summaries[0].keys())
        writer.writeheader()
        writer.writerows(offset_summaries)

    print(f"\n{'='*60}")
    print("=== Multi-Offset Robustness Summary ===")
    print(f"{'='*60}")
    print(f"{'Offset':>8s} | {'r_fc(Raw)':>12s} | {'r_fc(LW)':>12s} | "
          f"{'Δr':>12s} | {'Improved':>10s}")
    print(f"{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for s in offset_summaries:
        print(f"{s['offset_sec']:>6d}s | "
              f"{s['mean_r_fc_raw']:.4f}±{s['std_r_fc_raw']:.4f} | "
              f"{s['mean_r_fc_lw']:.4f}±{s['std_r_fc_lw']:.4f} | "
              f"{s['mean_delta_r']:+.4f}±{s['std_delta_r']:.4f} | "
              f"{s['n_improved']}/{s['n_subjects']} ({s['pct_improved']:.0f}%)")

    # Grand mean across offsets
    all_deltas = np.array([s["mean_delta_r"] for s in offset_summaries])
    all_pcts = np.array([s["pct_improved"] for s in offset_summaries])
    print(f"{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    print(f"{'Mean':>8s} | {'':>12s} | {'':>12s} | "
          f"{all_deltas.mean():+.4f}±{all_deltas.std():.4f} | "
          f"{all_pcts.mean():.0f}%")

    print(f"\nSaved: {csv1}")
    print(f"Saved: {csv2}")

    # ── Table 3: Per-network-pair (first offset only) ───────────────────────
    first_off = sorted(all_offset_stacks.keys())[0]
    stk0 = all_offset_stacks[first_off]
    fc_raw_grp = stk0["raw"].mean(axis=0)
    fc_lw_grp = stk0["bsnet"].mean(axis=0)
    fc_ref_grp = stk0["reference"].mean(axis=0)
    net_arr = np.array(net_indices)

    net_rows = []
    for ni in range(n_nets):
        for nj in range(ni, n_nets):
            mask_i = net_arr == ni
            mask_j = net_arr == nj
            if ni == nj:
                block_idx = np.triu_indices(mask_i.sum(), k=1)
                raw_vals = fc_raw_grp[np.ix_(mask_i, mask_j)][block_idx]
                lw_vals = fc_lw_grp[np.ix_(mask_i, mask_j)][block_idx]
                ref_vals = fc_ref_grp[np.ix_(mask_i, mask_j)][block_idx]
                pair_type = "within"
            else:
                raw_vals = fc_raw_grp[np.ix_(mask_i, mask_j)].flatten()
                lw_vals = fc_lw_grp[np.ix_(mask_i, mask_j)].flatten()
                ref_vals = fc_ref_grp[np.ix_(mask_i, mask_j)].flatten()
                pair_type = "between"

            if len(raw_vals) == 0:
                continue

            net_rows.append({
                "network_i": NETWORK_NAMES[ni],
                "network_j": NETWORK_NAMES[nj],
                "pair_type": pair_type,
                "mean_fc_raw": round(float(raw_vals.mean()), 4),
                "mean_fc_lw": round(float(lw_vals.mean()), 4),
                "mean_fc_reference": round(float(ref_vals.mean()), 4),
                "delta_raw_vs_ref": round(float(raw_vals.mean() - ref_vals.mean()), 4),
                "delta_lw_vs_ref": round(float(lw_vals.mean() - ref_vals.mean()), 4),
            })

    csv3 = out_dir / "NetworkViz_per_network_pair.csv"
    with open(csv3, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=net_rows[0].keys())
        writer.writeheader()
        writer.writerows(net_rows)
    print(f"Saved: {csv3}")


if __name__ == "__main__":
    main()
