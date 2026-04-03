"""FC matrix visualization: full vs. |FC|>=0.3 thresholded.

For each subject row:
  Left  : Full reference FC matrix (all pairs)
  Center: Thresholded FC matrix (|FC| < 0.3 → 0)
  Right : Difference mask (connections removed by threshold)

Usage:
    python src/scripts/visualize_fc_threshold.py \
        --atlas 4s256parcels \
        --n-subjects 4 \
        --threshold 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import apply_bsnet_theme, save_figure

CACHE_DIR = Path("data/ds000243/timeseries_cache_xcpd")
TR        = 2.5
MIN_SEC   = 600.0


def _load_subjects(atlas: str, n: int) -> list[dict]:
    files = sorted((CACHE_DIR / atlas).glob("*.npy"))[:n * 3]  # load extra, filter
    subjects = []
    for fp in files:
        ts = np.load(fp).astype(np.float64)
        if ts.shape[0] * TR < MIN_SEC:
            continue
        subjects.append({"sub_id": fp.stem.split("_")[0], "ts": ts})
        if len(subjects) == n:
            break
    return subjects


def _fc_matrix(ts: np.ndarray) -> np.ndarray:
    fc = np.corrcoef(ts.T)
    return np.nan_to_num(fc, nan=0.0)


def plot_fc_threshold(
    atlas: str,
    n_subjects: int,
    threshold: float,
) -> None:
    apply_bsnet_theme()
    subjects = _load_subjects(atlas, n_subjects)
    n = len(subjects)

    if n == 0:
        print("No subjects found.")
        return

    print(f"Computing group-average FC from {n} subjects...")

    # ── Group-average FC matrix ───────────────────────────────────────────────
    fc_stack = []
    for sub in subjects:
        fc = _fc_matrix(sub["ts"])
        np.fill_diagonal(fc, 0.0)
        fc_stack.append(fc)

    fc_mean = np.mean(fc_stack, axis=0)   # (n_rois, n_rois)
    np.fill_diagonal(fc_mean, np.nan)     # diagonal → NaN for display

    n_rois  = fc_mean.shape[0]
    n_pairs = n_rois * (n_rois - 1) // 2
    i_u, j_u = np.triu_indices(n_rois, k=1)
    fc_upper = fc_mean[i_u, j_u]

    # Thresholded
    fc_thresh = fc_mean.copy()
    mask_remove = np.abs(fc_thresh) < threshold
    fc_thresh[mask_remove] = 0.0

    # Removed
    fc_removed = fc_mean.copy()
    fc_removed[~mask_remove] = 0.0
    np.fill_diagonal(fc_removed, np.nan)

    n_kept    = int((np.abs(fc_upper) >= threshold).sum())
    n_removed = n_pairs - n_kept
    pct_kept  = n_kept / n_pairs * 100

    vabs = float(np.nanpercentile(np.abs(fc_upper), 99))
    vabs = max(vabs, 0.05)

    # ── Figure: 1 row × 3 cols ────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06,
                            left=0.04, right=0.95, top=0.88, bottom=0.04)

    col_titles = [
        f"Group-average FC (N={n})",
        f"Thresholded  |FC| ≥ {threshold}  ({pct_kept:.1f}% pairs kept)",
        f"Removed  |FC| < {threshold}  ({100-pct_kept:.1f}% pairs)",
    ]

    for col, (matrix, ctitle) in enumerate(
        zip([fc_mean, fc_thresh, fc_removed], col_titles)
    ):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-vabs, vmax=vabs,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_title(ctitle, fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("Mean Pearson r", fontsize=8.5)

        # Stats box
        if col == 1:
            ax.text(
                0.02, 0.02,
                f"{n_kept:,} / {n_pairs:,} pairs\n"
                f"mean |FC| = {np.mean(np.abs(fc_upper[np.abs(fc_upper)>=threshold])):.3f}",
                transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8.5, color="white",
                bbox=dict(fc="#333333", alpha=0.70, ec="none", pad=3),
            )
        elif col == 2:
            ax.text(
                0.02, 0.02,
                f"{n_removed:,} / {n_pairs:,} pairs\n"
                f"mean |FC| = {np.mean(np.abs(fc_upper[np.abs(fc_upper)<threshold])):.3f}",
                transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8.5, color="white",
                bbox=dict(fc="#333333", alpha=0.70, ec="none", pad=3),
            )

    fig.suptitle(
        f"Group-average FC Matrix — ds000243 XCP-D ({atlas}, N={n})\n"
        f"{n_rois} ROIs  ·  {n_pairs:,} unique pairs  ·  threshold |FC| ≥ {threshold}",
        fontsize=11.5, fontweight="bold",
    )

    out_name = f"fc_matrix_avg_threshold_{atlas}_t{str(threshold).replace('.','')}.png"
    out = save_figure(fig, out_name)
    print(f"Saved: {out}")
    plt.close(fig)

    # Console summary
    print(f"\n{'='*50}")
    print(f"Group average FC (N={n})  |  threshold |FC| >= {threshold}")
    print(f"  Total pairs : {n_pairs:,}")
    print(f"  Kept        : {n_kept:,}  ({pct_kept:.1f}%)")
    print(f"  Removed     : {n_removed:,}  ({100-pct_kept:.1f}%)")
    print(f"  mean|FC| kept   : {np.mean(np.abs(fc_upper[np.abs(fc_upper)>=threshold])):.3f}")
    print(f"  mean|FC| removed: {np.mean(np.abs(fc_upper[np.abs(fc_upper)<threshold])):.3f}")
    print("="*50)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Visualize FC matrices before/after |FC|>=threshold"
    )
    parser.add_argument("--atlas",      default="4s256parcels")
    parser.add_argument("--n-subjects", type=int,   default=4)
    parser.add_argument("--threshold",  type=float, default=0.3)
    args = parser.parse_args()

    plot_fc_threshold(args.atlas, args.n_subjects, args.threshold)
