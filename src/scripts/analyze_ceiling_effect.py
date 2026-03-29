#!/usr/bin/env python3
"""
Ceiling effect analysis for BS-NET attenuation correction.

Compares multiple correction methods on ABIDE PCP data to address the
overcorrection / ceiling effect where 53.6% of subjects hit ρ̂T=0.9999.

Methods compared:
    1. original  — Standard CTT disattenuation + hard clip (Spearman 1904)
    2. fisher_z  — Fisher z-space correction (Shou 2014, Teeuw 2021)
    3. partial   — Partial correction with alpha=0.5 (Zimmerman 2007)
    4. soft_clamp — Soft sigmoid clamping via tanh

Usage:
    # Run on ABIDE cached time series (requires prior run of run_abide_bsnet.py)
    python src/scripts/analyze_ceiling_effect.py

    # Specify atlas and max subjects
    python src/scripts/analyze_ceiling_effect.py --atlas cc200 --max-subjects 50

    # Parallel processing
    python src/scripts/analyze_ceiling_effect.py --n-jobs -1

    # Verbose
    python src/scripts/analyze_ceiling_effect.py -v
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ---- Project imports ----
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.bootstrap import CORRECTION_METHODS
from src.core.config import BSNetConfig
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)


# ============================================================================
# Worker function for parallel processing
# ============================================================================
def _process_subject_all_methods(args: tuple) -> dict | None:
    """Process one subject with all correction methods efficiently.

    Runs bootstrap resampling ONCE and applies all correction methods to the
    same bootstrap samples, avoiding 4x redundant computation.

    Args:
        args: (index, sub_id, ts_path, tr, short_sec, methods_list, n_bootstraps).

    Returns:
        Dict with r_fc_raw and rho_hat_T per method, or None on failure.
    """
    from src.core.bootstrap import (
        block_bootstrap_indices,
        correct_attenuation,
        estimate_optimal_block_length,
        fisher_z,
        fisher_z_inv,
    )
    from src.core.pipeline import compute_split_half_reliability

    i, sub_id, ts_path, tr, short_sec, methods, n_bootstraps = args

    # Load time series
    path = Path(ts_path)
    if not path.exists():
        return None

    ts = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)

    if ts.ndim == 1 or ts.shape[0] < 20:
        return None

    # Remove zero-variance ROIs
    valid = np.std(ts, axis=0) > 1e-8
    ts = ts[:, valid].astype(np.float64)

    n_vols, n_rois = ts.shape
    short_vols = int(short_sec / tr)

    if n_vols < short_vols + 10:
        return None

    # Reference FC (full scan)
    fc_full_vec = get_fc_matrix(ts, vectorized=True, use_shrinkage=True)

    # Short scan
    ts_short = ts[:short_vols, :]
    fc_short_vec = get_fc_matrix(ts_short, vectorized=True, use_shrinkage=True)

    # Baseline raw correlation
    r_fc_raw = float(np.corrcoef(fc_short_vec, fc_full_vec)[0, 1])

    # Config
    config = BSNetConfig(
        n_rois=n_rois, tr=tr,
        short_duration_sec=short_sec,
        target_duration_min=15,
        n_bootstraps=n_bootstraps, seed=42,
    )

    result = {
        "sub_id": sub_id,
        "n_vols": n_vols,
        "n_rois": n_rois,
        "tr": tr,
        "r_fc_raw": round(r_fc_raw, 6),
    }

    # ---- Run bootstrap ONCE, apply all methods to same samples ----
    np.random.seed(config.seed)
    t_samples = ts_short.shape[0]
    block_size = estimate_optimal_block_length(ts_short)
    empirical_prior = config.empirical_prior
    k = config.k_factor
    rel_coeff = config.reliability_coeff

    # Per-method z-score accumulators
    z_scores = {m: [] for m in methods}

    for _b in range(config.n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = ts_short[idx, :]

        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs_t = np.corrcoef(fc_full_vec, fc_obs_t)[0, 1]
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=True)

        # Apply each correction method to same bootstrap sample
        for method in methods:
            rho_est = correct_attenuation(
                r_obs_t, rel_coeff, r_split_t, k=k,
                empirical_prior=empirical_prior, method=method,
            )
            z_scores[method].append(fisher_z(rho_est))

    # Compute point estimates and CIs
    for method in methods:
        z_arr = np.array(z_scores[method])
        rho_hat_T = fisher_z_inv(np.nanmedian(z_arr))
        z_lo, z_hi = np.percentile(z_arr, [2.5, 97.5])

        result[f"rho_{method}"] = round(float(rho_hat_T), 6)
        result[f"ci_lo_{method}"] = round(float(fisher_z_inv(z_lo)), 6)
        result[f"ci_hi_{method}"] = round(float(fisher_z_inv(z_hi)), 6)

    return result


def _resolve_n_jobs(n_jobs: int) -> int:
    """Resolve n_jobs to actual worker count."""
    cpu_count = os.cpu_count() or 1
    if n_jobs == -1:
        return cpu_count
    if n_jobs <= 0:
        return max(1, cpu_count + n_jobs)
    return min(n_jobs, cpu_count)


# ============================================================================
# Load subject metadata from existing results CSV
# ============================================================================
def load_subjects_from_csv(csv_path: str, cache_dir: Path) -> list[dict]:
    """Load subject info from existing ABIDE results CSV.

    Args:
        csv_path: Path to existing results CSV.
        cache_dir: Directory containing cached .npy time series.

    Returns:
        List of subject dicts with keys: sub_id, ts_path, tr.
    """
    subjects = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub_id = row["sub_id"]
            atlas = row["atlas"]
            ts_path = cache_dir / f"{sub_id}_{atlas}.npy"
            if ts_path.exists():
                subjects.append({
                    "sub_id": sub_id,
                    "ts_path": str(ts_path),
                    "tr": float(row["tr"]),
                    "site": row.get("site", "unknown"),
                })
    return subjects


# ============================================================================
# Main analysis
# ============================================================================
def run_ceiling_analysis(
    csv_path: str,
    atlas: str = "cc200",
    max_subjects: int = 0,
    n_bootstraps: int = 100,
    n_jobs: int = 1,
    verbose: bool = False,
) -> Path:
    """Run ceiling effect analysis comparing correction methods.

    Args:
        csv_path: Path to existing ABIDE results CSV.
        atlas: Atlas name (for cache dir resolution).
        max_subjects: 0 = all.
        n_jobs: Parallel workers.
        verbose: Verbose logging.

    Returns:
        Path to output CSV.
    """
    base_dir = Path(csv_path).parents[1]
    cache_dir = base_dir / "timeseries_cache" / atlas
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load subjects
    subjects = load_subjects_from_csv(csv_path, cache_dir)
    if max_subjects > 0:
        subjects = subjects[:max_subjects]

    logger.info(f"Ceiling effect analysis: {len(subjects)} subjects, "
                f"atlas={atlas}, methods={list(CORRECTION_METHODS)}")

    methods = list(CORRECTION_METHODS)
    short_sec = 120

    # Build worker args
    worker_args = [
        (i, sub["sub_id"], sub["ts_path"], sub["tr"], short_sec, methods,
         n_bootstraps)
        for i, sub in enumerate(subjects)
    ]

    n_workers = _resolve_n_jobs(n_jobs)
    results = []
    t_start = time.time()

    if n_workers > 1:
        logger.info(f"Parallel mode: {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_subject_all_methods, arg): arg[0]
                for arg in worker_args
            }
            for done_count, future in enumerate(as_completed(futures), 1):
                res = future.result()
                if res is not None:
                    results.append(res)
                if done_count % 50 == 0 or done_count == 1:
                    logger.info(f"  [{done_count}/{len(subjects)}] processed")
    else:
        logger.info("Sequential mode")
        for i, arg in enumerate(worker_args):
            res = _process_subject_all_methods(arg)
            if res is not None:
                results.append(res)
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  [{i+1}/{len(subjects)}] processed")

    elapsed = time.time() - t_start
    logger.info(f"Completed: {len(results)}/{len(subjects)} subjects in {elapsed:.0f}s")

    # Save CSV
    csv_out = results_dir / f"ceiling_analysis_{atlas}.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results CSV: {csv_out}")

    # Summary statistics
    if results:
        r_fc_raw = np.array([r["r_fc_raw"] for r in results])

        logger.info("")
        logger.info("=" * 80)
        logger.info(" Ceiling Effect Analysis — Method Comparison")
        logger.info("=" * 80)
        logger.info(f"  N = {len(results)}, Atlas = {atlas}")
        logger.info(f"  r_FC raw: {np.mean(r_fc_raw):.4f} ± {np.std(r_fc_raw):.4f}")
        logger.info("")
        logger.info(
            f"  {'Method':<14} {'mean':>8} {'std':>8} {'median':>8} "
            f"{'ceiling%':>9} {'>0.99%':>8} {'improve':>8}"
        )
        logger.info("  " + "-" * 68)

        summary = {"n_subjects": len(results), "atlas": atlas}

        for method in methods:
            col = f"rho_{method}"
            vals = np.array([r[col] for r in results if not np.isnan(r[col])])
            if len(vals) == 0:
                continue

            ceiling_pct = float(np.mean(vals > 0.999)) * 100
            high_pct = float(np.mean(vals > 0.99)) * 100
            improvement = float(np.mean(vals - r_fc_raw[:len(vals)]))

            logger.info(
                f"  {method:<14} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
                f"{np.median(vals):>8.4f} {ceiling_pct:>8.1f}% {high_pct:>7.1f}% "
                f"{improvement:>+8.4f}"
            )

            summary[method] = {
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals)), 6),
                "median": round(float(np.median(vals)), 6),
                "min": round(float(np.min(vals)), 6),
                "max": round(float(np.max(vals)), 6),
                "ceiling_pct": round(ceiling_pct, 2),
                "gt_099_pct": round(high_pct, 2),
                "improvement": round(improvement, 6),
            }

        logger.info("=" * 80)

        # Save summary JSON
        summary_path = results_dir / f"ceiling_summary_{atlas}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary JSON: {summary_path}")

    # Generate comparison figure
    if results:
        _plot_comparison(results, methods, r_fc_raw, atlas, results_dir)

    return csv_out


# ============================================================================
# Visualization
# ============================================================================
def _plot_comparison(results, methods, r_fc_raw, atlas, output_dir):
    """Generate 4-panel comparison figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping figure")
        return

    method_labels = {
        "original": "Original\n(hard clip)",
        "fisher_z": "Fisher z-space\n(Shou 2014)",
        "partial": "Partial α=0.5\n(Zimmerman 2007)",
        "soft_clamp": "Soft clamp\n(tanh)",
    }
    colors = {
        "original": "#d62728",
        "fisher_z": "#1f77b4",
        "partial": "#2ca02c",
        "soft_clamp": "#ff7f0e",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Distribution of ρ̂T per method
    ax = axes[0, 0]
    for method in methods:
        vals = np.array([r[f"rho_{method}"] for r in results
                         if not np.isnan(r[f"rho_{method}"])])
        ax.hist(vals, bins=50, alpha=0.5, label=method_labels.get(method, method),
                color=colors.get(method), density=True)
    ax.axvline(np.mean(r_fc_raw), color="gray", ls="--", lw=1.5, label="r_FC raw mean")
    ax.set_xlabel("ρ̂T")
    ax.set_ylabel("Density")
    ax.set_title("A. Distribution of ρ̂T by correction method")
    ax.legend(fontsize=8, loc="upper left")

    # Panel B: Scatter — original vs fisher_z
    ax = axes[0, 1]
    orig = np.array([r["rho_original"] for r in results])
    fisher = np.array([r["rho_fisher_z"] for r in results])
    ax.scatter(orig, fisher, alpha=0.4, s=15, c="#1f77b4", edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("ρ̂T (original, hard clip)")
    ax.set_ylabel("ρ̂T (Fisher z-space)")
    ax.set_title("B. Original vs Fisher z correction")
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.4, 1.02)
    # Annotate ceiling subjects
    n_ceil_orig = np.sum(orig > 0.999)
    n_ceil_fz = np.sum(fisher > 0.999)
    ax.text(0.05, 0.95, f"Ceiling (>0.999):\n  Original: {n_ceil_orig}\n  Fisher z: {n_ceil_fz}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Panel C: Ceiling % bar chart
    ax = axes[1, 0]
    ceiling_pcts = []
    labels = []
    bar_colors = []
    for method in methods:
        vals = np.array([r[f"rho_{method}"] for r in results
                         if not np.isnan(r[f"rho_{method}"])])
        ceiling_pcts.append(float(np.mean(vals > 0.999)) * 100)
        labels.append(method_labels.get(method, method))
        bar_colors.append(colors.get(method, "#333"))

    bars = ax.bar(range(len(methods)), ceiling_pcts, color=bar_colors, alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% subjects at ceiling (ρ̂T > 0.999)")
    ax.set_title("C. Ceiling effect by correction method")
    for bar, pct in zip(bars, ceiling_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel D: Improvement (ρ̂T - r_fc_raw) boxplot
    ax = axes[1, 1]
    improvement_data = []
    bp_labels = []
    for method in methods:
        vals = np.array([r[f"rho_{method}"] for r in results
                         if not np.isnan(r[f"rho_{method}"])])
        improvement_data.append(vals - r_fc_raw[:len(vals)])
        bp_labels.append(method_labels.get(method, method))

    bp = ax.boxplot(improvement_data, tick_labels=bp_labels, patch_artist=True,
                    showfliers=False, widths=0.6)
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(colors.get(method, "#ccc"))
        patch.set_alpha(0.6)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_ylabel("Improvement (ρ̂T − r_FC)")
    ax.set_title("D. Improvement distribution by method")
    ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        f"BS-NET Ceiling Effect Analysis — ABIDE {atlas.upper()} (N={len(results)})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = output_dir / f"ceiling_analysis_{atlas}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {fig_path}")


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="BS-NET ceiling effect analysis — compare correction methods",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Existing ABIDE results CSV (auto-detected if not specified)",
    )
    parser.add_argument(
        "--atlas", choices=["cc200", "cc400"], default="cc200",
    )
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument(
        "--n-bootstraps", type=int, default=100,
        help="Bootstrap iterations (default: 100, use 30 for quick test)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel workers (1=sequential, -1=all cores)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Auto-detect CSV
    csv_path = args.csv
    if csv_path is None:
        candidates = sorted(
            Path("data/abide/results").glob(f"abide_bsnet_{args.atlas}_*.csv")
        )
        if not candidates:
            logger.error("No ABIDE results CSV found. Run run_abide_bsnet.py first.")
            sys.exit(1)
        csv_path = str(candidates[-1])
        logger.info(f"Auto-detected CSV: {csv_path}")

    run_ceiling_analysis(
        csv_path=csv_path,
        atlas=args.atlas,
        max_subjects=args.max_subjects,
        n_bootstraps=args.n_bootstraps,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
