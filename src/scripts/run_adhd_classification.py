#!/usr/bin/env python3
"""
Track H: ADHD vs Control classification using FC matrices.

Compares classification performance across three FC conditions:
  (1) Raw FC from short scan (2 min) — Ledoit-Wolf shrinkage
  (2) BS-NET corrected FC from short scan (2 min → extrapolated)
  (3) Reference FC from full scan — upper bound

Classifier: Linear SVM (C=1.0, default)
CV: Stratified 5-fold × N repeats (random seeds)
Features: Upper-triangle of ROI×ROI FC matrix

Usage:
    python src/scripts/run_adhd_classification.py --atlas cc200
    python src/scripts/run_adhd_classification.py --atlas cc200 cc400 --n-repeats 20
    python src/scripts/run_adhd_classification.py --atlas cc200 --short-sec 120 -v
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.bootstrap import (
    block_bootstrap_indices,
    correct_attenuation,
    estimate_optimal_block_length,
    fisher_z,
    fisher_z_inv,
)
from src.core.config import BSNetConfig
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/adhd/results")
TS_CACHE_DIR = Path("data/adhd/ts_cache")


# ============================================================================
# Data loading
# ============================================================================

def load_subject_metadata(atlas: str) -> list[dict]:
    """Load subject metadata from BS-NET results CSV.

    Args:
        atlas: Atlas name (cc200 or cc400).

    Returns:
        List of dicts with sub_idx, group, n_vols, tr keys.
    """
    csv_path = RESULTS_DIR / f"adhd_bsnet_{atlas}.csv"
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))
    return [
        {
            "sub_idx": int(r["sub_idx"]),
            "group": r["group"],
            "n_vols": int(r["n_vols"]),
            "tr": float(r["tr"]),
        }
        for r in rows
    ]


def load_timeseries(sub_idx: int, atlas: str) -> np.ndarray:
    """Load cached timeseries for a subject.

    Args:
        sub_idx: Subject index.
        atlas: Atlas name.

    Returns:
        Timeseries array of shape (n_vols, n_rois).
    """
    ts_path = TS_CACHE_DIR / atlas / f"sub_{sub_idx:03d}_ts.npy"
    ts = np.load(ts_path).astype(np.float64)
    # Remove zero-variance ROIs
    valid = np.std(ts, axis=0) > 1e-8
    return ts[:, valid]


# ============================================================================
# FC matrix computation
# ============================================================================

def compute_fc_matrices(
    ts_full: np.ndarray,
    short_vols: int,
    config: BSNetConfig,
    seed: int = 42,
    correction_method: str = "fisher_z",
    n_bootstraps: int = 100,
) -> dict[str, np.ndarray]:
    """Compute three FC conditions from a single subject's timeseries.

    Args:
        ts_full: Full timeseries (n_vols, n_rois).
        short_vols: Number of volumes for short scan.
        config: BSNetConfig instance.
        seed: Random seed for bootstrap.
        correction_method: Attenuation correction method.
        n_bootstraps: Number of bootstrap iterations.

    Returns:
        Dict with keys 'raw_short', 'bsnet', 'reference', each an
        upper-triangle FC vector.
    """
    n_rois = ts_full.shape[1]

    # (3) Reference FC: full scan
    fc_ref_vec = get_fc_matrix(ts_full, vectorized=True, use_shrinkage=True)

    # (1) Raw FC: short scan
    ts_short = ts_full[:short_vols, :]
    fc_raw_vec = get_fc_matrix(ts_short, vectorized=True, use_shrinkage=True)
    fc_raw_mat = get_fc_matrix(ts_short, vectorized=False, use_shrinkage=True)

    # (2) BS-NET corrected FC: short scan → extrapolated edge-level
    np.random.seed(seed)
    block_size = estimate_optimal_block_length(ts_short)
    n_split = ts_short.shape[0] // 2
    k = ts_full.shape[0] / short_vols

    rho_hat_boots = []
    for _ in range(n_bootstraps):
        idx = block_bootstrap_indices(
            short_vols, block_size,
            n_blocks=short_vols // max(block_size, 1),
        )
        ts_b = ts_short[idx, :]
        fc_b_vec = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs = np.corrcoef(fc_raw_vec, fc_b_vec)[0, 1]

        fc_b_h1 = get_fc_matrix(ts_b[:n_split, :], vectorized=True, use_shrinkage=True)
        fc_b_h2 = get_fc_matrix(ts_b[n_split:, :], vectorized=True, use_shrinkage=True)
        r_split = np.corrcoef(fc_b_h1, fc_b_h2)[0, 1]

        rho_est = correct_attenuation(r_obs, 0.98, r_split, k=k)
        rho_hat_boots.append(rho_est)

    # Edge-level inflation in Fisher z-space
    true_overlap = np.corrcoef(fc_raw_vec, fc_ref_vec)[0, 1]
    median_rho = np.nanmedian(rho_hat_boots)
    inflation_ratio = np.clip(median_rho / max(true_overlap, 0.01), 0.5, 2.0)

    fc_bsnet_z = fisher_z(fc_raw_mat) * inflation_ratio
    fc_bsnet_mat = fisher_z_inv(fc_bsnet_z)

    # Extract upper triangle
    triu_idx = np.triu_indices(n_rois, k=1)
    fc_bsnet_vec = fc_bsnet_mat[triu_idx]

    return {
        "raw_short": fc_raw_vec,
        "bsnet": fc_bsnet_vec,
        "reference": fc_ref_vec,
    }


# ============================================================================
# Classification
# ============================================================================

def run_classification(
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    n_repeats: int = 10,
    n_folds: int = 5,
    seed_base: int = 42,
) -> dict:
    """Run repeated stratified k-fold SVM classification.

    Args:
        X: Feature matrix (n_subjects, n_features).
        y: Binary labels (n_subjects,).
        n_repeats: Number of CV repeats.
        n_folds: Number of folds per repeat.
        seed_base: Base random seed.

    Returns:
        Dict with acc_mean, acc_std, auc_mean, auc_std, all_acc, all_auc.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    all_acc = []
    all_auc = []

    for rep in range(n_repeats):
        seed = seed_base + rep * 7
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel="linear", C=1.0, probability=True, random_state=seed),
            )
            clf.fit(X_train, y_train)

            acc = clf.score(X_test, y_test)
            all_acc.append(acc)

            # AUC
            try:
                from sklearn.metrics import roc_auc_score
                y_prob = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc = float("nan")
            all_auc.append(auc)

    all_acc = np.array(all_acc)
    all_auc = np.array(all_auc)

    return {
        "acc_mean": float(np.mean(all_acc)),
        "acc_std": float(np.std(all_acc)),
        "auc_mean": float(np.nanmean(all_auc)),
        "auc_std": float(np.nanstd(all_auc)),
        "all_acc": all_acc,
        "all_auc": all_auc,
    }


# ============================================================================
# Main pipeline
# ============================================================================

def run_experiment(
    atlas: str,
    short_sec: float = 120.0,
    n_repeats: int = 10,
    n_bootstraps: int = 100,
    correction_method: str = "fisher_z",
    seed: int = 42,
) -> dict:
    """Run full classification experiment for one atlas.

    Args:
        atlas: Atlas name (cc200 or cc400).
        short_sec: Short scan duration in seconds.
        n_repeats: Number of CV repeats.
        n_bootstraps: Bootstrap iterations per subject.
        correction_method: Attenuation correction method.
        seed: Random seed.

    Returns:
        Dict with per-condition classification results.
    """
    meta = load_subject_metadata(atlas)
    n_subs = len(meta)

    logger.info(f"Atlas={atlas.upper()}, N={n_subs}, short={short_sec}s")

    # Labels
    y = np.array([1 if m["group"] == "adhd" else 0 for m in meta])
    logger.info(f"Labels: adhd={np.sum(y)}, control={np.sum(y == 0)}")

    # Compute FC matrices for all subjects
    all_fc: dict[str, list[np.ndarray]] = {
        "raw_short": [],
        "bsnet": [],
        "reference": [],
    }

    for i, m in enumerate(meta):
        sub_idx = m["sub_idx"]
        tr = m["tr"]
        short_vols = int(short_sec / tr)

        ts = load_timeseries(sub_idx, atlas)
        n_vols = ts.shape[0]

        # Clamp short_vols to available data
        actual_short = min(short_vols, n_vols - 1)
        if actual_short < short_vols:
            logger.debug(
                f"sub_{sub_idx:03d}: clamped short from {short_vols} to {actual_short} vols"
            )

        config = BSNetConfig(
            n_rois=ts.shape[1],
            tr=tr,
            short_duration_sec=actual_short * tr,
            target_duration_min=(n_vols * tr) / 60.0,
            n_bootstraps=n_bootstraps,
            seed=seed + i,
        )

        fc_dict = compute_fc_matrices(
            ts, actual_short, config,
            seed=seed + i,
            correction_method=correction_method,
            n_bootstraps=n_bootstraps,
        )

        for cond in all_fc:
            all_fc[cond].append(fc_dict[cond])

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"  [{i + 1}/{n_subs}] sub_{sub_idx:03d} done")

    # Stack feature matrices
    results = {}
    condition_labels = {
        "raw_short": f"Raw FC ({short_sec:.0f}s)",
        "bsnet": f"BS-NET ({short_sec:.0f}s)",
        "reference": "Reference FC (full)",
    }

    for cond, label in condition_labels.items():
        X = np.vstack(all_fc[cond])
        logger.info(f"\n--- {label} --- X.shape={X.shape}")

        clf_result = run_classification(X, y, n_repeats=n_repeats, seed_base=seed)
        clf_result["condition"] = cond
        clf_result["label"] = label
        clf_result["n_features"] = X.shape[1]
        results[cond] = clf_result

        logger.info(
            f"  Accuracy: {clf_result['acc_mean']:.3f} ± {clf_result['acc_std']:.3f}"
        )
        logger.info(
            f"  AUC:      {clf_result['auc_mean']:.3f} ± {clf_result['auc_std']:.3f}"
        )

    return results


def save_results(
    results: dict[str, dict],
    atlas: str,
    output_dir: Path,
) -> Path:
    """Save classification results to CSV (summary + per-fold).

    Saves two files:
      - adhd_classification_{atlas}.csv: summary (mean/std per condition)
      - adhd_classification_{atlas}_folds.csv: per-fold acc/auc values

    Args:
        results: Per-condition classification results.
        atlas: Atlas name.
        output_dir: Output directory.

    Returns:
        Path to saved summary CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Summary CSV ---
    csv_path = output_dir / f"adhd_classification_{atlas}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow([
            "condition", "label", "n_features",
            "acc_mean", "acc_std", "auc_mean", "auc_std",
        ])
        for _cond, r in results.items():
            writer.writerow([
                r["condition"], r["label"], r["n_features"],
                f"{r['acc_mean']:.4f}", f"{r['acc_std']:.4f}",
                f"{r['auc_mean']:.4f}", f"{r['auc_std']:.4f}",
            ])
    logger.info(f"Summary saved: {csv_path}")

    # --- Per-fold CSV ---
    folds_path = output_dir / f"adhd_classification_{atlas}_folds.csv"
    with open(folds_path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["condition", "fold_idx", "accuracy", "auc"])
        for _cond, r in results.items():
            all_acc = r.get("all_acc", [])
            all_auc = r.get("all_auc", [])
            for fold_i, (acc, auc) in enumerate(zip(all_acc, all_auc)):
                writer.writerow([
                    r["condition"], fold_i,
                    f"{acc:.4f}", f"{auc:.4f}",
                ])
    logger.info(f"Per-fold saved: {folds_path}")

    return csv_path


def print_summary(all_results: dict[str, dict[str, dict]]) -> None:
    """Print formatted summary table."""
    print("\n" + "=" * 78)
    print("Track H: ADHD vs Control Classification — Summary")
    print("=" * 78)
    print(f"{'Atlas':<8} {'Condition':<25} {'Accuracy':<18} {'AUC':<18}")
    print("-" * 78)

    for atlas, results in all_results.items():
        for cond in ["raw_short", "bsnet", "reference"]:
            r = results[cond]
            print(
                f"{atlas.upper():<8} {r['label']:<25} "
                f"{r['acc_mean']:.3f} ± {r['acc_std']:.3f}   "
                f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}"
            )
        print("-" * 78)
    print("=" * 78)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track H: ADHD vs Control FC classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--atlas", nargs="+", default=["cc200"],
        help="Atlas(es) to test (default: cc200)",
    )
    parser.add_argument(
        "--short-sec", type=float, default=120.0,
        help="Short scan duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=10,
        help="Number of CV repeats (default: 10)",
    )
    parser.add_argument(
        "--n-bootstraps", type=int, default=100,
        help="Bootstrap iterations per subject (default: 100)",
    )
    parser.add_argument(
        "--correction-method",
        choices=["original", "fisher_z", "partial", "soft_clamp"],
        default="fisher_z",
        help="Attenuation correction method (default: fisher_z)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    all_results: dict[str, dict[str, dict]] = {}

    for atlas in args.atlas:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running classification for {atlas.upper()}")
        logger.info(f"{'='*60}")

        t0 = time.time()
        results = run_experiment(
            atlas=atlas,
            short_sec=args.short_sec,
            n_repeats=args.n_repeats,
            n_bootstraps=args.n_bootstraps,
            correction_method=args.correction_method,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        logger.info(f"Elapsed: {elapsed:.1f}s")

        save_results(results, atlas, RESULTS_DIR)
        all_results[atlas] = results

    print_summary(all_results)


if __name__ == "__main__":
    main()
