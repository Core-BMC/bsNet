#!/usr/bin/env python3
"""Evaluate signal recovery experiment: Condition A vs B vs C.

Reads imputed time series from Diffusion-TS output and compares FC quality
against reference FC.

Conditions:
  A: Naive imputation (Diffusion-TS, no reliability)
  B: Reliability-guided imputation (Diffusion-TS + BS-NET ρ̂T weighting)
  C: BS-NET only (ρ̂T from bootstrap + SB prophecy, no generation)

Metrics:
  - r_FC: Pearson correlation between imputed FC and reference FC (vectorized upper triangle)
  - edge_ICC: Per-edge intraclass correlation (mean over edges)
  - network_ARI: Adjusted Rand Index of community structure (Louvain)
  - stratified_r_FC: r_FC stratified by reliability tertile

Usage:
  PYTHONPATH=. python3 src/scripts/eval_signal_recovery.py \\
      --data-dir data/ds000243/signal_recovery/harvard_oxford \\
      --imputed-a results/imputed_naive.npy \\
      --imputed-b results/imputed_guided.npy \\
      --output results/signal_recovery_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FC computation
# ---------------------------------------------------------------------------

def compute_fc_upper_triangle(ts: np.ndarray) -> np.ndarray:
    """[T, N_ROI] → [N_edges] upper triangle of Pearson correlation."""
    fc = np.corrcoef(ts.T)
    idx = np.triu_indices_from(fc, k=1)
    return fc[idx].astype(np.float64)


def compute_fc_matrix(ts: np.ndarray) -> np.ndarray:
    """[T, N_ROI] → [N_ROI, N_ROI] Pearson correlation matrix."""
    return np.corrcoef(ts.T).astype(np.float64)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_r_fc(fc_pred: np.ndarray, fc_ref: np.ndarray) -> float:
    """Pearson correlation between two FC vectors."""
    r, _ = stats.pearsonr(fc_pred, fc_ref)
    return float(r)


def metric_edge_icc(
    fc_pred_vec: np.ndarray,
    fc_ref_vec: np.ndarray,
) -> float:
    """Mean absolute agreement ICC(2,1) over edges (simplified).

    Treats pred and ref as two raters measuring each edge.
    """
    n = len(fc_pred_vec)
    grand_mean = (fc_pred_vec.mean() + fc_ref_vec.mean()) / 2
    ss_between = n * ((fc_pred_vec.mean() - grand_mean) ** 2 +
                       (fc_ref_vec.mean() - grand_mean) ** 2)
    ss_within = ((fc_pred_vec - fc_ref_vec) ** 2).sum() / 2
    ms_between = ss_between / 1  # k-1 = 2-1 = 1
    ms_within = ss_within / n if n > 0 else 1e-8

    icc = (ms_between - ms_within) / (ms_between + ms_within)
    return float(np.clip(icc, -1, 1))


def metric_network_ari(
    fc_mat_pred: np.ndarray,
    fc_mat_ref: np.ndarray,
    n_communities: int = 7,
) -> float:
    """Adjusted Rand Index between community assignments.

    Uses spectral clustering as a deterministic proxy for Louvain.
    """
    try:
        from sklearn.cluster import SpectralClustering
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        logger.warning("sklearn not available, skipping ARI")
        return float("nan")

    def _cluster(fc_mat: np.ndarray) -> np.ndarray:
        # Convert correlation to affinity (shift to positive)
        affinity = (fc_mat + 1) / 2
        np.fill_diagonal(affinity, 0)
        affinity = np.clip(affinity, 0, 1)
        sc = SpectralClustering(
            n_clusters=n_communities,
            affinity="precomputed",
            random_state=42,
        )
        return sc.fit_predict(affinity)

    labels_pred = _cluster(fc_mat_pred)
    labels_ref = _cluster(fc_mat_ref)
    return float(adjusted_rand_score(labels_ref, labels_pred))


def metric_stratified_r_fc(
    fc_pred_vec: np.ndarray,
    fc_ref_vec: np.ndarray,
    edge_reliability: np.ndarray,
) -> dict[str, float]:
    """r_FC stratified by reliability tertile.

    Args:
        edge_reliability: [N_edges] per-edge reliability (upper triangle)
    """
    tertiles = np.percentile(edge_reliability, [33.3, 66.7])
    t1_mask = edge_reliability < tertiles[0]
    t2_mask = (edge_reliability >= tertiles[0]) & (edge_reliability < tertiles[1])
    t3_mask = edge_reliability >= tertiles[1]

    results = {}
    for name, mask in [("T1_low", t1_mask), ("T2_mid", t2_mask), ("T3_high", t3_mask)]:
        if mask.sum() > 2:
            r, _ = stats.pearsonr(fc_pred_vec[mask], fc_ref_vec[mask])
            results[f"r_FC_{name}"] = float(r)
        else:
            results[f"r_FC_{name}"] = float("nan")

    return results


# ---------------------------------------------------------------------------
# Evaluation for a single condition
# ---------------------------------------------------------------------------

def evaluate_condition(
    imputed_ts: np.ndarray | None,
    test_short: np.ndarray,
    test_long: np.ndarray,
    test_fc_ref: np.ndarray,
    edge_reliability: np.ndarray | None = None,
    condition_name: str = "unknown",
) -> list[dict]:
    """Evaluate one condition across all test subjects.

    Args:
        imputed_ts: [N_test, seq_len, N_ROI] imputed time series (None for Condition C)
        test_short: [N_test, short_len, N_ROI]
        test_long: [N_test, seq_len, N_ROI] reference
        test_fc_ref: [N_test, N_edges] reference FC
        edge_reliability: [N_test, N_ROI, N_ROI] (optional, for stratified metric)
        condition_name: label for this condition

    Returns:
        List of per-subject result dicts
    """
    n_test = test_short.shape[0]
    results = []

    for i in range(n_test):
        row: dict[str, Any] = {"subject_idx": i, "condition": condition_name}

        if imputed_ts is not None:
            # Compute FC from imputed time series
            fc_pred_vec = compute_fc_upper_triangle(imputed_ts[i])
            fc_pred_mat = compute_fc_matrix(imputed_ts[i])
        else:
            # Condition C: use short scan FC directly
            fc_pred_vec = compute_fc_upper_triangle(test_short[i])
            fc_pred_mat = compute_fc_matrix(test_short[i])

        fc_ref_vec = test_fc_ref[i]
        fc_ref_mat = compute_fc_matrix(test_long[i])

        # Core metrics
        row["r_FC"] = metric_r_fc(fc_pred_vec, fc_ref_vec)
        row["edge_ICC"] = metric_edge_icc(fc_pred_vec, fc_ref_vec)
        row["network_ARI"] = metric_network_ari(fc_pred_mat, fc_ref_mat)

        # Stratified metrics (if reliability available)
        if edge_reliability is not None:
            n_roi = edge_reliability.shape[-1]
            idx = np.triu_indices(n_roi, k=1)
            edge_rel_vec = edge_reliability[i][idx]
            strat = metric_stratified_r_fc(fc_pred_vec, fc_ref_vec, edge_rel_vec)
            row.update(strat)

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate signal recovery experiment")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--imputed-a", type=str, default=None,
                        help="Path to imputed time series (Condition A: naive)")
    parser.add_argument("--imputed-b", type=str, default=None,
                        help="Path to imputed time series (Condition B: reliability-guided)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: data_dir/eval_results.csv)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output) if args.output else data_dir / "eval_results.csv"

    # Load data
    test_short = np.load(data_dir / "test_short.npy")
    test_long = np.load(data_dir / "test_long.npy")
    test_fc_ref = np.load(data_dir / "test_fc_ref.npy")

    edge_rel = None
    rel_path = data_dir / "test_edge_reliability.npy"
    if rel_path.exists():
        edge_rel = np.load(rel_path)

    with open(data_dir / "split_info.json") as f:
        split_info = json.load(f)

    n_test = test_short.shape[0]
    logger.info(f"Evaluating {n_test} test subjects")

    all_results = []

    # --- Condition C: Raw short FC (baseline) ---
    logger.info("Condition C: Raw short FC (no imputation)")
    results_c = evaluate_condition(
        imputed_ts=None,
        test_short=test_short,
        test_long=test_long,
        test_fc_ref=test_fc_ref,
        edge_reliability=edge_rel,
        condition_name="C_raw_short",
    )
    all_results.extend(results_c)

    # --- Condition A: Naive imputation ---
    if args.imputed_a:
        logger.info(f"Condition A: Naive imputation from {args.imputed_a}")
        imputed_a = np.load(args.imputed_a)
        results_a = evaluate_condition(
            imputed_ts=imputed_a,
            test_short=test_short,
            test_long=test_long,
            test_fc_ref=test_fc_ref,
            edge_reliability=edge_rel,
            condition_name="A_naive_imputation",
        )
        all_results.extend(results_a)

    # --- Condition B: Reliability-guided imputation ---
    if args.imputed_b:
        logger.info(f"Condition B: Reliability-guided imputation from {args.imputed_b}")
        imputed_b = np.load(args.imputed_b)
        results_b = evaluate_condition(
            imputed_ts=imputed_b,
            test_short=test_short,
            test_long=test_long,
            test_fc_ref=test_fc_ref,
            edge_reliability=edge_rel,
            condition_name="B_reliability_guided",
        )
        all_results.extend(results_b)

    # --- Write results ---
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        logger.info(f"\nResults saved to {output_path}")

        # Summary
        logger.info("\n=== Summary ===")
        for cond in sorted({r["condition"] for r in all_results}):
            cond_rows = [r for r in all_results if r["condition"] == cond]
            r_fcs = [r["r_FC"] for r in cond_rows]
            logger.info(
                f"  {cond}: r_FC = {np.mean(r_fcs):.4f} ± {np.std(r_fcs):.4f} "
                f"(n={len(r_fcs)})"
            )
    else:
        logger.warning("No results to save. Provide --imputed-a and/or --imputed-b.")


if __name__ == "__main__":
    main()
