#!/usr/bin/env python3
"""Downstream FC analysis with ρ̂T-stratified quality assessment.

Demonstrates that BS-NET's ρ̂T is a meaningful quality predictor for
downstream FC analyses — not just a correlation metric.

Core hypothesis:
  Higher ρ̂T → higher-quality FC → better downstream performance.
  Proved via tertile stratification: T1 (low) < T2 (mid) < T3 (high).

Analyses:
  1. Subject-level FC similarity (short vs reference)
  2. Group-mean connectome similarity (ADHD / Control / All)
  3. Group contrast: edge-wise Cohen's d pattern correlation
  4. Classification: ADHD vs Control (Linear SVM, 5-fold × 5 repeats)
  5. Graph metrics: degree, clustering, modularity (binarized)
  6. ρ̂T-stratified downstream quality — dose-response proof

FC types compared:
  - raw_short:  np.corrcoef on 2-min (60 TRs) — baseline
  - lw_short:   Ledoit-Wolf shrinkage on 2-min — BS-NET component
  - reference:  np.corrcoef on remaining scan — "gold standard"

Usage:
    PYTHONPATH=. python src/scripts/run_downstream_analysis.py

    # Quick test (N=50)
    PYTHONPATH=. python src/scripts/run_downstream_analysis.py --max-subjects 50

    # Skip SVM classification
    PYTHONPATH=. python src/scripts/run_downstream_analysis.py --skip-classification
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.data_loader import get_fc_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SHORT_TRS = 60

# BS-NET results CSV (from run_adhd200_pcp_filtered.py)
BSNET_CSV = Path("data/adhd/pcp/results/adhd200_multiseed_cc200_10seeds_filtered_strict.csv")


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_adhd200_subjects(filter_mode: str = "strict") -> list[dict]:
    """Load ADHD-200 PCP subjects with strict filter."""
    json_path = Path("data/adhd/pcp/results/adhd200_subjects_cc200.json")
    with open(json_path) as f:
        subs = json.load(f)

    known = [s for s in subs if s["group"] in ("Control", "ADHD")]
    for s in known:
        s["k"] = s["n_trs"] / SHORT_TRS
        s["ref_s"] = (s["n_trs"] - SHORT_TRS) * s["tr"]

    if filter_mode == "strict":
        filtered = [s for s in known if s["k"] >= 2.0 and s["ref_s"] >= 300]
    elif filter_mode == "moderate":
        filtered = [s for s in known if s["k"] >= 2.0 and s["ref_s"] >= 180]
    else:
        filtered = [s for s in known if s["k"] >= 2.0]

    return filtered


def load_bsnet_rho(csv_path: Path = BSNET_CSV) -> dict[str, float]:
    """Load per-subject ρ̂T from BS-NET results CSV.

    Returns:
        Dict mapping sub_id → rho_hat_T_mean.
    """
    rho_map: dict[str, float] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub_id = row["sub_id"]
            rho = float(row["rho_hat_T_mean"])
            rho_map[sub_id] = rho
    logger.info(f"Loaded ρ̂T for {len(rho_map)} subjects from {csv_path.name}")
    return rho_map


def compute_fc_triplet(sub: dict) -> dict | None:
    """Compute 3 FC matrices for one subject: raw_short, lw_short, reference.

    Returns dict with vectorized upper-triangle FC arrays.
    """
    try:
        ts = np.load(sub["ts_path"])
    except Exception:
        return None

    n_trs = ts.shape[0]
    if n_trs <= SHORT_TRS + 10:
        return None

    ts_short = ts[:SHORT_TRS]
    ts_ref = ts[SHORT_TRS:]

    fc_raw_short = get_fc_matrix(ts_short, vectorized=True, use_shrinkage=False)
    fc_lw_short = get_fc_matrix(ts_short, vectorized=True, use_shrinkage=True)
    fc_ref = get_fc_matrix(ts_ref, vectorized=True, use_shrinkage=False)

    return {
        "sub_id": sub["sub_id"],
        "site": sub["site"],
        "group": sub["group"],
        "age": sub.get("age", -1.0),
        "fc_raw_short": fc_raw_short,
        "fc_lw_short": fc_lw_short,
        "fc_ref": fc_ref,
        "n_rois": ts.shape[1],
        "n_trs": n_trs,
        "tr": sub["tr"],
    }


# ═══════════════════════════════════════════════════════════════════════
# Tertile stratification helpers
# ═══════════════════════════════════════════════════════════════════════

def assign_tertiles(data: list[dict], rho_map: dict[str, float]) -> list[dict]:
    """Assign ρ̂T values and tertile labels (T1/T2/T3) to each subject.

    Modifies data in-place and returns only subjects with valid ρ̂T.
    """
    valid = []
    for d in data:
        rho = rho_map.get(d["sub_id"])
        if rho is not None:
            d["rho_hat_T"] = rho
            valid.append(d)

    if not valid:
        return valid

    rho_vals = np.array([d["rho_hat_T"] for d in valid])
    t1_cutoff = float(np.percentile(rho_vals, 33.33))
    t2_cutoff = float(np.percentile(rho_vals, 66.67))

    for d in valid:
        if d["rho_hat_T"] <= t1_cutoff:
            d["tertile"] = "T1_low"
        elif d["rho_hat_T"] <= t2_cutoff:
            d["tertile"] = "T2_mid"
        else:
            d["tertile"] = "T3_high"

    # Log tertile summary
    for t_label in ["T1_low", "T2_mid", "T3_high"]:
        t_subs = [d for d in valid if d["tertile"] == t_label]
        t_rhos = [d["rho_hat_T"] for d in t_subs]
        logger.info(
            f"  {t_label}: N={len(t_subs)}, "
            f"ρ̂T={np.mean(t_rhos):.4f}±{np.std(t_rhos):.4f} "
            f"[{np.min(t_rhos):.3f}–{np.max(t_rhos):.3f}]"
        )

    return valid


def get_tertile_subsets(data: list[dict]) -> dict[str, list[dict]]:
    """Split data into tertile subsets."""
    return {
        t: [d for d in data if d.get("tertile") == t]
        for t in ["T1_low", "T2_mid", "T3_high"]
    }


# ═══════════════════════════════════════════════════════════════════════
# Analysis 1: Subject-level FC Similarity
# ═══════════════════════════════════════════════════════════════════════

def analysis_subject_fc_similarity(data: list[dict]) -> dict:
    """Per-subject FC similarity between short/LW and reference."""
    r_raw = []
    r_lw = []
    for d in data:
        r_raw.append(float(np.corrcoef(d["fc_raw_short"], d["fc_ref"])[0, 1]))
        r_lw.append(float(np.corrcoef(d["fc_lw_short"], d["fc_ref"])[0, 1]))

    r_raw = np.array(r_raw)
    r_lw = np.array(r_lw)
    improvement = r_lw - r_raw

    results = {
        "raw_short": {
            "mean": round(float(np.mean(r_raw)), 4),
            "std": round(float(np.std(r_raw)), 4),
        },
        "lw_short": {
            "mean": round(float(np.mean(r_lw)), 4),
            "std": round(float(np.std(r_lw)), 4),
        },
        "improvement": {
            "mean": round(float(np.mean(improvement)), 4),
            "pct_improved": round(float(100 * np.mean(improvement > 0)), 1),
        },
    }

    logger.info(f"  Subject-level FC similarity to reference:")
    logger.info(f"    raw_short: r = {np.mean(r_raw):.4f} ± {np.std(r_raw):.4f}")
    logger.info(f"    lw_short:  r = {np.mean(r_lw):.4f} ± {np.std(r_lw):.4f}")
    logger.info(f"    Δ = {np.mean(improvement):+.4f} ({100*np.mean(improvement>0):.1f}% improved)")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis 2: Group-mean Connectome Similarity
# ═══════════════════════════════════════════════════════════════════════

def analysis_connectome_similarity(data: list[dict]) -> dict:
    """Compare group-mean FC matrices."""
    results = {}

    for group in ["ADHD", "Control", "all"]:
        if group == "all":
            subset = data
        else:
            subset = [d for d in data if d["group"] == group]

        if len(subset) < 3:
            continue

        fc_raw_mean = np.mean([d["fc_raw_short"] for d in subset], axis=0)
        fc_lw_mean = np.mean([d["fc_lw_short"] for d in subset], axis=0)
        fc_ref_mean = np.mean([d["fc_ref"] for d in subset], axis=0)

        r_raw_ref = float(np.corrcoef(fc_raw_mean, fc_ref_mean)[0, 1])
        r_lw_ref = float(np.corrcoef(fc_lw_mean, fc_ref_mean)[0, 1])

        results[group] = {
            "n": len(subset),
            "raw_vs_ref": round(r_raw_ref, 4),
            "lw_vs_ref": round(r_lw_ref, 4),
            "delta": round(r_lw_ref - r_raw_ref, 4),
        }

    logger.info("  Group-mean FC similarity to reference:")
    for group, res in results.items():
        logger.info(
            f"    {group:10s} (N={res['n']:3d}): "
            f"raw r={res['raw_vs_ref']:.4f}  LW r={res['lw_vs_ref']:.4f}  "
            f"Δ={res['delta']:+.4f}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis 3: Group Contrast (Edge-wise Cohen's d)
# ═══════════════════════════════════════════════════════════════════════

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
    """Compute Cohen's d for each edge (column)."""
    n1, n2 = group1.shape[0], group2.shape[0]
    m1, m2 = np.mean(group1, axis=0), np.mean(group2, axis=0)
    s1, s2 = np.var(group1, axis=0, ddof=1), np.var(group2, axis=0, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    pooled_std = np.maximum(pooled_std, 1e-10)
    return (m1 - m2) / pooled_std


def _group_contrast_core(data: list[dict]) -> dict:
    """Core group contrast computation. Returns results with d_map arrays."""
    adhd = [d for d in data if d["group"] == "ADHD"]
    ctrl = [d for d in data if d["group"] == "Control"]

    if len(adhd) < 5 or len(ctrl) < 5:
        return {"error": f"Too few subjects: ADHD={len(adhd)}, Control={len(ctrl)}"}

    results = {}
    for fc_type in ["fc_raw_short", "fc_lw_short", "fc_ref"]:
        fc_adhd = np.array([d[fc_type] for d in adhd])
        fc_ctrl = np.array([d[fc_type] for d in ctrl])

        d_map = compute_cohens_d(fc_adhd, fc_ctrl)
        results[fc_type] = {
            "d_mean": float(np.mean(np.abs(d_map))),
            "d_max": float(np.max(np.abs(d_map))),
            "n_sig_edges_small": int(np.sum(np.abs(d_map) > 0.2)),
            "n_sig_edges_medium": int(np.sum(np.abs(d_map) > 0.5)),
            "d_map": d_map,
        }

    d_ref = results["fc_ref"]["d_map"]
    d_raw = results["fc_raw_short"]["d_map"]
    d_lw = results["fc_lw_short"]["d_map"]

    r_raw_ref = float(np.corrcoef(d_raw, d_ref)[0, 1])
    r_lw_ref = float(np.corrcoef(d_lw, d_ref)[0, 1])

    results["pattern_corr"] = {
        "raw_vs_ref": round(r_raw_ref, 4),
        "lw_vs_ref": round(r_lw_ref, 4),
        "delta": round(r_lw_ref - r_raw_ref, 4),
    }
    results["n_adhd"] = len(adhd)
    results["n_ctrl"] = len(ctrl)

    return results


def analysis_group_contrast(data: list[dict]) -> dict:
    """Edge-wise group contrast: ADHD vs Control effect sizes."""
    adhd = [d for d in data if d["group"] == "ADHD"]
    ctrl = [d for d in data if d["group"] == "Control"]
    logger.info(f"Group contrast: ADHD={len(adhd)}, Control={len(ctrl)}")

    results = _group_contrast_core(data)

    if "pattern_corr" in results:
        pc = results["pattern_corr"]
        logger.info(f"  Effect size pattern correlation with reference:")
        logger.info(f"    raw_short vs ref:  r = {pc['raw_vs_ref']:.4f}")
        logger.info(f"    lw_short  vs ref:  r = {pc['lw_vs_ref']:.4f}")
        logger.info(f"    Δ = {pc['delta']:+.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis 4: Classification (SVM)
# ═══════════════════════════════════════════════════════════════════════

def _classify_core(
    data: list[dict],
    n_repeats: int = 5,
    fc_types: list[str] | None = None,
) -> dict:
    """Core SVM classification. Returns accuracy per FC type."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    if fc_types is None:
        fc_types = ["fc_raw_short", "fc_lw_short", "fc_ref"]

    labels = np.array([1 if d["group"] == "ADHD" else 0 for d in data])

    # Need at least 5 per class for 5-fold CV
    n_adhd = int(np.sum(labels == 1))
    n_ctrl = int(np.sum(labels == 0))
    if n_adhd < 5 or n_ctrl < 5:
        return {"error": f"Too few subjects: ADHD={n_adhd}, Control={n_ctrl}"}

    n_splits = min(5, n_adhd, n_ctrl)

    results = {}
    for fc_type in fc_types:
        X = np.array([d[fc_type] for d in data])
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=42
        )
        clf = make_pipeline(
            StandardScaler(), LinearSVC(max_iter=5000, random_state=42)
        )
        scores = []
        for train_idx, test_idx in cv.split(X, labels):
            clf.fit(X[train_idx], labels[train_idx])
            scores.append(clf.score(X[test_idx], labels[test_idx]))

        results[fc_type] = {
            "accuracy_mean": round(float(np.mean(scores)), 4),
            "accuracy_std": round(float(np.std(scores)), 4),
            "n_folds": len(scores),
        }

    return results


def analysis_classification(data: list[dict], n_repeats: int = 5) -> dict:
    """ADHD vs Control classification using Linear SVM."""
    results = _classify_core(data, n_repeats=n_repeats)

    for fc_type in ["fc_raw_short", "fc_lw_short", "fc_ref"]:
        if fc_type in results:
            r = results[fc_type]
            logger.info(
                f"  {fc_type:15s}: Acc = {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}"
            )

    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis 5: Graph Metrics
# ═══════════════════════════════════════════════════════════════════════

def _vec_to_matrix(vec: np.ndarray, n_rois: int) -> np.ndarray:
    """Convert vectorized upper triangle back to symmetric matrix."""
    mat = np.zeros((n_rois, n_rois))
    idx = np.triu_indices(n_rois, k=1)
    mat[idx] = vec
    mat += mat.T
    np.fill_diagonal(mat, 1.0)
    return mat


def compute_graph_metrics(
    fc_vec: np.ndarray,
    n_rois: int,
    threshold: float = 0.2,
) -> dict:
    """Compute graph metrics from FC vector (thresholded binarized adjacency)."""
    mat = _vec_to_matrix(fc_vec, n_rois)
    adj = (np.abs(mat) > threshold).astype(float)
    np.fill_diagonal(adj, 0)

    n = adj.shape[0]
    degree = np.sum(adj, axis=1)
    mean_degree = float(np.mean(degree))
    density = float(np.sum(adj)) / (n * (n - 1))

    # Clustering coefficient (local transitivity)
    triangles = np.diag(adj @ adj @ adj) / 2
    denom = degree * (degree - 1)
    denom[denom == 0] = 1
    clustering = triangles / denom
    mean_clustering = float(np.mean(clustering))

    # Algebraic connectivity (Fiedler value)
    D = np.diag(degree)
    L = D - adj
    try:
        eigvals = np.linalg.eigvalsh(L)
        algebraic_connectivity = float(eigvals[1]) if len(eigvals) > 1 else 0.0
    except Exception:
        algebraic_connectivity = 0.0

    return {
        "mean_degree": round(mean_degree, 2),
        "density": round(density, 4),
        "mean_clustering": round(mean_clustering, 4),
        "algebraic_connectivity": round(algebraic_connectivity, 4),
    }


def analysis_graph_metrics(data: list[dict]) -> dict:
    """Compare graph metrics across FC types."""
    n_rois = data[0]["n_rois"]
    results = {}

    for fc_type in ["fc_raw_short", "fc_lw_short", "fc_ref"]:
        metrics_list = [
            compute_graph_metrics(d[fc_type], n_rois) for d in data
        ]
        agg = {}
        for key in metrics_list[0]:
            vals = [m[key] for m in metrics_list]
            agg[key] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
            }
        results[fc_type] = agg

    # Per-subject metric correlation with reference
    metric_corrs = {}
    for metric_name in ["mean_degree", "density", "mean_clustering"]:
        ref_vals, raw_vals, lw_vals = [], [], []
        for d in data:
            ref_m = compute_graph_metrics(d["fc_ref"], n_rois)
            raw_m = compute_graph_metrics(d["fc_raw_short"], n_rois)
            lw_m = compute_graph_metrics(d["fc_lw_short"], n_rois)
            ref_vals.append(ref_m[metric_name])
            raw_vals.append(raw_m[metric_name])
            lw_vals.append(lw_m[metric_name])

        r_raw = float(np.corrcoef(raw_vals, ref_vals)[0, 1])
        r_lw = float(np.corrcoef(lw_vals, ref_vals)[0, 1])
        metric_corrs[metric_name] = {
            "raw_vs_ref": round(r_raw, 4),
            "lw_vs_ref": round(r_lw, 4),
            "delta": round(r_lw - r_raw, 4),
        }

    results["metric_preservation"] = metric_corrs

    logger.info("  Graph metric preservation (per-subject corr with reference):")
    for name, corrs in metric_corrs.items():
        logger.info(
            f"    {name:25s}: raw r={corrs['raw_vs_ref']:.3f}  "
            f"LW r={corrs['lw_vs_ref']:.3f}  Δ={corrs['delta']:+.3f}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis 6: ρ̂T-Stratified Downstream Quality (Dose-Response Proof)
# ═══════════════════════════════════════════════════════════════════════

def _per_subject_fc_similarity(d: dict) -> float:
    """Single-subject raw-short vs reference correlation."""
    return float(np.corrcoef(d["fc_raw_short"], d["fc_ref"])[0, 1])


def analysis_rho_stratified(
    data: list[dict],
    skip_classification: bool = False,
) -> dict:
    """ρ̂T-stratified analysis: dose-response proof.

    For each tertile (T1/T2/T3), compute:
      - Subject-level FC similarity (raw_short vs ref)
      - Group-mean connectome similarity
      - Cohen's d pattern correlation
      - SVM classification accuracy (optional)
      - Graph metric preservation

    Expected result: monotonic improvement T1 < T2 < T3 across all metrics.
    """
    tertiles = get_tertile_subsets(data)
    results = {}

    for t_label, t_data in tertiles.items():
        if len(t_data) < 10:
            logger.warning(f"  {t_label}: only {len(t_data)} subjects, skipping")
            continue

        rho_vals = [d["rho_hat_T"] for d in t_data]
        t_result: dict = {
            "n": len(t_data),
            "rho_hat_T_mean": round(float(np.mean(rho_vals)), 4),
            "rho_hat_T_std": round(float(np.std(rho_vals)), 4),
            "rho_hat_T_range": [
                round(float(np.min(rho_vals)), 4),
                round(float(np.max(rho_vals)), 4),
            ],
        }

        # 6a. Subject-level FC similarity
        r_fc_vals = [_per_subject_fc_similarity(d) for d in t_data]
        t_result["fc_similarity"] = {
            "mean": round(float(np.mean(r_fc_vals)), 4),
            "std": round(float(np.std(r_fc_vals)), 4),
        }

        # 6b. Group-mean connectome similarity
        fc_raw_mean = np.mean([d["fc_raw_short"] for d in t_data], axis=0)
        fc_ref_mean = np.mean([d["fc_ref"] for d in t_data], axis=0)
        r_group_mean = float(np.corrcoef(fc_raw_mean, fc_ref_mean)[0, 1])
        t_result["group_mean_fc_similarity"] = round(r_group_mean, 4)

        # 6c. Cohen's d pattern correlation
        contrast = _group_contrast_core(t_data)
        if "pattern_corr" in contrast:
            t_result["cohens_d_pattern_corr"] = contrast["pattern_corr"]["raw_vs_ref"]
            t_result["cohens_d_n_adhd"] = contrast["n_adhd"]
            t_result["cohens_d_n_ctrl"] = contrast["n_ctrl"]
        else:
            t_result["cohens_d_pattern_corr"] = None
            t_result["cohens_d_note"] = contrast.get("error", "insufficient data")

        # 6d. Classification accuracy
        if not skip_classification:
            try:
                clf_result = _classify_core(
                    t_data, n_repeats=5, fc_types=["fc_raw_short"]
                )
                if "fc_raw_short" in clf_result:
                    t_result["svm_accuracy"] = clf_result["fc_raw_short"]["accuracy_mean"]
                else:
                    t_result["svm_accuracy"] = None
                    t_result["svm_note"] = clf_result.get("error", "failed")
            except Exception as e:
                t_result["svm_accuracy"] = None
                t_result["svm_note"] = str(e)
        else:
            t_result["svm_accuracy"] = None

        # 6e. Graph metric preservation (mean degree correlation)
        n_rois = t_data[0]["n_rois"]
        ref_degrees, raw_degrees = [], []
        for d in t_data:
            ref_m = compute_graph_metrics(d["fc_ref"], n_rois)
            raw_m = compute_graph_metrics(d["fc_raw_short"], n_rois)
            ref_degrees.append(ref_m["mean_degree"])
            raw_degrees.append(raw_m["mean_degree"])
        if np.std(ref_degrees) > 1e-8 and np.std(raw_degrees) > 1e-8:
            r_degree = float(np.corrcoef(raw_degrees, ref_degrees)[0, 1])
        else:
            r_degree = 0.0
        t_result["graph_degree_preservation"] = round(r_degree, 4)

        results[t_label] = t_result

        logger.info(
            f"  {t_label} (N={t_result['n']}, ρ̂T={t_result['rho_hat_T_mean']:.3f}): "
            f"FC_sim={t_result['fc_similarity']['mean']:.3f}  "
            f"GroupFC={t_result['group_mean_fc_similarity']:.3f}  "
            f"Cohen_d_r={t_result.get('cohens_d_pattern_corr', 'N/A')}  "
            f"SVM={t_result.get('svm_accuracy', 'N/A')}  "
            f"Degree_r={t_result['graph_degree_preservation']:.3f}"
        )

    # Summary: monotonicity check
    t_labels = ["T1_low", "T2_mid", "T3_high"]
    available = [t for t in t_labels if t in results]

    if len(available) == 3:
        monotonic_metrics = {}
        for metric in ["fc_similarity", "group_mean_fc_similarity", "cohens_d_pattern_corr"]:
            vals = []
            for t in available:
                if metric == "fc_similarity":
                    v = results[t]["fc_similarity"]["mean"]
                elif metric == "group_mean_fc_similarity":
                    v = results[t]["group_mean_fc_similarity"]
                else:
                    v = results[t].get("cohens_d_pattern_corr")
                vals.append(v)

            if all(v is not None for v in vals):
                is_mono = vals[0] < vals[1] < vals[2]
                monotonic_metrics[metric] = {
                    "values": [round(v, 4) for v in vals],
                    "monotonically_increasing": is_mono,
                }

        results["monotonicity_check"] = monotonic_metrics

        n_monotonic = sum(
            1 for m in monotonic_metrics.values()
            if m.get("monotonically_increasing")
        )
        logger.info(
            f"\n  Monotonicity check: {n_monotonic}/{len(monotonic_metrics)} metrics "
            f"show T1 < T2 < T3 dose-response"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# Analysis 7: Fingerprinting (Test-Retest Subject Identification)
# ═══════════════════════════════════════════════════════════════════════

def analysis_fingerprinting(data: list[dict]) -> dict:
    """FC fingerprinting: can we identify subjects from short scans?

    For each subject, compute similarity between their short-scan FC and
    every subject's reference FC. The subject is 'identified' if the
    highest similarity is with their own reference FC.
    """
    n = len(data)
    if n < 10:
        return {"error": "Too few subjects for fingerprinting"}

    results = {}
    for fc_type in ["fc_raw_short", "fc_lw_short"]:
        # Build similarity matrix: short(i) × ref(j)
        correct = 0
        ranks = []
        for i in range(n):
            fc_short_i = data[i][fc_type]
            sims = []
            for j in range(n):
                fc_ref_j = data[j]["fc_ref"]
                r = float(np.corrcoef(fc_short_i, fc_ref_j)[0, 1])
                sims.append(r)

            sims = np.array(sims)
            # Rank of the true match (self)
            rank = int(np.sum(sims > sims[i])) + 1  # 1-based rank
            ranks.append(rank)
            if rank == 1:
                correct += 1

        id_rate = correct / n
        results[fc_type] = {
            "identification_rate": round(id_rate, 4),
            "mean_rank": round(float(np.mean(ranks)), 2),
            "median_rank": round(float(np.median(ranks)), 1),
        }
        logger.info(
            f"  {fc_type:15s}: ID rate = {id_rate:.3f} "
            f"(mean rank = {np.mean(ranks):.1f})"
        )

    delta = (
        results["fc_lw_short"]["identification_rate"]
        - results["fc_raw_short"]["identification_rate"]
    )
    results["delta_id_rate"] = round(delta, 4)

    return results


# ═══════════════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════════════

def export_per_subject_csv(
    data: list[dict],
    out_path: Path,
) -> None:
    """Export per-subject downstream metrics to CSV for plotting."""
    rows = []
    for d in data:
        r_fc_raw = float(np.corrcoef(d["fc_raw_short"], d["fc_ref"])[0, 1])
        r_fc_lw = float(np.corrcoef(d["fc_lw_short"], d["fc_ref"])[0, 1])

        row = {
            "sub_id": d["sub_id"],
            "site": d["site"],
            "group": d["group"],
            "age": d.get("age", -1),
            "n_trs": d["n_trs"],
            "n_rois": d["n_rois"],
            "rho_hat_T": d.get("rho_hat_T", ""),
            "tertile": d.get("tertile", ""),
            "r_fc_raw_vs_ref": round(r_fc_raw, 6),
            "r_fc_lw_vs_ref": round(r_fc_lw, 6),
            "lw_improvement": round(r_fc_lw - r_fc_raw, 6),
        }
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"  Per-subject CSV: {out_path} ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Downstream FC analysis with ρ̂T-stratified quality assessment"
    )
    parser.add_argument("--dataset", default="adhd200", choices=["adhd200", "abide"])
    parser.add_argument("--filter-mode", default="strict")
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip SVM classification (requires sklearn)",
    )
    parser.add_argument(
        "--skip-fingerprint",
        action="store_true",
        help="Skip fingerprinting (O(N²), slow for large N)",
    )
    parser.add_argument(
        "--bsnet-csv",
        type=str,
        default=str(BSNET_CSV),
        help="Path to BS-NET results CSV with ρ̂T values",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    t_start = time.time()

    # ── Load subjects ──
    logger.info(f"Loading {args.dataset} subjects ({args.filter_mode} filter)...")
    subjects = load_adhd200_subjects(args.filter_mode)

    if args.max_subjects > 0:
        subjects = subjects[:args.max_subjects]
    logger.info(f"Subjects: {len(subjects)}")

    # ── Load ρ̂T values ──
    bsnet_csv_path = Path(args.bsnet_csv)
    if bsnet_csv_path.exists():
        rho_map = load_bsnet_rho(bsnet_csv_path)
    else:
        logger.warning(f"BS-NET CSV not found: {bsnet_csv_path}. Skipping stratified analysis.")
        rho_map = {}

    # ── Compute FC triplets ──
    logger.info("Computing FC matrices (raw_short, lw_short, reference)...")
    t0 = time.time()

    try:
        from tqdm import tqdm
        data = []
        for sub in tqdm(subjects, desc="FC computation", unit="sub"):
            result = compute_fc_triplet(sub)
            if result is not None:
                data.append(result)
    except ImportError:
        data = []
        for i, sub in enumerate(subjects):
            result = compute_fc_triplet(sub)
            if result is not None:
                data.append(result)
            if (i + 1) % 100 == 0:
                logger.info(f"  {i+1}/{len(subjects)} computed")

    logger.info(f"FC computed for {len(data)} subjects ({time.time()-t0:.1f}s)")

    # ── Assign tertiles ──
    if rho_map:
        logger.info("\nAssigning ρ̂T tertiles...")
        data = assign_tertiles(data, rho_map)
        logger.info(f"Subjects with ρ̂T: {len(data)}")

    # ── Run analyses ──
    all_results: dict = {
        "meta": {
            "dataset": args.dataset,
            "filter_mode": args.filter_mode,
            "n_subjects": len(data),
            "n_adhd": len([d for d in data if d["group"] == "ADHD"]),
            "n_control": len([d for d in data if d["group"] == "Control"]),
            "short_trs": SHORT_TRS,
            "has_rho_tertiles": bool(rho_map),
        }
    }

    logger.info("\n" + "=" * 60)
    logger.info("Analysis 1: Subject-level FC Similarity")
    logger.info("=" * 60)
    all_results["fc_similarity"] = analysis_subject_fc_similarity(data)

    logger.info("\n" + "=" * 60)
    logger.info("Analysis 2: Group-mean Connectome Similarity")
    logger.info("=" * 60)
    all_results["connectome_similarity"] = analysis_connectome_similarity(data)

    logger.info("\n" + "=" * 60)
    logger.info("Analysis 3: Group Contrast (Cohen's d)")
    logger.info("=" * 60)
    contrast_results = analysis_group_contrast(data)
    # Remove d_map arrays for JSON serialization
    all_results["group_contrast"] = {
        k: ({kk: vv for kk, vv in v.items() if kk != "d_map"} if isinstance(v, dict) else v)
        for k, v in contrast_results.items()
    }

    if not args.skip_classification:
        logger.info("\n" + "=" * 60)
        logger.info("Analysis 4: Classification (ADHD vs Control)")
        logger.info("=" * 60)
        try:
            all_results["classification"] = analysis_classification(data)
        except ImportError:
            logger.warning("sklearn not available — skipping classification")
    else:
        logger.info("\n(Skipping classification)")

    logger.info("\n" + "=" * 60)
    logger.info("Analysis 5: Graph Metrics")
    logger.info("=" * 60)
    all_results["graph_metrics"] = analysis_graph_metrics(data)

    # ── Analysis 6: ρ̂T-Stratified (if available) ──
    if rho_map:
        logger.info("\n" + "=" * 60)
        logger.info("Analysis 6: ρ̂T-Stratified Downstream Quality (Dose-Response)")
        logger.info("=" * 60)
        all_results["rho_stratified"] = analysis_rho_stratified(
            data, skip_classification=args.skip_classification
        )

    # ── Analysis 7: Fingerprinting ──
    if not args.skip_fingerprint:
        logger.info("\n" + "=" * 60)
        logger.info("Analysis 7: FC Fingerprinting (Test-Retest Identification)")
        logger.info("=" * 60)
        if len(data) > 200:
            logger.info(f"  (N={len(data)}, O(N²) — this may take a few minutes)")
        all_results["fingerprinting"] = analysis_fingerprinting(data)
    else:
        logger.info("\n(Skipping fingerprinting)")

    # ── Save results ──
    out_dir = Path("data/adhd/pcp/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "adhd200_downstream_analysis.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Per-subject CSV for plotting
    csv_path = out_dir / "adhd200_downstream_per_subject.csv"
    export_per_subject_csv(data, csv_path)

    elapsed = time.time() - t_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"All results saved to: {json_path}")
    logger.info(f"Per-subject CSV: {csv_path}")
    logger.info(f"Total elapsed: {elapsed:.1f}s")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
