#!/usr/bin/env python3
"""Reliability-aware unsupervised patient separation on ADHD-200 PCP.

This script quantifies how FC-based clustering aligns with ADHD/Control labels
across rho-hat-T quality strata (T1/T2/T3), using nilearn connectome methods.

Primary target:
  - KMeans(k=2) performance trend across rho strata

Secondary targets:
  - FC method comparison (correlation / partial correlation / tangent)
  - Algorithm comparison (KMeans / GMM / Spectral)

Outputs:
  - data/adhd/pcp/results/adhd200_reliability_clustering_runs.csv
  - data/adhd/pcp/results/adhd200_reliability_clustering_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SHORT_TRS = 60

SUBJECTS_JSON = Path("data/adhd/pcp/results/adhd200_subjects_cc200.json")
RHO_CSV = Path("data/adhd/pcp/results/adhd200_multiseed_cc200_10seeds_filtered_strict.csv")
OUT_DIR = Path("data/adhd/pcp/results")
RUNS_CSV = OUT_DIR / "adhd200_reliability_clustering_runs.csv"
SUMMARY_CSV = OUT_DIR / "adhd200_reliability_clustering_summary.csv"

FC_METHODS = ("correlation", "partial correlation", "tangent")
ALGORITHMS = ("kmeans", "gmm", "spectral")


def _load_subjects_strict() -> list[dict]:
    """Load ADHD-200 subjects and apply strict filter."""
    with open(SUBJECTS_JSON) as f:
        subjects = json.load(f)

    known = [s for s in subjects if s.get("group") in ("ADHD", "Control")]
    for s in known:
        n_trs = int(s["n_trs"])
        tr = float(s["tr"])
        s["k"] = n_trs / SHORT_TRS
        s["ref_s"] = (n_trs - SHORT_TRS) * tr

    strict = [s for s in known if s["k"] >= 2.0 and s["ref_s"] >= 300]
    strict = [s for s in strict if Path(s["ts_path"]).exists()]
    logger.info(f"Subjects strict-filtered: {len(strict)}")
    return strict


def _load_rho_map() -> dict[str, float]:
    """Load per-subject rho_hat_T mean."""
    rho_map: dict[str, float] = {}
    with open(RHO_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rho_map[row["sub_id"]] = float(row["rho_hat_T_mean"])
    logger.info(f"Loaded rho_hat_T for {len(rho_map)} subjects")
    return rho_map


def _attach_rho_tertile(subjects: list[dict], rho_map: dict[str, float]) -> list[dict]:
    """Attach rho value and tertile label."""
    valid: list[dict] = []
    for s in subjects:
        rho = rho_map.get(s["sub_id"])
        if rho is None:
            continue
        s2 = dict(s)
        s2["rho_hat_T"] = rho
        valid.append(s2)

    if not valid:
        return valid

    rho_vals = np.array([s["rho_hat_T"] for s in valid], dtype=float)
    q1 = float(np.percentile(rho_vals, 33.33))
    q2 = float(np.percentile(rho_vals, 66.67))

    for s in valid:
        rho = s["rho_hat_T"]
        if rho <= q1:
            s["stratum"] = "T1_low"
        elif rho <= q2:
            s["stratum"] = "T2_mid"
        else:
            s["stratum"] = "T3_high"

    return valid


def _labels_from_subjects(subjects: list[dict]) -> np.ndarray:
    """ADHD=1, Control=0."""
    return np.array([1 if s["group"] == "ADHD" else 0 for s in subjects], dtype=int)


def _best_flip_bal_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy with best binary flip mapping."""
    a = balanced_accuracy_score(y_true, y_pred)
    b = balanced_accuracy_score(y_true, 1 - y_pred)
    return float(max(a, b))


def _build_features(
    timeseries_list: list[np.ndarray],
    fc_method: str,
    pca_var: float,
) -> np.ndarray:
    """Build connectome features with nilearn + scaling + PCA."""
    cm = ConnectivityMeasure(
        kind=fc_method,
        vectorize=True,
        discard_diagonal=True,
        standardize="zscore_sample",
    )
    x = cm.fit_transform(timeseries_list)

    x = StandardScaler().fit_transform(x)

    n_samples = x.shape[0]
    if n_samples >= 4:
        pca = PCA(n_components=pca_var, random_state=42)
        x = pca.fit_transform(x)

    return x


def _cluster_predict(x: np.ndarray, algo: str, seed: int) -> np.ndarray:
    """Run one clustering algorithm."""
    if algo == "kmeans":
        model = KMeans(n_clusters=2, n_init=50, random_state=seed)
        return model.fit_predict(x)

    if algo == "gmm":
        model = GaussianMixture(
            n_components=2,
            covariance_type="diag",
            n_init=5,
            random_state=seed,
        )
        return model.fit_predict(x)

    if algo == "spectral":
        n_neighbors = max(2, min(10, x.shape[0] - 1))
        model = SpectralClustering(
            n_clusters=2,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=seed,
        )
        return model.fit_predict(x)

    raise ValueError(f"Unknown algo: {algo}")


def _evaluate(
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute clustering evaluation metrics."""
    ari = float(adjusted_rand_score(y, y_pred))
    nmi = float(normalized_mutual_info_score(y, y_pred))
    bal_acc = _best_flip_bal_acc(y, y_pred)

    if len(np.unique(y_pred)) < 2:
        sil = float("nan")
    else:
        sil = float(silhouette_score(x, y_pred))

    return {
        "ari": ari,
        "nmi": nmi,
        "balanced_acc": bal_acc,
        "silhouette": sil,
    }


def _balanced_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return class-balanced indices via random downsampling."""
    idx_adhd = np.where(y == 1)[0]
    idx_ctrl = np.where(y == 0)[0]
    n_keep = min(len(idx_adhd), len(idx_ctrl))
    pick_adhd = rng.choice(idx_adhd, size=n_keep, replace=False)
    pick_ctrl = rng.choice(idx_ctrl, size=n_keep, replace=False)
    idx = np.concatenate([pick_adhd, pick_ctrl])
    idx.sort()
    return idx


def _permutation_p_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """One-sided permutation p-values for ARI and balanced accuracy."""
    obs_ari = float(adjusted_rand_score(y_true, y_pred))
    obs_bal = _best_flip_bal_acc(y_true, y_pred)

    null_ari = np.empty(n_perm, dtype=float)
    null_bal = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        y_perm = rng.permutation(y_true)
        null_ari[i] = adjusted_rand_score(y_perm, y_pred)
        null_bal[i] = _best_flip_bal_acc(y_perm, y_pred)

    ari_p = float((1 + np.sum(null_ari >= obs_ari)) / (n_perm + 1))
    bal_p = float((1 + np.sum(null_bal >= obs_bal)) / (n_perm + 1))
    return {"ari_p_perm": ari_p, "bal_acc_p_perm": bal_p}


def _run_one_repeat(
    *,
    stratum_name: str,
    stratum_subs: list[dict],
    short_trs: int,
    rep: int,
    random_seed: int,
    pca_var: float,
    balance_classes: bool,
    n_permutations: int,
    permute_primary_only: bool,
) -> dict:
    """Run one repeat for one stratum (worker-safe)."""
    n = len(stratum_subs)
    n_adhd = sum(1 for s in stratum_subs if s["group"] == "ADHD")
    n_ctrl = n - n_adhd
    rho_mean = float(np.mean([s["rho_hat_T"] for s in stratum_subs]))
    rho_std = float(np.std([s["rho_hat_T"] for s in stratum_subs]))

    rep_seed = random_seed + rep
    rep_rng = np.random.default_rng(rep_seed)

    ts_all = [np.load(s["ts_path"])[:short_trs] for s in stratum_subs]
    y_all = _labels_from_subjects(stratum_subs)

    if balance_classes:
        idx = _balanced_indices(y_all, rep_rng)
    else:
        idx = np.arange(len(y_all))

    ts_rep = [ts_all[i] for i in idx]
    y_rep = y_all[idx]

    n_rep = len(y_rep)
    n_rep_adhd = int(np.sum(y_rep == 1))
    n_rep_ctrl = n_rep - n_rep_adhd

    rows: list[dict] = []
    feature_dims: dict[str, int] = {}
    for fc_method in FC_METHODS:
        x = _build_features(ts_rep, fc_method=fc_method, pca_var=pca_var)
        feature_dims[fc_method] = int(x.shape[1])

        for algo in ALGORITHMS:
            y_pred = _cluster_predict(x, algo=algo, seed=rep_seed)
            metrics = _evaluate(x, y_rep, y_pred)

            pvals: dict[str, float] = {}
            run_perm = n_permutations > 0 and (
                (not permute_primary_only) or (algo == "kmeans")
            )
            if run_perm:
                perm_rng = np.random.default_rng(rep_seed + 100_000)
                pvals = _permutation_p_values(
                    y_true=y_rep,
                    y_pred=y_pred,
                    n_perm=n_permutations,
                    rng=perm_rng,
                )

            rows.append({
                "stratum": stratum_name,
                "n_subjects": n,
                "n_adhd": n_adhd,
                "n_control": n_ctrl,
                "n_subjects_analysis": n_rep,
                "n_adhd_analysis": n_rep_adhd,
                "n_control_analysis": n_rep_ctrl,
                "balance_classes": balance_classes,
                "rho_mean": rho_mean,
                "rho_std": rho_std,
                "fc_method": fc_method,
                "algorithm": algo,
                "repeat": rep,
                **metrics,
                **pvals,
            })

    return {
        "rep": rep,
        "feature_dims": feature_dims,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reliability-aware clustering (nilearn FC + rho strata)",
    )
    parser.add_argument("--short-trs", type=int, default=SHORT_TRS)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--pca-var",
        type=float,
        default=0.90,
        help="PCA variance ratio for dimensionality reduction (default: 0.90).",
    )
    parser.add_argument(
        "--min-subjects",
        type=int,
        default=40,
        help="Minimum subjects per stratum to run clustering.",
    )
    parser.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repeat-wise class-balance correction via downsampling (default: true).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=200,
        help="Permutation count for p-values (0 to disable).",
    )
    parser.add_argument(
        "--permute-primary-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run permutation only for KMeans if true (default: true).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of worker processes for repeat-level parallelism (default: 1).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.short_trs != SHORT_TRS:
        logger.warning(
            f"short_trs={args.short_trs} differs from default {SHORT_TRS}; "
            "this script currently slices first short_trs from cached timeseries.",
        )

    subs = _load_subjects_strict()
    rho_map = _load_rho_map()
    subs = _attach_rho_tertile(subs, rho_map)
    logger.info(f"Subjects with rho_hat_T: {len(subs)}")
    logger.info(
        "Options: "
        f"balance_classes={args.balance_classes}, "
        f"n_permutations={args.n_permutations}, "
        f"permute_primary_only={args.permute_primary_only}, "
        f"n_jobs={args.n_jobs}",
    )

    # strata: all + tertiles
    strata = {
        "all": subs,
        "T1_low": [s for s in subs if s["stratum"] == "T1_low"],
        "T2_mid": [s for s in subs if s["stratum"] == "T2_mid"],
        "T3_high": [s for s in subs if s["stratum"] == "T3_high"],
    }

    runs: list[dict] = []

    for stratum_name, stratum_subs in strata.items():
        n = len(stratum_subs)
        n_adhd = sum(1 for s in stratum_subs if s["group"] == "ADHD")
        n_ctrl = n - n_adhd
        logger.info(f"\n[{stratum_name}] N={n} (ADHD={n_adhd}, Control={n_ctrl})")

        if n < args.min_subjects or min(n_adhd, n_ctrl) < 10:
            logger.warning(f"Skipping {stratum_name}: insufficient samples.")
            continue

        if args.n_jobs <= 1:
            for rep in range(args.n_repeats):
                try:
                    result = _run_one_repeat(
                        stratum_name=stratum_name,
                        stratum_subs=stratum_subs,
                        short_trs=args.short_trs,
                        rep=rep,
                        random_seed=args.random_seed,
                        pca_var=args.pca_var,
                        balance_classes=args.balance_classes,
                        n_permutations=args.n_permutations,
                        permute_primary_only=args.permute_primary_only,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed: stratum={stratum_name}, rep={rep}, err={e}",
                    )
                    continue
                if rep == 0:
                    for fc_method in FC_METHODS:
                        logger.info(
                            f"  FC={fc_method:18s} features={result['feature_dims'][fc_method]}",
                        )
                runs.extend(result["rows"])
        else:
            with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
                futs = {
                    ex.submit(
                        _run_one_repeat,
                        stratum_name=stratum_name,
                        stratum_subs=stratum_subs,
                        short_trs=args.short_trs,
                        rep=rep,
                        random_seed=args.random_seed,
                        pca_var=args.pca_var,
                        balance_classes=args.balance_classes,
                        n_permutations=args.n_permutations,
                        permute_primary_only=args.permute_primary_only,
                    ): rep
                    for rep in range(args.n_repeats)
                }
                got_first = False
                for fut in as_completed(futs):
                    rep = futs[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        logger.warning(
                            f"Failed: stratum={stratum_name}, rep={rep}, err={e}",
                        )
                        continue
                    if not got_first:
                        for fc_method in FC_METHODS:
                            logger.info(
                                f"  FC={fc_method:18s} features={result['feature_dims'][fc_method]}",
                            )
                        got_first = True
                    runs.extend(result["rows"])

    if not runs:
        logger.error("No valid clustering runs were produced.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save per-run CSV
    run_fields = list(runs[0].keys())
    with open(RUNS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(runs)
    logger.info(f"Saved runs: {RUNS_CSV}")

    # Aggregate summary
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for r in runs:
        key = (r["stratum"], r["fc_method"], r["algorithm"])
        grouped.setdefault(key, []).append(r)

    summary_rows: list[dict] = []
    for (stratum, fc_method, algorithm), rows in grouped.items():
        row0 = rows[0]
        ari_p = [r["ari_p_perm"] for r in rows if "ari_p_perm" in r]
        bal_p = [r["bal_acc_p_perm"] for r in rows if "bal_acc_p_perm" in r]
        summary_rows.append({
            "stratum": stratum,
            "fc_method": fc_method,
            "algorithm": algorithm,
            "n_subjects": row0["n_subjects"],
            "n_adhd": row0["n_adhd"],
            "n_control": row0["n_control"],
            "n_subjects_analysis_mean": float(np.mean([r["n_subjects_analysis"] for r in rows])),
            "n_adhd_analysis_mean": float(np.mean([r["n_adhd_analysis"] for r in rows])),
            "n_control_analysis_mean": float(np.mean([r["n_control_analysis"] for r in rows])),
            "balance_classes": row0["balance_classes"],
            "rho_mean": row0["rho_mean"],
            "rho_std": row0["rho_std"],
            "n_runs": len(rows),
            "ari_mean": float(np.mean([r["ari"] for r in rows])),
            "ari_std": float(np.std([r["ari"] for r in rows])),
            "nmi_mean": float(np.mean([r["nmi"] for r in rows])),
            "nmi_std": float(np.std([r["nmi"] for r in rows])),
            "bal_acc_mean": float(np.mean([r["balanced_acc"] for r in rows])),
            "bal_acc_std": float(np.std([r["balanced_acc"] for r in rows])),
            "silhouette_mean": float(np.nanmean([r["silhouette"] for r in rows])),
            "silhouette_std": float(np.nanstd([r["silhouette"] for r in rows])),
            "ari_p_perm_mean": float(np.mean(ari_p)) if ari_p else float("nan"),
            "ari_sig_rate_p05": float(np.mean(np.array(ari_p) < 0.05)) if ari_p else float("nan"),
            "bal_acc_p_perm_mean": float(np.mean(bal_p)) if bal_p else float("nan"),
            "bal_acc_sig_rate_p05": float(np.mean(np.array(bal_p) < 0.05)) if bal_p else float("nan"),
        })

    summary_rows = sorted(
        summary_rows,
        key=lambda r: (r["stratum"], r["algorithm"], r["fc_method"]),
    )

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"Saved summary: {SUMMARY_CSV}")

    # Console short report (primary: KMeans)
    logger.info("\nPrimary report (KMeans):")
    for stratum in ("all", "T1_low", "T2_mid", "T3_high"):
        subset = [r for r in summary_rows if r["stratum"] == stratum and r["algorithm"] == "kmeans"]
        if not subset:
            continue
        best = max(subset, key=lambda r: r["ari_mean"])
        ptxt = ""
        if not np.isnan(best["ari_p_perm_mean"]):
            ptxt = (
                f" pARI={best['ari_p_perm_mean']:.3f}"
                f" (sig<.05={best['ari_sig_rate_p05']:.2f})"
            )
        logger.info(
            f"  {stratum:7s} | best FC={best['fc_method']:18s} "
            f"ARI={best['ari_mean']:.3f}±{best['ari_std']:.3f} "
            f"BalAcc={best['bal_acc_mean']:.3f}±{best['bal_acc_std']:.3f}"
            f"{ptxt}",
        )


if __name__ == "__main__":
    main()
