#!/usr/bin/env python3
"""Reliability-aware ADHD vs Control classification on ADHD-200 PCP.

Primary design (citation-aligned):
  - FC representations: correlation / partial correlation / tangent
  - Linear classifiers: logistic_l2 / linear_svm
  - Site-generalization evaluation: LOSO (Leave-One-Site-Out)
  - Optional rho-hat-T sample weighting
  - Permutation test (primary-only by default)

Outputs:
  - data/adhd/pcp/results/adhd200_reliability_classification_runs.csv
  - data/adhd/pcp/results/adhd200_reliability_classification_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

SHORT_TRS = 60

SUBJECTS_JSON = Path("data/adhd/pcp/results/adhd200_subjects_cc200.json")
RHO_CSV = Path("data/adhd/pcp/results/adhd200_multiseed_cc200_10seeds_filtered_strict.csv")
OUT_DIR = Path("data/adhd/pcp/results")
RUNS_CSV = OUT_DIR / "adhd200_reliability_classification_runs.csv"
SUMMARY_CSV = OUT_DIR / "adhd200_reliability_classification_summary.csv"

FC_METHODS = ("correlation", "partial correlation", "tangent")
MODELS = ("logistic_l2", "linear_svm")

BLAS_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


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


def _prepare_repeat_subjects(
    stratum_subs: list[dict],
    balance_classes: bool,
    rng: np.random.Generator,
) -> list[dict]:
    """Prepare repeat-level subject subset."""
    if not balance_classes:
        return stratum_subs

    y = _labels_from_subjects(stratum_subs)
    idx = _balanced_indices(y, rng)
    return [stratum_subs[i] for i in idx]


def _load_timeseries(subs: list[dict], short_trs: int) -> list[np.ndarray]:
    """Load sliced timeseries for selected subjects."""
    return [np.load(s["ts_path"])[:short_trs].astype(np.float64) for s in subs]


def _build_covariates(
    train_subs: list[dict],
    test_subs: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Build fold-safe covariates: age(z), sex(binary), site(one-hot)."""
    train_age = np.array([float(s.get("age", np.nan)) for s in train_subs], dtype=float)
    test_age = np.array([float(s.get("age", np.nan)) for s in test_subs], dtype=float)

    age_mean = np.nanmean(train_age) if np.isfinite(np.nanmean(train_age)) else 0.0
    age_std = np.nanstd(train_age)
    if not np.isfinite(age_std) or age_std < 1e-8:
        age_std = 1.0
    train_age_z = ((np.where(np.isfinite(train_age), train_age, age_mean) - age_mean) / age_std)[:, None]
    test_age_z = ((np.where(np.isfinite(test_age), test_age, age_mean) - age_mean) / age_std)[:, None]

    train_sex = np.array([1.0 if str(s.get("sex", "")).upper() == "M" else 0.0 for s in train_subs])[:, None]
    test_sex = np.array([1.0 if str(s.get("sex", "")).upper() == "M" else 0.0 for s in test_subs])[:, None]

    train_sites = [str(s.get("site", "UNK")) for s in train_subs]
    test_sites = [str(s.get("site", "UNK")) for s in test_subs]
    uniq_sites = sorted(set(train_sites))
    site_to_col = {site: i for i, site in enumerate(uniq_sites)}

    train_site_oh = np.zeros((len(train_subs), len(uniq_sites)), dtype=float)
    test_site_oh = np.zeros((len(test_subs), len(uniq_sites)), dtype=float)
    for i, site in enumerate(train_sites):
        train_site_oh[i, site_to_col[site]] = 1.0
    for i, site in enumerate(test_sites):
        j = site_to_col.get(site)
        if j is not None:
            test_site_oh[i, j] = 1.0

    x_train_cov = np.hstack([train_age_z, train_sex, train_site_oh])
    x_test_cov = np.hstack([test_age_z, test_sex, test_site_oh])
    return x_train_cov, x_test_cov


def _build_fc_train_test(
    train_ts: list[np.ndarray],
    test_ts: list[np.ndarray],
    fc_method: str,
    pca_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fold-safe FC features (train fit -> train/test transform)."""
    cm = ConnectivityMeasure(
        kind=fc_method,
        vectorize=True,
        discard_diagonal=True,
        standardize="zscore_sample",
    )
    x_train = cm.fit_transform(train_ts)
    x_test = cm.transform(test_ts)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    if x_train.shape[0] >= 4 and x_train.shape[1] >= 4 and pca_var < 1.0:
        pca = PCA(n_components=pca_var, random_state=42)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    return x_train, x_test


def _make_splits(
    y: np.ndarray,
    groups: np.ndarray,
    eval_scheme: str,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate split indices."""
    if eval_scheme == "loso":
        logo = LeaveOneGroupOut()
        return list(logo.split(np.zeros_like(y), y, groups))

    if eval_scheme == "stratified_kfold":
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros_like(y), y))

    raise ValueError(f"Unknown eval_scheme: {eval_scheme}")


def _fit_predict_one_split(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    seed: int,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit model and return predicted labels + decision scores."""
    if model_name == "logistic_l2":
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
    elif model_name == "linear_svm":
        clf = SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced",
            probability=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if sample_weight is not None:
        clf.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test).astype(int)

    if hasattr(clf, "decision_function"):
        score = clf.decision_function(x_test)
    elif hasattr(clf, "predict_proba"):
        score = clf.predict_proba(x_test)[:, 1]
    else:
        score = y_pred.astype(float)
    return y_pred, np.asarray(score, dtype=float)


def _permute_within_groups(
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute labels within site groups (for LOSO null)."""
    y_perm = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        y_perm[idx] = y_perm[rng.permutation(idx)]
    return y_perm


def _evaluate_cv(
    *,
    subs: list[dict],
    ts_list: list[np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fc_method: str,
    model_name: str,
    pca_var: float,
    include_covariates: bool,
    rho_weight_gamma: float,
    seed: int,
) -> dict[str, float]:
    """Evaluate one config with fixed folds and labels."""
    y_pred_all = np.full(len(y), -1, dtype=int)
    score_all = np.full(len(y), np.nan, dtype=float)
    feature_dims: list[int] = []
    used_splits = 0

    for split_i, (train_idx, test_idx) in enumerate(splits):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        train_ts = [ts_list[i] for i in train_idx]
        test_ts = [ts_list[i] for i in test_idx]
        x_train_fc, x_test_fc = _build_fc_train_test(
            train_ts=train_ts,
            test_ts=test_ts,
            fc_method=fc_method,
            pca_var=pca_var,
        )

        if include_covariates:
            train_subs = [subs[i] for i in train_idx]
            test_subs = [subs[i] for i in test_idx]
            x_train_cov, x_test_cov = _build_covariates(train_subs, test_subs)
            x_train = np.hstack([x_train_fc, x_train_cov])
            x_test = np.hstack([x_test_fc, x_test_cov])
        else:
            x_train = x_train_fc
            x_test = x_test_fc

        feature_dims.append(int(x_train.shape[1]))

        if rho_weight_gamma > 0:
            rho_train = np.array([float(subs[i]["rho_hat_T"]) for i in train_idx], dtype=float)
            sample_weight = np.clip(rho_train, 1e-6, None) ** rho_weight_gamma
        else:
            sample_weight = None

        y_pred, score = _fit_predict_one_split(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            model_name=model_name,
            seed=seed + split_i * 31,
            sample_weight=sample_weight,
        )

        y_pred_all[test_idx] = y_pred
        score_all[test_idx] = score
        used_splits += 1

    valid = y_pred_all >= 0
    n_eval = int(np.sum(valid))
    if n_eval == 0:
        return {
            "balanced_acc": float("nan"),
            "roc_auc": float("nan"),
            "auprc": float("nan"),
            "n_eval_subjects": 0,
            "n_splits_used": 0,
            "feature_dim": float("nan"),
        }

    y_true_v = y[valid]
    y_pred_v = y_pred_all[valid]
    score_v = score_all[valid]

    bal = float(balanced_accuracy_score(y_true_v, y_pred_v))
    try:
        auc = float(roc_auc_score(y_true_v, score_v))
    except Exception:
        auc = float("nan")
    try:
        auprc = float(average_precision_score(y_true_v, score_v))
    except Exception:
        auprc = float("nan")

    feat_dim = float(np.median(feature_dims)) if feature_dims else float("nan")
    return {
        "balanced_acc": bal,
        "roc_auc": auc,
        "auprc": auprc,
        "n_eval_subjects": n_eval,
        "n_splits_used": used_splits,
        "feature_dim": feat_dim,
    }


def _permutation_pvals(
    *,
    subs: list[dict],
    ts_list: list[np.ndarray],
    y_true: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fc_method: str,
    model_name: str,
    pca_var: float,
    include_covariates: bool,
    rho_weight_gamma: float,
    seed: int,
    n_permutations: int,
    eval_scheme: str,
) -> dict[str, float]:
    """Compute one-sided permutation p-values for BalAcc/AUC/AUPRC."""
    obs = _evaluate_cv(
        subs=subs,
        ts_list=ts_list,
        y=y_true,
        groups=groups,
        splits=splits,
        fc_method=fc_method,
        model_name=model_name,
        pca_var=pca_var,
        include_covariates=include_covariates,
        rho_weight_gamma=rho_weight_gamma,
        seed=seed,
    )
    obs_bal = obs["balanced_acc"]
    obs_auc = obs["roc_auc"]
    obs_auprc = obs["auprc"]

    rng = np.random.default_rng(seed + 100_000)
    null_bal = np.empty(n_permutations, dtype=float)
    null_auc = np.empty(n_permutations, dtype=float)
    null_auprc = np.empty(n_permutations, dtype=float)

    for i in range(n_permutations):
        if eval_scheme == "loso":
            y_perm = _permute_within_groups(y_true, groups, rng)
        else:
            y_perm = rng.permutation(y_true)

        m = _evaluate_cv(
            subs=subs,
            ts_list=ts_list,
            y=y_perm,
            groups=groups,
            splits=splits,
            fc_method=fc_method,
            model_name=model_name,
            pca_var=pca_var,
            include_covariates=include_covariates,
            rho_weight_gamma=rho_weight_gamma,
            seed=seed + i * 17,
        )
        null_bal[i] = m["balanced_acc"]
        null_auc[i] = m["roc_auc"]
        null_auprc[i] = m["auprc"]

    def p_one_sided_ge(null: np.ndarray, obs_val: float) -> float:
        valid = np.isfinite(null)
        if (not np.isfinite(obs_val)) or np.sum(valid) == 0:
            return float("nan")
        n = int(np.sum(valid))
        return float((1 + np.sum(null[valid] >= obs_val)) / (n + 1))

    return {
        "bal_acc_p_perm": p_one_sided_ge(null_bal, obs_bal),
        "roc_auc_p_perm": p_one_sided_ge(null_auc, obs_auc),
        "auprc_p_perm": p_one_sided_ge(null_auprc, obs_auprc),
    }


def _run_one_repeat(
    *,
    stratum_name: str,
    stratum_subs: list[dict],
    rep: int,
    short_trs: int,
    random_seed: int,
    pca_var: float,
    eval_scheme: str,
    n_folds: int,
    balance_classes: bool,
    include_covariates: bool,
    rho_weight_gamma: float,
    n_permutations: int,
    permute_primary_only: bool,
    primary_fc: str,
    primary_model: str,
) -> dict:
    """Worker-safe repeat run."""
    rep_seed = random_seed + rep * 1009
    rep_rng = np.random.default_rng(rep_seed)

    subs_rep = _prepare_repeat_subjects(stratum_subs, balance_classes, rep_rng)
    y = _labels_from_subjects(subs_rep)
    groups = np.array([str(s["site"]) for s in subs_rep], dtype=object)
    ts_list = _load_timeseries(subs_rep, short_trs=short_trs)

    n = len(subs_rep)
    n_adhd = int(np.sum(y == 1))
    n_ctrl = int(np.sum(y == 0))
    rho_vals = np.array([float(s["rho_hat_T"]) for s in subs_rep], dtype=float)
    splits = _make_splits(y, groups, eval_scheme=eval_scheme, n_folds=n_folds, seed=rep_seed)

    rows: list[dict] = []
    for fc_method in FC_METHODS:
        for model_name in MODELS:
            obs = _evaluate_cv(
                subs=subs_rep,
                ts_list=ts_list,
                y=y,
                groups=groups,
                splits=splits,
                fc_method=fc_method,
                model_name=model_name,
                pca_var=pca_var,
                include_covariates=include_covariates,
                rho_weight_gamma=rho_weight_gamma,
                seed=rep_seed,
            )

            pvals: dict[str, float] = {}
            run_perm = n_permutations > 0 and (
                (not permute_primary_only)
                or (fc_method == primary_fc and model_name == primary_model)
            )
            if run_perm:
                pvals = _permutation_pvals(
                    subs=subs_rep,
                    ts_list=ts_list,
                    y_true=y,
                    groups=groups,
                    splits=splits,
                    fc_method=fc_method,
                    model_name=model_name,
                    pca_var=pca_var,
                    include_covariates=include_covariates,
                    rho_weight_gamma=rho_weight_gamma,
                    seed=rep_seed,
                    n_permutations=n_permutations,
                    eval_scheme=eval_scheme,
                )

            rows.append({
                "stratum": stratum_name,
                "repeat": rep,
                "eval_scheme": eval_scheme,
                "fc_method": fc_method,
                "model": model_name,
                "n_subjects_analysis": n,
                "n_adhd_analysis": n_adhd,
                "n_control_analysis": n_ctrl,
                "n_sites_analysis": int(len(np.unique(groups))),
                "rho_mean_analysis": float(np.mean(rho_vals)),
                "rho_std_analysis": float(np.std(rho_vals)),
                "balance_classes": balance_classes,
                "include_covariates": include_covariates,
                "rho_weight_gamma": rho_weight_gamma,
                "pca_var": pca_var,
                **obs,
                **pvals,
            })

    return {"rows": rows}


def _warn_blas_oversubscription(n_jobs: int) -> None:
    """Warn if likely BLAS oversubscription in multi-process mode."""
    if n_jobs <= 1:
        return
    bad = []
    for k in BLAS_ENV_KEYS:
        v = os.environ.get(k)
        if v is None or v != "1":
            bad.append(f"{k}={v if v is not None else '<unset>'}")
    if bad:
        logger.warning(
            "n_jobs>1 detected without strict BLAS thread pinning. "
            "Set OMP/MKL/OPENBLAS/NUMEXPR threads to 1 to avoid oversubscription. "
            f"Current: {', '.join(bad)}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reliability-aware ADHD classification (LOSO/Stratified CV).",
    )
    parser.add_argument("--short-trs", type=int, default=SHORT_TRS)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--eval-scheme",
        choices=["loso", "stratified_kfold"],
        default="loso",
        help="Evaluation split scheme (default: loso).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Used only for stratified_kfold.",
    )
    parser.add_argument(
        "--pca-var",
        type=float,
        default=0.90,
        help="PCA variance ratio for FC features (default: 0.90).",
    )
    parser.add_argument(
        "--min-subjects",
        type=int,
        default=40,
        help="Minimum subjects per stratum to run.",
    )
    parser.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repeat-wise class-balance correction via downsampling (default: true).",
    )
    parser.add_argument(
        "--include-covariates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include age/sex/site covariates (default: true).",
    )
    parser.add_argument(
        "--rho-weight-gamma",
        type=float,
        default=1.0,
        help="Sample-weight exponent for rho_hat_T (<=0 disables weighting).",
    )
    parser.add_argument(
        "--primary-fc",
        choices=list(FC_METHODS),
        default="tangent",
        help="Primary FC method for permutation reporting.",
    )
    parser.add_argument(
        "--primary-model",
        choices=list(MODELS),
        default="logistic_l2",
        help="Primary model for permutation reporting.",
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
        help="Run permutation only for primary FC+model (default: true).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes for repeat-level parallelism.",
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
            "script slices first short_trs volumes from cached timeseries.",
        )

    _warn_blas_oversubscription(args.n_jobs)

    subs = _load_subjects_strict()
    rho_map = _load_rho_map()
    subs = _attach_rho_tertile(subs, rho_map)
    logger.info(f"Subjects with rho_hat_T: {len(subs)}")
    logger.info(
        "Options: "
        f"eval_scheme={args.eval_scheme}, "
        f"balance_classes={args.balance_classes}, "
        f"include_covariates={args.include_covariates}, "
        f"rho_weight_gamma={args.rho_weight_gamma:.2f}, "
        f"n_permutations={args.n_permutations}, "
        f"permute_primary_only={args.permute_primary_only}, "
        f"n_jobs={args.n_jobs}",
    )

    strata = {
        "all": subs,
        "T1_low": [s for s in subs if s["stratum"] == "T1_low"],
        "T2_mid": [s for s in subs if s["stratum"] == "T2_mid"],
        "T3_high": [s for s in subs if s["stratum"] == "T3_high"],
    }

    runs: list[dict] = []

    for stratum_name, stratum_subs in strata.items():
        n = len(stratum_subs)
        y0 = _labels_from_subjects(stratum_subs)
        n_adhd = int(np.sum(y0 == 1))
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
                        rep=rep,
                        short_trs=args.short_trs,
                        random_seed=args.random_seed,
                        pca_var=args.pca_var,
                        eval_scheme=args.eval_scheme,
                        n_folds=args.n_folds,
                        balance_classes=args.balance_classes,
                        include_covariates=args.include_covariates,
                        rho_weight_gamma=args.rho_weight_gamma,
                        n_permutations=args.n_permutations,
                        permute_primary_only=args.permute_primary_only,
                        primary_fc=args.primary_fc,
                        primary_model=args.primary_model,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed: stratum={stratum_name}, rep={rep}, err={e}",
                    )
                    continue
                runs.extend(result["rows"])
        else:
            with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
                futs = {
                    ex.submit(
                        _run_one_repeat,
                        stratum_name=stratum_name,
                        stratum_subs=stratum_subs,
                        rep=rep,
                        short_trs=args.short_trs,
                        random_seed=args.random_seed,
                        pca_var=args.pca_var,
                        eval_scheme=args.eval_scheme,
                        n_folds=args.n_folds,
                        balance_classes=args.balance_classes,
                        include_covariates=args.include_covariates,
                        rho_weight_gamma=args.rho_weight_gamma,
                        n_permutations=args.n_permutations,
                        permute_primary_only=args.permute_primary_only,
                        primary_fc=args.primary_fc,
                        primary_model=args.primary_model,
                    ): rep
                    for rep in range(args.n_repeats)
                }
                for fut in as_completed(futs):
                    rep = futs[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        logger.warning(
                            f"Failed: stratum={stratum_name}, rep={rep}, err={e}",
                        )
                        continue
                    runs.extend(result["rows"])

    if not runs:
        logger.error("No valid classification runs were produced.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_fields = list(runs[0].keys())
    with open(RUNS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(runs)
    logger.info(f"Saved runs: {RUNS_CSV}")

    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for r in runs:
        key = (r["stratum"], r["fc_method"], r["model"], r["eval_scheme"])
        grouped.setdefault(key, []).append(r)

    summary_rows: list[dict] = []
    for (stratum, fc_method, model_name, eval_scheme), rows in grouped.items():
        row0 = rows[0]
        bal_p = [r["bal_acc_p_perm"] for r in rows if "bal_acc_p_perm" in r]
        auc_p = [r["roc_auc_p_perm"] for r in rows if "roc_auc_p_perm" in r]
        pr_p = [r["auprc_p_perm"] for r in rows if "auprc_p_perm" in r]

        summary_rows.append({
            "stratum": stratum,
            "eval_scheme": eval_scheme,
            "fc_method": fc_method,
            "model": model_name,
            "n_runs": len(rows),
            "n_subjects_analysis_mean": float(np.mean([r["n_subjects_analysis"] for r in rows])),
            "n_adhd_analysis_mean": float(np.mean([r["n_adhd_analysis"] for r in rows])),
            "n_control_analysis_mean": float(np.mean([r["n_control_analysis"] for r in rows])),
            "n_sites_analysis_mean": float(np.mean([r["n_sites_analysis"] for r in rows])),
            "rho_mean_analysis_mean": float(np.mean([r["rho_mean_analysis"] for r in rows])),
            "rho_std_analysis_mean": float(np.mean([r["rho_std_analysis"] for r in rows])),
            "balance_classes": row0["balance_classes"],
            "include_covariates": row0["include_covariates"],
            "rho_weight_gamma": row0["rho_weight_gamma"],
            "pca_var": row0["pca_var"],
            "feature_dim_mean": float(np.nanmean([r["feature_dim"] for r in rows])),
            "n_eval_subjects_mean": float(np.mean([r["n_eval_subjects"] for r in rows])),
            "n_splits_used_mean": float(np.mean([r["n_splits_used"] for r in rows])),
            "bal_acc_mean": float(np.nanmean([r["balanced_acc"] for r in rows])),
            "bal_acc_std": float(np.nanstd([r["balanced_acc"] for r in rows])),
            "roc_auc_mean": float(np.nanmean([r["roc_auc"] for r in rows])),
            "roc_auc_std": float(np.nanstd([r["roc_auc"] for r in rows])),
            "auprc_mean": float(np.nanmean([r["auprc"] for r in rows])),
            "auprc_std": float(np.nanstd([r["auprc"] for r in rows])),
            "bal_acc_p_perm_mean": float(np.mean(bal_p)) if bal_p else float("nan"),
            "bal_acc_sig_rate_p05": float(np.mean(np.array(bal_p) < 0.05)) if bal_p else float("nan"),
            "roc_auc_p_perm_mean": float(np.mean(auc_p)) if auc_p else float("nan"),
            "roc_auc_sig_rate_p05": float(np.mean(np.array(auc_p) < 0.05)) if auc_p else float("nan"),
            "auprc_p_perm_mean": float(np.mean(pr_p)) if pr_p else float("nan"),
            "auprc_sig_rate_p05": float(np.mean(np.array(pr_p) < 0.05)) if pr_p else float("nan"),
        })

    summary_rows = sorted(
        summary_rows,
        key=lambda r: (r["stratum"], r["eval_scheme"], r["model"], r["fc_method"]),
    )

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"Saved summary: {SUMMARY_CSV}")

    logger.info("\nPrimary report:")
    for stratum in ("all", "T1_low", "T2_mid", "T3_high"):
        subset = [
            r for r in summary_rows
            if r["stratum"] == stratum
            and r["fc_method"] == args.primary_fc
            and r["model"] == args.primary_model
            and r["eval_scheme"] == args.eval_scheme
        ]
        if not subset:
            continue
        r = subset[0]
        ptxt = ""
        if not np.isnan(r["bal_acc_p_perm_mean"]):
            ptxt = (
                f" pBalAcc={r['bal_acc_p_perm_mean']:.3f}"
                f" (sig<.05={r['bal_acc_sig_rate_p05']:.2f})"
            )
        logger.info(
            f"  {stratum:7s} | FC={r['fc_method']:18s} model={r['model']:11s} "
            f"BalAcc={r['bal_acc_mean']:.3f}±{r['bal_acc_std']:.3f} "
            f"AUC={r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.3f}"
            f"{ptxt}",
        )


if __name__ == "__main__":
    main()
