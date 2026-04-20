#!/usr/bin/env python3
"""Reliability-gated BP vs SZ classification on Keane BS-NET features.

Scientific design implemented in this script:
1) Confirmatory family (single pre-specified test):
   - feature=primary_feature (default: fc_bsnet_pred)
   - model=primary_model (default: logistic_l2)
   - gate_mode=hard, rho_quantile=primary_gate_quantile (default: 0.4)
   - endpoint: balanced accuracy (BalAcc)
   - permutation p-value with optional Holm correction in family

2) Exploratory family:
   - baseline (gate_mode=none)
   - hard-gating grid (rho quantiles)
   - soft-weight grid (rho gammas)
   - optional all feature/model combinations
   - permutation p-values with BH-FDR correction in family

Leakage control:
- For hard gating, threshold is estimated from train-fold rho only.
- The same threshold is applied to train and test in that fold.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

FEATURES_NPZ = Path("data/keane/results/keane_bsnet_features.npz")
RHO_CSV = Path("data/keane/results/keane_bsnet_recomputed.csv")
OUT_DIR = Path("data/keane/results")


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment."""
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    adj_sorted = np.empty(m, dtype=float)
    for i, p in enumerate(p_sorted):
        adj_sorted[i] = min(1.0, p * (m - i))
    # enforce monotonicity
    for i in range(1, m):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])
    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return out


def _bh_fdr_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    q_sorted = np.empty(m, dtype=float)
    for i, p in enumerate(p_sorted, start=1):
        q_sorted[i - 1] = p * m / i
    # enforce monotonicity from end
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = q_sorted
    return out


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    model_name: str,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    else:
        score = y_pred.astype(float)
    return y_pred, np.asarray(score, dtype=float)


def _is_valid_class_counts(y: np.ndarray, min_per_class: int) -> bool:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return False
    return bool(np.all(counts >= min_per_class))


def _evaluate_once(
    x: np.ndarray,
    y: np.ndarray,
    rho: np.ndarray,
    model_name: str,
    n_folds: int,
    seed: int,
    gate_mode: str,
    rho_quantile: float,
    rho_gamma: float,
    min_class_count_train: int,
    min_class_count_test: int,
) -> dict[str, float]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_pred_all = np.full(len(y), -1, dtype=int)
    score_all = np.full(len(y), np.nan, dtype=float)
    eval_mask = np.zeros(len(y), dtype=bool)
    n_splits_used = 0

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        train_sel = np.ones(len(train_idx), dtype=bool)
        test_sel = np.ones(len(test_idx), dtype=bool)

        if gate_mode == "hard":
            thr = float(np.nanquantile(rho[train_idx], rho_quantile))
            if not np.isfinite(thr):
                continue
            train_sel = np.isfinite(rho[train_idx]) & (rho[train_idx] >= thr)
            test_sel = np.isfinite(rho[test_idx]) & (rho[test_idx] >= thr)
        elif gate_mode in ("none", "soft"):
            pass
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode}")

        tr_idx = train_idx[train_sel]
        te_idx = test_idx[test_sel]
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue

        y_train = y[tr_idx]
        y_test = y[te_idx]
        if not _is_valid_class_counts(y_train, min_per_class=min_class_count_train):
            continue
        if not _is_valid_class_counts(y_test, min_per_class=min_class_count_test):
            continue

        x_train = x[tr_idx]
        x_test = x[te_idx]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        sample_weight = None
        if gate_mode == "soft" and rho_gamma > 0:
            w = np.power(np.clip(rho[tr_idx], 1e-8, None), rho_gamma)
            w = w / np.mean(w)
            sample_weight = w.astype(float)

        y_pred, score = _fit_predict(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            model_name=model_name,
            seed=seed + fold_i * 17,
            sample_weight=sample_weight,
        )
        y_pred_all[te_idx] = y_pred
        score_all[te_idx] = score
        eval_mask[te_idx] = True
        n_splits_used += 1

    n_eval = int(np.sum(eval_mask))
    coverage = float(n_eval / len(y)) if len(y) > 0 else float("nan")
    if n_eval <= 1 or len(np.unique(y[eval_mask])) < 2:
        return {
            "bal_acc": float("nan"),
            "roc_auc": float("nan"),
            "auprc": float("nan"),
            "n_splits_used": float(n_splits_used),
            "n_eval_subjects": float(n_eval),
            "coverage": coverage,
        }

    y_eval = y[eval_mask]
    yp_eval = y_pred_all[eval_mask]
    sc_eval = score_all[eval_mask]
    bal = float(balanced_accuracy_score(y_eval, yp_eval))
    try:
        auc = float(roc_auc_score(y_eval, sc_eval))
    except Exception:
        auc = float("nan")
    try:
        auprc = float(average_precision_score(y_eval, sc_eval))
    except Exception:
        auprc = float("nan")
    return {
        "bal_acc": bal,
        "roc_auc": auc,
        "auprc": auprc,
        "n_splits_used": float(n_splits_used),
        "n_eval_subjects": float(n_eval),
        "coverage": coverage,
    }


def _perm_pvals(
    x: np.ndarray,
    y: np.ndarray,
    rho: np.ndarray,
    model_name: str,
    n_folds: int,
    seed: int,
    gate_mode: str,
    rho_quantile: float,
    rho_gamma: float,
    min_class_count_train: int,
    min_class_count_test: int,
    n_permutations: int,
    progress_label: str = "",
) -> dict[str, float]:
    obs = _evaluate_once(
        x=x,
        y=y,
        rho=rho,
        model_name=model_name,
        n_folds=n_folds,
        seed=seed,
        gate_mode=gate_mode,
        rho_quantile=rho_quantile,
        rho_gamma=rho_gamma,
        min_class_count_train=min_class_count_train,
        min_class_count_test=min_class_count_test,
    )
    obs_bal = obs["bal_acc"]
    obs_auc = obs["roc_auc"]
    obs_pr = obs["auprc"]

    rng = np.random.default_rng(seed + 100_000)
    null_bal = np.empty(n_permutations, dtype=float)
    null_auc = np.empty(n_permutations, dtype=float)
    null_pr = np.empty(n_permutations, dtype=float)

    if tqdm is not None:
        perm_iter = tqdm(
            range(n_permutations),
            desc=progress_label or "permutations",
            unit="perm",
            leave=False,
        )
    else:
        logger.info("%s permutations: %d", progress_label or "Running", n_permutations)
        perm_iter = range(n_permutations)

    for i in perm_iter:
        y_perm = rng.permutation(y)
        m = _evaluate_once(
            x=x,
            y=y_perm,
            rho=rho,
            model_name=model_name,
            n_folds=n_folds,
            seed=seed + i * 19,
            gate_mode=gate_mode,
            rho_quantile=rho_quantile,
            rho_gamma=rho_gamma,
            min_class_count_train=min_class_count_train,
            min_class_count_test=min_class_count_test,
        )
        null_bal[i] = m["bal_acc"]
        null_auc[i] = m["roc_auc"]
        null_pr[i] = m["auprc"]

    def _p(null: np.ndarray, obs_v: float) -> float:
        valid = np.isfinite(null)
        if (not np.isfinite(obs_v)) or np.sum(valid) == 0:
            return float("nan")
        n = int(np.sum(valid))
        return float((1 + np.sum(null[valid] >= obs_v)) / (n + 1))

    return {
        "bal_acc_p_perm": _p(null_bal, obs_bal),
        "roc_auc_p_perm": _p(null_auc, obs_auc),
        "auprc_p_perm": _p(null_pr, obs_pr),
    }


def _build_settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    def _add(s: dict[str, Any]) -> None:
        key = (
            s["family"],
            s["feature"],
            s["model"],
            s["gate_mode"],
            s["rho_quantile"],
            s["rho_gamma"],
        )
        if key in seen:
            return
        seen.add(key)
        settings.append(s)

    # Confirmatory (single pre-specified setting)
    _add(
        {
            "family": "confirmatory",
            "is_primary": 1,
            "feature": args.primary_feature,
            "model": args.primary_model,
            "gate_mode": "hard",
            "rho_quantile": float(args.primary_gate_quantile),
            "rho_gamma": 0.0,
        },
    )

    if args.exploratory_scope == "all":
        combos = [(f, m) for f in args.feature_names for m in args.model_names]
    else:
        combos = [(args.primary_feature, args.primary_model)]

    for feat, model_name in combos:
        _add(
            {
                "family": "exploratory",
                "is_primary": 0,
                "feature": feat,
                "model": model_name,
                "gate_mode": "none",
                "rho_quantile": float("nan"),
                "rho_gamma": 0.0,
            },
        )

        for q in args.rho_quantiles:
            _add(
                {
                    "family": "exploratory",
                    "is_primary": 0,
                    "feature": feat,
                    "model": model_name,
                    "gate_mode": "hard",
                    "rho_quantile": float(q),
                    "rho_gamma": 0.0,
                },
            )

        for g in args.rho_gammas:
            _add(
                {
                    "family": "exploratory",
                    "is_primary": 0,
                    "feature": feat,
                    "model": model_name,
                    "gate_mode": "soft",
                    "rho_quantile": float("nan"),
                    "rho_gamma": float(g),
                },
            )
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reliability-gated BP vs SZ classification on Keane BS-NET features.",
    )
    parser.add_argument("--features-npz", type=Path, default=FEATURES_NPZ)
    parser.add_argument("--rho-csv", type=Path, default=RHO_CSV)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument(
        "--feature-names",
        nargs="+",
        default=["fc_raw_short", "fc_bsnet_pred", "fc_reference"],
    )
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=["logistic_l2", "linear_svm"],
    )
    parser.add_argument("--primary-feature", type=str, default="fc_bsnet_pred")
    parser.add_argument("--primary-model", type=str, default="logistic_l2")
    parser.add_argument("--primary-gate-quantile", type=float, default=0.4)
    parser.add_argument("--rho-quantiles", nargs="+", type=float, default=[0.3, 0.4, 0.5])
    parser.add_argument("--rho-gammas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--min-class-count-train", type=int, default=5)
    parser.add_argument("--min-class-count-test", type=int, default=1)
    parser.add_argument(
        "--min-valid-splits",
        type=int,
        default=2,
        help="Minimum valid CV splits required per repeat; below this, setting is marked invalid and permutation is skipped.",
    )
    parser.add_argument(
        "--exploratory-scope",
        type=str,
        choices=["primary", "all"],
        default="primary",
    )
    parser.add_argument("--permute-primary-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.features_npz.exists():
        raise FileNotFoundError(f"Missing features NPZ: {args.features_npz}")
    if not args.rho_csv.exists():
        raise FileNotFoundError(f"Missing rho CSV: {args.rho_csv}")

    npz = np.load(args.features_npz, allow_pickle=True)
    sub_ids = np.asarray(npz["sub_ids"], dtype=object).astype(str)
    labels3 = np.asarray(npz["labels_threeclass"], dtype=int)
    mask_bp_sz = labels3 != 0
    y_all = np.array([0 if c == 1 else 1 for c in labels3[mask_bp_sz]], dtype=int)  # BP=0, SZ=1
    sub_bp_sz = sub_ids[mask_bp_sz]

    rho_map: dict[str, float] = {}
    with open(args.rho_csv) as f:
        for row in csv.DictReader(f):
            sid = row.get("sub_id", "")
            if sid:
                rho_map[sid] = float(row["rho_hat_T"])
    missing_rho = [sid for sid in sub_bp_sz if sid not in rho_map]
    if missing_rho:
        raise RuntimeError(f"Missing rho_hat_T for {len(missing_rho)} BP/SZ subjects")
    rho_all = np.array([rho_map[sid] for sid in sub_bp_sz], dtype=float)
    finite_mask_bp = np.isfinite(rho_all)
    if not np.all(finite_mask_bp):
        n_drop = int(np.sum(~finite_mask_bp))
        logger.warning("Dropping %d subjects with non-finite rho_hat_T", n_drop)
        y_all = y_all[finite_mask_bp]
        rho_all = rho_all[finite_mask_bp]
        sub_bp_sz = sub_bp_sz[finite_mask_bp]

    settings = _build_settings(args)
    logger.info(
        "Settings: confirmatory=1, exploratory=%d, repeats=%d, perm=%d",
        sum(1 for s in settings if s["family"] == "exploratory"),
        args.n_repeats,
        args.n_permutations,
    )

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    runs_csv = out_dir / f"keane_bsnet_bp_sz_gated_runs{tag}.csv"
    summary_csv = out_dir / f"keane_bsnet_bp_sz_gated_summary{tag}.csv"

    jobs: list[tuple[int, dict[str, Any], int]] = []
    for i, s in enumerate(settings):
        for rep in range(args.n_repeats):
            jobs.append((i, s, rep))

    if tqdm is not None:
        job_iter = tqdm(jobs, desc="BPvsSZ gated runs", unit="run")
    else:
        logger.info("Total runs: %d", len(jobs))
        job_iter = jobs

    rows_runs: list[dict[str, Any]] = []
    for setting_idx, s, rep in job_iter:
        feat = s["feature"]
        model_name = s["model"]
        x = np.asarray(npz[feat], dtype=np.float64)[mask_bp_sz]
        x = x[finite_mask_bp]
        seed = args.random_seed + rep * 101

        m = _evaluate_once(
            x=x,
            y=y_all,
            rho=rho_all,
            model_name=model_name,
            n_folds=args.n_folds,
            seed=seed,
            gate_mode=s["gate_mode"],
            rho_quantile=float(s["rho_quantile"]) if np.isfinite(s["rho_quantile"]) else 0.0,
            rho_gamma=float(s["rho_gamma"]),
            min_class_count_train=args.min_class_count_train,
            min_class_count_test=args.min_class_count_test,
        )

        row: dict[str, Any] = {
            "setting_idx": setting_idx,
            "family": s["family"],
            "is_primary": s["is_primary"],
            "feature": feat,
            "model": model_name,
            "gate_mode": s["gate_mode"],
            "rho_quantile": s["rho_quantile"],
            "rho_gamma": s["rho_gamma"],
            "repeat": rep,
            "n_subjects_total": int(len(y_all)),
            "n_eval_subjects": int(m["n_eval_subjects"]) if np.isfinite(m["n_eval_subjects"]) else np.nan,
            "n_splits_used": int(m["n_splits_used"]) if np.isfinite(m["n_splits_used"]) else np.nan,
            "invalid_setting": 0,
            "coverage": float(m["coverage"]),
            "bal_acc": float(m["bal_acc"]),
            "roc_auc": float(m["roc_auc"]),
            "auprc": float(m["auprc"]),
            "bal_acc_p_perm": np.nan,
            "roc_auc_p_perm": np.nan,
            "auprc_p_perm": np.nan,
        }

        do_perm = (not args.permute_primary_only) or (s["family"] == "confirmatory")
        valid_for_setting = (
            np.isfinite(row["bal_acc"])
            and np.isfinite(row["n_splits_used"])
            and int(row["n_splits_used"]) >= int(args.min_valid_splits)
        )
        if not valid_for_setting:
            row["invalid_setting"] = 1
            if rep == 0:
                logger.warning(
                    "Invalid setting (skip permutation): family=%s feature=%s model=%s gate=%s q=%s gamma=%s valid_splits=%s (< %d)",
                    s["family"],
                    s["feature"],
                    s["model"],
                    s["gate_mode"],
                    f"{float(s['rho_quantile']):.3f}" if np.isfinite(float(s["rho_quantile"])) else "nan",
                    f"{float(s['rho_gamma']):.3f}",
                    row["n_splits_used"],
                    int(args.min_valid_splits),
                )

        if do_perm and args.n_permutations > 0 and valid_for_setting:
            label = (
                f"{s['family']}/{feat}/{model_name}/{s['gate_mode']}"
                f"/rep{rep + 1} [{args.n_permutations} perm]"
            )
            pvals = _perm_pvals(
                x=x,
                y=y_all,
                rho=rho_all,
                model_name=model_name,
                n_folds=args.n_folds,
                seed=seed,
                gate_mode=s["gate_mode"],
                rho_quantile=float(s["rho_quantile"]) if np.isfinite(s["rho_quantile"]) else 0.0,
                rho_gamma=float(s["rho_gamma"]),
                min_class_count_train=args.min_class_count_train,
                min_class_count_test=args.min_class_count_test,
                n_permutations=args.n_permutations,
                progress_label=label,
            )
            row.update(pvals)

        rows_runs.append(row)

    run_fields = [
        "setting_idx",
        "family",
        "is_primary",
        "feature",
        "model",
        "gate_mode",
        "rho_quantile",
        "rho_gamma",
        "repeat",
        "n_subjects_total",
        "n_eval_subjects",
        "n_splits_used",
        "invalid_setting",
        "coverage",
        "bal_acc",
        "roc_auc",
        "auprc",
        "bal_acc_p_perm",
        "roc_auc_p_perm",
        "auprc_p_perm",
    ]
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=run_fields)
        w.writeheader()
        w.writerows(rows_runs)

    rows_summary: list[dict[str, Any]] = []
    metrics = ["bal_acc", "roc_auc", "auprc", "coverage", "n_eval_subjects", "n_splits_used"]
    pmetrics = ["bal_acc_p_perm", "roc_auc_p_perm", "auprc_p_perm"]

    for sidx in sorted({int(r["setting_idx"]) for r in rows_runs}):
        rr = [r for r in rows_runs if int(r["setting_idx"]) == sidx]
        s = rr[0]
        row: dict[str, Any] = {
            "setting_idx": sidx,
            "family": s["family"],
            "is_primary": s["is_primary"],
            "feature": s["feature"],
            "model": s["model"],
            "gate_mode": s["gate_mode"],
            "rho_quantile": s["rho_quantile"],
            "rho_gamma": s["rho_gamma"],
            "n_subjects_total": s["n_subjects_total"],
            "n_repeats": len(rr),
            "invalid_rate": float(np.mean([int(r["invalid_setting"]) for r in rr])),
        }

        for k in metrics + pmetrics:
            vals = np.array([float(r[k]) for r in rr], dtype=float)
            valid = np.isfinite(vals)
            if np.any(valid):
                row[f"{k}_mean"] = float(np.mean(vals[valid]))
                row[f"{k}_std"] = float(np.std(vals[valid]))
            else:
                row[f"{k}_mean"] = float("nan")
                row[f"{k}_std"] = float("nan")
        pvals = np.array([float(r["bal_acc_p_perm"]) for r in rr], dtype=float)
        validp = np.isfinite(pvals)
        row["bal_acc_sig_rate_p05"] = float(np.mean(pvals[validp] < 0.05)) if np.any(validp) else np.nan
        rows_summary.append(row)

    # Multiple-comparison correction by family
    for fam, mode in [("confirmatory", "holm"), ("exploratory", "fdr_bh")]:
        idxs = [i for i, r in enumerate(rows_summary) if r["family"] == fam and np.isfinite(r["bal_acc_p_perm_mean"])]
        if not idxs:
            continue
        p = np.array([rows_summary[i]["bal_acc_p_perm_mean"] for i in idxs], dtype=float)
        adj = _holm_adjust(p) if mode == "holm" else _bh_fdr_adjust(p)
        for ii, a in zip(idxs, adj):
            if mode == "holm":
                rows_summary[ii]["bal_acc_p_holm"] = float(a)
                rows_summary[ii]["bal_acc_sig_holm_p05"] = float(a < 0.05)
            else:
                rows_summary[ii]["bal_acc_q_fdr"] = float(a)
                rows_summary[ii]["bal_acc_sig_fdr_q05"] = float(a < 0.05)

    # ensure keys exist for csv
    for r in rows_summary:
        r.setdefault("bal_acc_p_holm", np.nan)
        r.setdefault("bal_acc_sig_holm_p05", np.nan)
        r.setdefault("bal_acc_q_fdr", np.nan)
        r.setdefault("bal_acc_sig_fdr_q05", np.nan)

    summary_fields = [
        "setting_idx",
        "family",
        "is_primary",
        "feature",
        "model",
        "gate_mode",
        "rho_quantile",
        "rho_gamma",
        "n_subjects_total",
        "n_repeats",
        "invalid_rate",
        "coverage_mean",
        "coverage_std",
        "n_eval_subjects_mean",
        "n_eval_subjects_std",
        "n_splits_used_mean",
        "n_splits_used_std",
        "bal_acc_mean",
        "bal_acc_std",
        "roc_auc_mean",
        "roc_auc_std",
        "auprc_mean",
        "auprc_std",
        "bal_acc_p_perm_mean",
        "bal_acc_p_perm_std",
        "roc_auc_p_perm_mean",
        "roc_auc_p_perm_std",
        "auprc_p_perm_mean",
        "auprc_p_perm_std",
        "bal_acc_sig_rate_p05",
        "bal_acc_p_holm",
        "bal_acc_sig_holm_p05",
        "bal_acc_q_fdr",
        "bal_acc_sig_fdr_q05",
    ]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(rows_summary)

    logger.info(f"Saved runs: {runs_csv}")
    logger.info(f"Saved summary: {summary_csv}")

    pri = [r for r in rows_summary if int(r.get("is_primary", 0)) == 1]
    if pri:
        p = pri[0]
        logger.info(
            "Primary report: BalAcc=%.3f±%.3f AUC=%.3f±%.3f cov=%.2f p=%.3f holm=%.3f",
            p["bal_acc_mean"],
            p["bal_acc_std"],
            p["roc_auc_mean"],
            p["roc_auc_std"],
            p["coverage_mean"],
            p["bal_acc_p_perm_mean"],
            p["bal_acc_p_holm"],
        )


if __name__ == "__main__":
    main()
