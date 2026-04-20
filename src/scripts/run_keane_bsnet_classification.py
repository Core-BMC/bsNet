#!/usr/bin/env python3
"""Classification on Keane BS-NET feature NPZ (primary: BP vs SZ)."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

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
OUT_DIR = Path("data/keane/results")


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    model_name: str,
    seed: int,
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

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test).astype(int)
    if hasattr(clf, "decision_function"):
        score = clf.decision_function(x_test)
    else:
        score = y_pred.astype(float)
    return y_pred, np.asarray(score, dtype=float)


def _evaluate_once(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_folds: int,
    seed: int,
) -> dict[str, float]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_pred_all = np.full(len(y), -1, dtype=int)
    score_all = np.full(len(y), np.nan, dtype=float)

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        y_pred, score = _fit_predict(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            model_name=model_name,
            seed=seed + fold_i * 17,
        )
        y_pred_all[test_idx] = y_pred
        score_all[test_idx] = score

    bal = float(balanced_accuracy_score(y, y_pred_all))
    try:
        auc = float(roc_auc_score(y, score_all))
    except Exception:
        auc = float("nan")
    try:
        auprc = float(average_precision_score(y, score_all))
    except Exception:
        auprc = float("nan")

    return {"bal_acc": bal, "roc_auc": auc, "auprc": auprc}


def _perm_pvals(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_folds: int,
    seed: int,
    n_permutations: int,
    progress_label: str = "",
) -> dict[str, float]:
    obs = _evaluate_once(x=x, y=y, model_name=model_name, n_folds=n_folds, seed=seed)
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
            model_name=model_name,
            n_folds=n_folds,
            seed=seed + i * 19,
        )
        null_bal[i] = m["bal_acc"]
        null_auc[i] = m["roc_auc"]
        null_pr[i] = m["auprc"]

        if tqdm is None and n_permutations >= 10:
            step = max(1, n_permutations // 10)
            if (i + 1) % step == 0 or (i + 1) == n_permutations:
                logger.info(
                    "%s permutations: %d/%d",
                    progress_label or "Running",
                    i + 1,
                    n_permutations,
                )

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


def main() -> None:
    parser = argparse.ArgumentParser(description="BP vs SZ classification on Keane BS-NET features.")
    parser.add_argument("--features-npz", type=Path, default=FEATURES_NPZ)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument(
        "--primary-feature",
        type=str,
        default="fc_bsnet_pred",
        choices=["fc_raw_short", "fc_bsnet_pred", "fc_reference"],
    )
    parser.add_argument(
        "--primary-model",
        type=str,
        default="logistic_l2",
        choices=["logistic_l2", "linear_svm"],
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

    npz = np.load(args.features_npz, allow_pickle=True)
    labels3 = np.asarray(npz["labels_threeclass"], dtype=int)
    mask = labels3 != 0  # BP/SZ only
    y = np.array([0 if c == 1 else 1 for c in labels3[mask]], dtype=int)  # BP=0, SZ=1

    feature_names = ["fc_raw_short", "fc_bsnet_pred", "fc_reference"]
    models = ["logistic_l2", "linear_svm"]
    if len(np.unique(y)) < 2:
        raise RuntimeError("BP/SZ labels are not both present in features NPZ.")

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    runs_csv = out_dir / f"keane_bsnet_bp_sz_runs{tag}.csv"
    summary_csv = out_dir / f"keane_bsnet_bp_sz_summary{tag}.csv"

    rows_runs: list[dict] = []
    jobs: list[tuple[str, str, int]] = []
    for feat in feature_names:
        for model_name in models:
            for rep in range(args.n_repeats):
                jobs.append((feat, model_name, rep))

    if tqdm is not None:
        job_iter = tqdm(jobs, desc="BPvsSZ runs", unit="run")
    else:
        logger.info("Total runs: %d", len(jobs))
        job_iter = jobs

    for feat, model_name, rep in job_iter:
        x_all = np.asarray(npz[feat], dtype=np.float64)[mask]
        seed = args.random_seed + rep * 101
        m = _evaluate_once(
            x=x_all,
            y=y,
            model_name=model_name,
            n_folds=args.n_folds,
            seed=seed,
        )
        row = {
            "task": "bp_vs_sz",
            "feature": feat,
            "model": model_name,
            "repeat": rep,
            "n_subjects": int(len(y)),
            "bal_acc": float(m["bal_acc"]),
            "roc_auc": float(m["roc_auc"]),
            "auprc": float(m["auprc"]),
            "bal_acc_p_perm": np.nan,
            "roc_auc_p_perm": np.nan,
            "auprc_p_perm": np.nan,
        }

        do_perm = (not args.permute_primary_only) or (
            feat == args.primary_feature and model_name == args.primary_model
        )
        if do_perm and args.n_permutations > 0:
            progress_label = (
                f"{feat}/{model_name}/rep{rep + 1} "
                f"[{args.n_permutations} perm]"
            )
            pvals = _perm_pvals(
                x=x_all,
                y=y,
                model_name=model_name,
                n_folds=args.n_folds,
                seed=seed,
                n_permutations=args.n_permutations,
                progress_label=progress_label,
            )
            row.update(pvals)
        rows_runs.append(row)

    run_fields = [
        "task",
        "feature",
        "model",
        "repeat",
        "n_subjects",
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

    rows_summary: list[dict] = []
    keys = ["bal_acc", "roc_auc", "auprc", "bal_acc_p_perm", "roc_auc_p_perm", "auprc_p_perm"]
    for feat in feature_names:
        for model_name in models:
            rr = [r for r in rows_runs if r["feature"] == feat and r["model"] == model_name]
            row = {
                "task": "bp_vs_sz",
                "feature": feat,
                "model": model_name,
                "n_subjects": int(len(y)),
                "n_repeats": int(len(rr)),
            }
            for k in keys:
                vals = np.array([float(r[k]) for r in rr], dtype=float)
                row[f"{k}_mean"] = float(np.nanmean(vals))
                row[f"{k}_std"] = float(np.nanstd(vals))
            pvals = np.array([float(r["bal_acc_p_perm"]) for r in rr], dtype=float)
            valid = np.isfinite(pvals)
            row["bal_acc_sig_rate_p05"] = float(np.mean(pvals[valid] < 0.05)) if np.any(valid) else np.nan
            rows_summary.append(row)

    summary_fields = [
        "task",
        "feature",
        "model",
        "n_subjects",
        "n_repeats",
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
    ]
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(rows_summary)

    logger.info(f"Saved runs: {runs_csv}")
    logger.info(f"Saved summary: {summary_csv}")

    primary = [r for r in rows_summary if r["feature"] == args.primary_feature and r["model"] == args.primary_model]
    if primary:
        p = primary[0]
        logger.info(
            "Primary report (BP vs SZ): feature=%s model=%s BalAcc=%.3f±%.3f AUC=%.3f±%.3f pBalAcc=%.3f (sig<.05=%.2f)",
            p["feature"],
            p["model"],
            p["bal_acc_mean"],
            p["bal_acc_std"],
            p["roc_auc_mean"],
            p["roc_auc_std"],
            p["bal_acc_p_perm_mean"],
            p["bal_acc_sig_rate_p05"],
        )


if __name__ == "__main__":
    main()
