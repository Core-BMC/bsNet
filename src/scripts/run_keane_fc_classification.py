#!/usr/bin/env python3
"""Run FC-only group classification on converted Keane datasets.

This script uses precomputed FC matrices (not raw BOLD), so it evaluates
downstream discrimination from FC features directly. It is intended as
exploratory utility analysis and should not be framed as standalone biomarker.

Input:
  - data/ds005073/results/keane_restfc_combined.npz
  - data/ds005073/results/keane_restfc_metadata.csv

Outputs:
  - data/ds005073/results/keane_fc_classification_runs.csv
  - data/ds005073/results/keane_fc_classification_summary.csv
"""

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

NPZ_PATH = Path("data/ds005073/results/keane_restfc_combined.npz")
META_CSV = Path("data/ds005073/results/keane_restfc_metadata.csv")
OUT_DIR = Path("data/ds005073/results")
RUNS_CSV = OUT_DIR / "keane_fc_classification_runs.csv"
SUMMARY_CSV = OUT_DIR / "keane_fc_classification_summary.csv"

MODELS = ("logistic_l2", "linear_svm")
TASKS = ("hc_vs_psychosis", "bp_vs_sz")


def _load_data() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Missing NPZ: {NPZ_PATH}")
    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {META_CSV}")

    npz = np.load(NPZ_PATH)
    fc_all = np.asarray(npz["fc_all"], dtype=np.float64)
    cohort_codes = np.asarray(npz["cohort_codes"], dtype=int)

    with open(META_CSV) as f:
        meta_rows = list(csv.DictReader(f))
    if len(meta_rows) != fc_all.shape[0]:
        raise RuntimeError(
            f"Metadata/FC mismatch: meta={len(meta_rows)}, fc={fc_all.shape[0]}",
        )
    return fc_all, cohort_codes, meta_rows


def _vectorize_upper(fc: np.ndarray) -> np.ndarray:
    n_roi = fc.shape[1]
    iu = np.triu_indices(n_roi, k=1)
    return fc[:, iu[0], iu[1]]


def _subset_task(
    x_all: np.ndarray,
    cohort_codes: np.ndarray,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    if task == "hc_vs_psychosis":
        # 0=HC, 1=BP, 2=SZ
        y = np.array([0 if c == 0 else 1 for c in cohort_codes], dtype=int)
        return x_all, y

    if task == "bp_vs_sz":
        mask = cohort_codes != 0
        x = x_all[mask]
        y = np.array([0 if c == 1 else 1 for c in cohort_codes[mask]], dtype=int)  # BP=0, SZ=1
        return x, y

    raise ValueError(f"Unknown task: {task}")


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
) -> dict[str, float]:
    obs = _evaluate_once(x=x, y=y, model_name=model_name, n_folds=n_folds, seed=seed)
    obs_bal = obs["bal_acc"]
    obs_auc = obs["roc_auc"]
    obs_pr = obs["auprc"]

    rng = np.random.default_rng(seed + 100_000)
    null_bal = np.empty(n_permutations, dtype=float)
    null_auc = np.empty(n_permutations, dtype=float)
    null_pr = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
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
    parser = argparse.ArgumentParser(description="FC-only classification on Keane datasets.")
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    fc_all, cohort_codes, _ = _load_data()
    x_all = _vectorize_upper(fc_all)
    logger.info(
        f"Loaded FC: N={fc_all.shape[0]}, ROI={fc_all.shape[1]}, features={x_all.shape[1]}",
    )

    logger.warning(
        "HC vs Psychosis task is dataset-confounded "
        "(HC from ds003404, psychosis from ds005073). "
        "Treat as exploratory only.",
    )

    runs: list[dict] = []
    for task in TASKS:
        x, y = _subset_task(x_all, cohort_codes=cohort_codes, task=task)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        logger.info(f"\n[{task}] N={len(y)} (class1={n_pos}, class0={n_neg})")

        for model_name in MODELS:
            logger.info(f"  model={model_name}")
            for rep in range(args.n_repeats):
                seed = args.random_seed + rep * 101
                m = _evaluate_once(
                    x=x,
                    y=y,
                    model_name=model_name,
                    n_folds=args.n_folds,
                    seed=seed,
                )
                pvals = {}
                # permutation on first repeat only for speed
                if rep == 0 and args.n_permutations > 0:
                    pvals = _perm_pvals(
                        x=x,
                        y=y,
                        model_name=model_name,
                        n_folds=args.n_folds,
                        seed=seed,
                        n_permutations=args.n_permutations,
                    )

                runs.append({
                    "task": task,
                    "model": model_name,
                    "repeat": rep,
                    "n_subjects": len(y),
                    "n_class1": n_pos,
                    "n_class0": n_neg,
                    **m,
                    "bal_acc_p_perm": float(pvals.get("bal_acc_p_perm", np.nan)),
                    "roc_auc_p_perm": float(pvals.get("roc_auc_p_perm", np.nan)),
                    "auprc_p_perm": float(pvals.get("auprc_p_perm", np.nan)),
                })

    if not runs:
        logger.error("No runs generated.")
        return

    out_tag = args.output_tag.strip()
    if out_tag:
        safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in out_tag)
        runs_csv = OUT_DIR / f"keane_fc_classification_runs_{safe_tag}.csv"
        summary_csv = OUT_DIR / f"keane_fc_classification_summary_{safe_tag}.csv"
    else:
        runs_csv = RUNS_CSV
        summary_csv = SUMMARY_CSV

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)
    logger.info(f"Saved runs: {runs_csv}")

    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        key = (r["task"], r["model"])
        grouped.setdefault(key, []).append(r)

    summary_rows: list[dict] = []
    for (task, model_name), rows in grouped.items():
        p_bal = [r["bal_acc_p_perm"] for r in rows if np.isfinite(r["bal_acc_p_perm"])]
        p_auc = [r["roc_auc_p_perm"] for r in rows if np.isfinite(r["roc_auc_p_perm"])]
        p_pr = [r["auprc_p_perm"] for r in rows if np.isfinite(r["auprc_p_perm"])]
        row0 = rows[0]
        summary_rows.append({
            "task": task,
            "model": model_name,
            "n_runs": len(rows),
            "n_subjects": row0["n_subjects"],
            "n_class1": row0["n_class1"],
            "n_class0": row0["n_class0"],
            "bal_acc_mean": float(np.mean([r["bal_acc"] for r in rows])),
            "bal_acc_std": float(np.std([r["bal_acc"] for r in rows])),
            "roc_auc_mean": float(np.nanmean([r["roc_auc"] for r in rows])),
            "roc_auc_std": float(np.nanstd([r["roc_auc"] for r in rows])),
            "auprc_mean": float(np.nanmean([r["auprc"] for r in rows])),
            "auprc_std": float(np.nanstd([r["auprc"] for r in rows])),
            "bal_acc_p_perm": float(np.mean(p_bal)) if p_bal else float("nan"),
            "roc_auc_p_perm": float(np.mean(p_auc)) if p_auc else float("nan"),
            "auprc_p_perm": float(np.mean(p_pr)) if p_pr else float("nan"),
        })

    summary_rows = sorted(summary_rows, key=lambda r: (r["task"], r["model"]))
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    logger.info(f"Saved summary: {summary_csv}")

    logger.info("\nPrimary report (logistic_l2):")
    for task in TASKS:
        cand = [r for r in summary_rows if r["task"] == task and r["model"] == "logistic_l2"]
        if not cand:
            continue
        r = cand[0]
        logger.info(
            f"  {task:16s} | BalAcc={r['bal_acc_mean']:.3f}±{r['bal_acc_std']:.3f} "
            f"AUC={r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.3f} "
            f"pBalAcc={r['bal_acc_p_perm']:.3f}",
        )


if __name__ == "__main__":
    main()
