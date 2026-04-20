#!/usr/bin/env python3
"""Recompute Keane BS-NET metrics from cached per-subject time series.

This script is designed for the streamed Keane pipeline outputs:
  data/derivatives/bsnet/sub-*/sub-*_ts.npy

It recomputes BS-NET with a chosen correction method (recommended: fisher_z),
and exports:
  1) per-subject metrics CSV
  2) edge-feature NPZ for downstream classification
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction

logger = logging.getLogger(__name__)

BSNET_DIR = Path("data/derivatives/bsnet")
OUT_DIR = Path("data/keane/results")


def _compute_fc_lw(ts: np.ndarray) -> np.ndarray:
    lw = LedoitWolf()
    lw.fit(ts)
    cov = lw.covariance_
    d = np.sqrt(np.diag(cov))
    d[d == 0] = 1e-10
    fc = cov / np.outer(d, d)
    np.fill_diagonal(fc, 1.0)
    return np.clip(fc, -1.0, 1.0)


def _cohort_from_sub_id(sub_id: str) -> tuple[str, int]:
    if sub_id.startswith("sub-C"):
        return "HC", 0
    if sub_id.startswith("sub-B"):
        return "BP", 1
    if sub_id.startswith("sub-S"):
        return "SZ", 2
    return "UNK", -1


def _load_tr(sub_dir: Path, tr_default: float) -> float:
    js = sub_dir / f"{sub_dir.name}_bsnet_results.json"
    if not js.exists():
        return tr_default
    try:
        with open(js) as f:
            d = json.load(f)
        tr = float(d.get("tr", tr_default))
        if np.isfinite(tr) and tr > 0:
            return tr
    except Exception:
        pass
    return tr_default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute Keane BS-NET from cached ts.npy files.",
    )
    parser.add_argument("--bsnet-dir", type=Path, default=BSNET_DIR)
    parser.add_argument("--short-sec", type=float, default=120.0)
    parser.add_argument("--tr-default", type=float, default=0.785)
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument(
        "--correction-method",
        type=str,
        default="fisher_z",
        choices=["original", "fisher_z", "partial", "soft_clamp"],
    )
    parser.add_argument("--output-tag", type=str, default="")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    out_csv = out_dir / f"keane_bsnet_recomputed{tag}.csv"
    out_npz = out_dir / f"keane_bsnet_features{tag}.npz"

    sub_dirs = sorted([p for p in args.bsnet_dir.glob("sub-*") if p.is_dir()])
    if not sub_dirs:
        raise FileNotFoundError(f"No subject dirs found under {args.bsnet_dir}")

    rows: list[dict] = []
    fc_raw_list: list[np.ndarray] = []
    fc_bsnet_list: list[np.ndarray] = []
    fc_ref_list: list[np.ndarray] = []
    labels_three: list[int] = []

    n_fail = 0
    for sub_dir in sub_dirs:
        sub_id = sub_dir.name
        ts_path = sub_dir / f"{sub_id}_ts.npy"
        if not ts_path.exists():
            logger.warning(f"[{sub_id}] missing ts file: {ts_path}")
            n_fail += 1
            continue

        cohort, label3 = _cohort_from_sub_id(sub_id)
        if label3 < 0:
            logger.warning(f"[{sub_id}] unknown cohort prefix (skip)")
            n_fail += 1
            continue

        ts = np.load(ts_path)
        if ts.ndim != 2 or ts.shape[1] < 2:
            logger.warning(f"[{sub_id}] invalid ts shape: {ts.shape}")
            n_fail += 1
            continue

        tr = _load_tr(sub_dir, tr_default=args.tr_default)
        short_vols = int(args.short_sec / tr)
        n_vols = int(ts.shape[0])
        if n_vols < short_vols + 10:
            logger.warning(f"[{sub_id}] too short: n_vols={n_vols}, short_vols={short_vols}")
            n_fail += 1
            continue

        total_min = (n_vols * tr) / 60.0
        ts_short = ts[:short_vols, :]

        fc_full = _compute_fc_lw(ts)
        fc_short = _compute_fc_lw(ts_short)
        n_roi = fc_full.shape[0]
        iu = np.triu_indices(n_roi, k=1)
        ref_vec = fc_full[iu]
        raw_vec = fc_short[iu]
        r_fc_raw = float(np.corrcoef(raw_vec, ref_vec)[0, 1])

        cfg = BSNetConfig(
            tr=tr,
            short_duration_sec=float(args.short_sec),
            target_duration_min=float(total_min),
            n_bootstraps=int(args.n_bootstraps),
        )
        res = run_bootstrap_prediction(
            short_obs=ts_short,
            fc_reference=ref_vec,
            config=cfg,
            correction_method=args.correction_method,
        )

        inflate = float(res.rho_hat_T) / max(r_fc_raw, 0.01)
        pred_vec = np.clip(raw_vec * inflate, -1.0, 1.0)
        r_fc_bsnet = float(np.corrcoef(pred_vec, ref_vec)[0, 1])

        row = {
            "sub_id": sub_id,
            "cohort": cohort,
            "label_threeclass": label3,
            "n_vols": n_vols,
            "n_roi": n_roi,
            "tr": tr,
            "short_vols": short_vols,
            "total_min": total_min,
            "r_fc_raw": r_fc_raw,
            "r_fc_bsnet": r_fc_bsnet,
            "rho_hat_T": float(res.rho_hat_T),
            "ci_lower": float(res.ci_lower),
            "ci_upper": float(res.ci_upper),
            "correction_method": args.correction_method,
        }
        rows.append(row)
        fc_raw_list.append(raw_vec.astype(np.float32))
        fc_bsnet_list.append(pred_vec.astype(np.float32))
        fc_ref_list.append(ref_vec.astype(np.float32))
        labels_three.append(label3)

    if not rows:
        raise RuntimeError("No valid subjects processed.")

    fieldnames = [
        "sub_id",
        "cohort",
        "label_threeclass",
        "n_vols",
        "n_roi",
        "tr",
        "short_vols",
        "total_min",
        "r_fc_raw",
        "r_fc_bsnet",
        "rho_hat_T",
        "ci_lower",
        "ci_upper",
        "correction_method",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        out_npz,
        sub_ids=np.array([r["sub_id"] for r in rows], dtype=object),
        cohorts=np.array([r["cohort"] for r in rows], dtype=object),
        labels_threeclass=np.array(labels_three, dtype=np.int32),
        fc_raw_short=np.stack(fc_raw_list, axis=0),
        fc_bsnet_pred=np.stack(fc_bsnet_list, axis=0),
        fc_reference=np.stack(fc_ref_list, axis=0),
    )

    n_hc = sum(1 for r in rows if r["cohort"] == "HC")
    n_bp = sum(1 for r in rows if r["cohort"] == "BP")
    n_sz = sum(1 for r in rows if r["cohort"] == "SZ")
    rho_vals = np.array([float(r["rho_hat_T"]) for r in rows], dtype=float)
    raw_vals = np.array([float(r["r_fc_raw"]) for r in rows], dtype=float)

    logger.info(f"Processed subjects: {len(rows)} (fail/skip: {n_fail})")
    logger.info(f"Cohorts: HC={n_hc}, BP={n_bp}, SZ={n_sz}")
    logger.info(
        "Mean reliability: r_FC(raw)=%.4f, rho_hat_T(%s)=%.4f, delta=%.4f",
        float(np.mean(raw_vals)),
        args.correction_method,
        float(np.mean(rho_vals)),
        float(np.mean(rho_vals - raw_vals)),
    )
    logger.info(f"Saved subject metrics: {out_csv}")
    logger.info(f"Saved feature NPZ: {out_npz}")


if __name__ == "__main__":
    main()
