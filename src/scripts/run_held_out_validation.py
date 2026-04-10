"""Held-out prediction validation for BS-NET (sliding-window design).

Fundamental proof experiment: does BS-NET's ρ̂T accurately predict
what you would observe with the *remaining* (longer) scan?

Design (2-segment, clinically realistic):
────────────────────────────────────────────────────────────────────────────
  |←── A (short_sec, e.g. 120 s) ──→|←──── B (remaining full scan) ────→|
        sliding-window bootstrap           fc_reference (LW)
────────────────────────────────────────────────────────────────────────────

  1. A  → BS-NET input  (run_sliding_window_prediction)
  2. B  → reference FC  (LW-shrinkage FC vector)
           AND reference  r_FC(A, B)

Rationale for sliding-window:
  run_sliding_window_prediction() applies the full BS-NET pipeline
  (block bootstrap + SB prophecy + Bayesian prior + attenuation)
  over multiple overlapping sub-windows of A, then averages in
  Fisher-z space.  This is more stable than single-window bootstrap
  and matches the intended clinical usage of BS-NET.

Clinical analogy:
  - Total scan 12 min  →  A = first 2 min,  B = remaining 10 min
  - ρ̂T(SW) should approach r_FC(A, B)  (what the longer scan gives)

Metrics (Spearman ρ on upper-triangle FC vectors):
  r_fc_AB      : raw FC correlation(A, B)     — uncorrected baseline
  rho_hat_T_sw : BS-NET sliding-window pred.  — corrected estimate
  rho_hat_T_bp : BS-NET simple bootstrap      — comparison reference
  r_fc_BB      : split-half of B              — within-B ceiling

Hypothesis:
  ρ̂T(SW) ≈ ρ̂T(BP) >> r_FC(A,B)   (both correct upward)
  ρ̂T(SW) closer to r_FC_BB ceiling than raw r_FC_AB

Strong-FC subset (|FC_B| >= fc_thresh) computed in parallel to
address zero-inflation concern.

Usage:
    python src/scripts/run_held_out_validation.py \\
        --atlas 4s256parcels \\
        --short-sec 120 \\
        --n-seeds 5 \\
        --n-jobs 4 \\
        --out-csv data/ds000243/results/held_out_validation_sw_4s256parcels.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction, run_sliding_window_prediction
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CACHE_DIR = Path("data/ds000243/timeseries_cache_xcpd")
RESULTS_DIR = Path("data/ds000243/results")
TR = 2.5
# Need A + at least MIN_B_SEC of reference
MIN_B_SEC = 120.0       # reference segment must be ≥ 2 min
CORRECTION_METHOD = "fisher_z"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pearson_fc_vec(ts: np.ndarray) -> np.ndarray:
    """Return upper-triangle Pearson FC vector; NaN → 0.

    Args:
        ts: Time series array, shape (n_tp, n_rois).

    Returns:
        1-D upper-triangle FC vector.
    """
    fc = np.corrcoef(ts.T)
    fc = np.nan_to_num(fc, nan=0.0)
    i, j = np.triu_indices_from(fc, k=1)
    return fc[i, j].astype(np.float64)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Spearman ρ.
    """
    r, _ = spearmanr(a, b)
    return float(r)


# ── Per-subject worker ────────────────────────────────────────────────────────

def _process_subject(
    fp: Path,
    short_sec: float,
    n_seeds: int,
    fc_thresh: float,
) -> list[dict] | None:
    """Run held-out validation for one subject.

    Splits timeseries into A (short) and B (remaining full reference).
    Runs both sliding-window and simple bootstrap BS-NET on A, using B
    as the reference FC.  Ground truth is r_FC(A, B).

    Args:
        fp: Path to .npy timeseries file, shape (n_tp, n_rois).
        short_sec: Duration of segment A in seconds.
        n_seeds: Number of random seeds.
        fc_thresh: |FC| threshold for strong-connection subset (based on B).

    Returns:
        List of result dicts (one per seed), or None on failure.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    sub_id = fp.stem.split("_")[0]
    try:
        ts = np.load(fp).astype(np.float64)
    except Exception as exc:
        logger.warning(f"{sub_id}: load failed — {exc}")
        return None

    n_tp, n_rois = ts.shape
    total_sec = n_tp * TR
    n_A = int(short_sec / TR)

    if n_A >= n_tp:
        logger.info(f"{sub_id}: scan too short for A={short_sec}s; skip")
        return None

    ts_A = ts[:n_A, :]
    ts_B = ts[n_A:, :]
    sec_A = ts_A.shape[0] * TR
    sec_B = ts_B.shape[0] * TR

    if sec_B < MIN_B_SEC:
        logger.info(f"{sub_id}: B={sec_B:.0f}s < {MIN_B_SEC}s; skip")
        return None

    # ── Remove zero-variance ROIs (across full scan) ──────────────────────────
    # Zero-variance ROIs in short windows cause NaN in corrcoef after
    # block bootstrap resampling (48 TPs × 256 ROIs is rank-deficient).
    roi_std = ts.std(axis=0)
    valid_rois = roi_std > 1e-6
    if not valid_rois.all():
        n_removed = int((~valid_rois).sum())
        logger.debug(f"{sub_id}: removing {n_removed} zero-variance ROIs")
        ts   = ts[:, valid_rois]
        ts_A = ts[:n_A, :]
        ts_B = ts[n_A:, :]
        n_rois = ts.shape[1]

    # ── FC vectors ────────────────────────────────────────────────────────────
    fc_A = _pearson_fc_vec(ts_A)
    fc_B = _pearson_fc_vec(ts_B)

    # Split-half of B for within-B ceiling
    n_half = ts_B.shape[0] // 2
    fc_B1  = _pearson_fc_vec(ts_B[:n_half, :])
    fc_B2  = _pearson_fc_vec(ts_B[n_half:, :])
    r_fc_BB = _spearman(fc_B1, fc_B2)

    # Ground truth: raw correlation A vs B
    r_fc_AB = _spearman(fc_A, fc_B)

    # Reference FC for BS-NET (LW shrinkage on B)
    fc_B_lw = get_fc_matrix(ts_B, vectorized=True, use_shrinkage=True)

    # Strong-connection mask based on |FC_B|
    strong_mask  = np.abs(fc_B) >= fc_thresh
    n_strong     = int(strong_mask.sum())
    r_fc_AB_str  = _spearman(fc_A[strong_mask], fc_B[strong_mask]) if n_strong > 10 else float("nan")

    records = []
    for seed in range(n_seeds):
        config = BSNetConfig(
            n_rois=n_rois,
            tr=TR,
            short_duration_sec=int(short_sec),
            target_duration_min=15,
            n_bootstraps=100,
            reliability_coeff=0.98,
            empirical_prior=(0.25, 0.05),
            seed=seed,
        )

        # ── Sliding-window prediction ─────────────────────────────────────────
        try:
            res_sw = run_sliding_window_prediction(
                short_obs=ts_A,
                fc_reference=fc_B_lw,
                config=config,
                correction_method=CORRECTION_METHOD,
                window_sec=short_sec / 2.0,   # 60 s windows for 120 s scan
                step_sec=short_sec / 8.0,     # 15 s step → ~75% overlap
            )
            rho_sw    = res_sw.rho_hat_T
            ci_sw_lo  = res_sw.ci_lower
            ci_sw_hi  = res_sw.ci_upper
        except Exception as exc:
            logger.warning(f"{sub_id} seed={seed} SW failed: {exc}")
            rho_sw = ci_sw_lo = ci_sw_hi = float("nan")

        # ── Simple bootstrap prediction (comparison) ──────────────────────────
        try:
            res_bp = run_bootstrap_prediction(
                short_obs=ts_A,
                fc_reference=fc_B_lw,
                config=config,
                correction_method=CORRECTION_METHOD,
            )
            rho_bp   = res_bp.rho_hat_T
            ci_bp_lo = res_bp.ci_lower
            ci_bp_hi = res_bp.ci_upper
        except Exception as exc:
            logger.warning(f"{sub_id} seed={seed} BP failed: {exc}")
            rho_bp = ci_bp_lo = ci_bp_hi = float("nan")

        records.append({
            "sub_id":        sub_id,
            "seed":          seed,
            "n_rois":        n_rois,
            "sec_A":         sec_A,
            "sec_B":         sec_B,
            "total_sec":     total_sec,
            "n_pairs":       len(fc_A),
            "n_strong":      n_strong,
            "fc_thresh":     fc_thresh,
            # Ground truth (no BS-NET)
            "r_fc_AB":       r_fc_AB,
            "r_fc_BB":       r_fc_BB,       # within-B split-half ceiling
            # Sliding-window BS-NET
            "rho_hat_T_sw":  rho_sw,
            "ci_sw_lo":      ci_sw_lo,
            "ci_sw_hi":      ci_sw_hi,
            # Simple bootstrap BS-NET
            "rho_hat_T_bp":  rho_bp,
            "ci_bp_lo":      ci_bp_lo,
            "ci_bp_hi":      ci_bp_hi,
            # Strong-FC subset
            "r_fc_AB_strong": r_fc_AB_str,
        })

    logger.info(
        f"{sub_id}: {total_sec:.0f}s | B={sec_B:.0f}s | "
        f"r_AB={r_fc_AB:.3f} → ρ̂T(SW)={rho_sw:.3f} ρ̂T(BP)={rho_bp:.3f} | "
        f"ceiling(BB)={r_fc_BB:.3f}"
    )
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def run_held_out_validation(
    atlas: str,
    short_sec: float,
    n_seeds: int,
    fc_thresh: float,
    n_jobs: int,
    out_csv: Path,
) -> pd.DataFrame:
    """Run sliding-window held-out validation across all subjects.

    Args:
        atlas: Atlas directory name under CACHE_DIR.
        short_sec: Duration of short segment A in seconds.
        n_seeds: Number of random seeds per subject.
        fc_thresh: |FC_B| threshold for strong-connection subset.
        n_jobs: Parallel workers.
        out_csv: Output CSV path.

    Returns:
        DataFrame with per-subject, per-seed results.
    """
    atlas_dir = CACHE_DIR / atlas
    if not atlas_dir.exists():
        raise FileNotFoundError(f"Atlas cache not found: {atlas_dir}")

    files = sorted(atlas_dir.glob("*.npy"))
    logger.info(f"Found {len(files)} subjects in {atlas_dir}")

    all_records: list[dict] = []

    if n_jobs == 1:
        for fp in files:
            recs = _process_subject(fp, short_sec, n_seeds, fc_thresh)
            if recs:
                all_records.extend(recs)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {
                ex.submit(_process_subject, fp, short_sec, n_seeds, fc_thresh): fp
                for fp in files
            }
            for fut in as_completed(futs):
                recs = fut.result()
                if recs:
                    all_records.extend(recs)

    if not all_records:
        logger.error("No valid results collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved {len(df)} records → {out_csv}")

    # ── Summary ───────────────────────────────────────────────────────────────
    cols = ["r_fc_AB", "rho_hat_T_sw", "rho_hat_T_bp", "r_fc_BB", "r_fc_AB_strong"]
    agg  = df.groupby("sub_id")[cols].mean()

    print(f"\n{'='*65}")
    print(f"Held-out Validation (Sliding-Window) — atlas={atlas}")
    print(f"  Subjects   : {agg.shape[0]}")
    print(f"  Short scan : {short_sec:.0f}s  (segment A)")
    print(f"  Reference  : remaining full scan  (segment B)")
    print(f"  Threshold  : |FC_B| >= {fc_thresh}")
    print(f"{'='*65}")
    print(f"  All-pairs (mean across subjects × seeds):")
    print(f"    r_FC(A,B)      = {agg['r_fc_AB'].mean():.3f}   ← uncorrected baseline")
    print(f"    ρ̂T (SW)        = {agg['rho_hat_T_sw'].mean():.3f}   ← BS-NET sliding-window")
    print(f"    ρ̂T (bootstrap) = {agg['rho_hat_T_bp'].mean():.3f}   ← BS-NET simple bootstrap")
    print(f"    r_FC(B1,B2)    = {agg['r_fc_BB'].mean():.3f}   ← within-B split-half ceiling")
    print(f"  Strong-FC (|FC_B|>={fc_thresh}):")
    print(f"    r_FC(A,B)      = {agg['r_fc_AB_strong'].mean():.3f}")
    print(f"{'='*65}")

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="BS-NET held-out validation: 2-segment + sliding window"
    )
    parser.add_argument("--atlas",     default="4s256parcels",
                        help="Atlas key under data/ds000243/timeseries_cache_xcpd/")
    parser.add_argument("--short-sec", type=float, default=120.0,
                        help="Duration of short scan segment A in seconds (default: 120)")
    parser.add_argument("--n-seeds",   type=int,   default=5,
                        help="Random seeds per subject (default: 5)")
    parser.add_argument("--fc-thresh", type=float, default=0.20,
                        help="|FC_B| threshold for strong-FC subset (default: 0.20)")
    parser.add_argument("--n-jobs",    type=int,   default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--out-csv",   type=Path,  default=None)
    args = parser.parse_args()

    if args.out_csv is None:
        args.out_csv = RESULTS_DIR / f"held_out_validation_sw_{args.atlas}.csv"

    run_held_out_validation(
        atlas=args.atlas,
        short_sec=args.short_sec,
        n_seeds=args.n_seeds,
        fc_thresh=args.fc_thresh,
        n_jobs=args.n_jobs,
        out_csv=args.out_csv,
    )
