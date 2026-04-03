"""Held-out prediction validation for BS-NET.

Fundamental proof experiment: does BS-NET's ρ̂T actually predict
what you would observe with *more independent data*?

Design (3-segment split):
────────────────────────────────────────────────────────────────────────
  |←── A (120 s) ──→|←────── B (T_ref s) ──────→|←── C (T_ref s) ──→|
        short scan      reference (known to BS-NET)    held-out (blind)
────────────────────────────────────────────────────────────────────────

  1. A  → BS-NET short observation  (short_obs)
  2. B  → reference FC (fc_reference)   — the "15-min proxy" within this subject
  3. C  → held-out FC (fc_holdout)      — *never seen* by BS-NET

Metrics (all Spearman ρ on upper-triangle FC vectors):
  r_fc_AC   : raw FC correlation(A, C)          — uncorrected baseline
  rho_hat_T : BS-NET prediction (using A + B)   — corrected estimate
  r_fc_BC   : FC correlation(B, C)              — expected ceiling (same-subject, independent)
  r_fc_full : FC correlation(A, B+C)            — "what more data gives" (ground truth)

Hypothesis:
  ρ̂T ≈ r_fc_BC ≈ r_fc_full >> r_fc_AC

Also computes strong-FC version (|FC_C| > fc_thresh) to address zero-inflation.

Usage:
    python src/scripts/run_held_out_validation.py \\
        --atlas 4s256parcels \\
        --short-sec 120 \\
        --n-seeds 10 \\
        --n-jobs 4 \\
        --out-csv data/ds000243/results/held_out_validation_4s256parcels.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CACHE_DIR = Path("data/ds000243/timeseries_cache_xcpd")
RESULTS_DIR = Path("data/ds000243/results")
TR = 2.5
MIN_TOTAL_SEC = 600.0   # need enough for A + B + C
CORRECTION_METHOD = "fisher_z"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pearson_fc_vec(ts: np.ndarray) -> np.ndarray:
    """Return upper-triangle Pearson FC vector with NaN → 0.

    Args:
        ts: Time series array, shape (n_tp, n_rois).

    Returns:
        1-D array of upper-triangle FC values.
    """
    fc = np.corrcoef(ts.T)
    fc = np.nan_to_num(fc, nan=0.0)
    i, j = np.triu_indices_from(fc, k=1)
    return fc[i, j].astype(np.float64)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation between two FC vectors.

    Args:
        a: First FC vector.
        b: Second FC vector.

    Returns:
        Spearman ρ.
    """
    r, _ = spearmanr(a, b)
    return float(r)


def _split_timeseries(
    ts: np.ndarray,
    short_sec: float,
    tr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Split timeseries into A (short), B (reference), C (held-out).

    B and C receive equal halves of whatever remains after removing A.

    Args:
        ts: Full timeseries, shape (n_tp, n_rois).
        short_sec: Duration of segment A in seconds.
        tr: Repetition time in seconds.

    Returns:
        Tuple (ts_A, ts_B, ts_C) or None if insufficient data.
    """
    n_tp = ts.shape[0]
    n_A = int(short_sec / tr)

    if n_A >= n_tp:
        logger.debug(f"n_A={n_A} >= n_tp={n_tp}; skip")
        return None

    remaining = n_tp - n_A
    if remaining < 2:
        return None

    n_B = remaining // 2
    n_C = remaining - n_B  # may differ by 1

    if n_B < 4 or n_C < 4:   # need at least 4 TPs per segment
        return None

    ts_A = ts[:n_A, :]
    ts_B = ts[n_A : n_A + n_B, :]
    ts_C = ts[n_A + n_B :, :]
    return ts_A, ts_B, ts_C


# ── Per-subject worker ────────────────────────────────────────────────────────

def _process_subject(
    fp: Path,
    short_sec: float,
    n_seeds: int,
    fc_thresh: float,
) -> list[dict] | None:
    """Run held-out validation for one subject across multiple seeds.

    Args:
        fp: Path to .npy timeseries file, shape (n_tp, n_rois).
        short_sec: Duration of segment A in seconds.
        n_seeds: Number of random seeds for bootstrap.
        fc_thresh: |FC| threshold for strong-connection subset.

    Returns:
        List of result dicts (one per seed), or None on failure.
    """
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    sub_id = fp.stem.split("_")[0]
    try:
        ts = np.load(fp).astype(np.float64)
    except Exception as exc:
        logger.warning(f"{sub_id}: load failed — {exc}")
        return None

    total_sec = ts.shape[0] * TR
    if total_sec < MIN_TOTAL_SEC:
        logger.info(f"{sub_id}: {total_sec:.0f}s < {MIN_TOTAL_SEC}s; skip")
        return None

    segs = _split_timeseries(ts, short_sec, TR)
    if segs is None:
        logger.warning(f"{sub_id}: cannot split; skip")
        return None

    ts_A, ts_B, ts_C = segs
    sec_A = ts_A.shape[0] * TR
    sec_B = ts_B.shape[0] * TR
    sec_C = ts_C.shape[0] * TR
    n_rois = ts.shape[1]

    # Pre-compute FC vectors for each segment
    fc_A = _pearson_fc_vec(ts_A)
    fc_B = _pearson_fc_vec(ts_B)
    fc_C = _pearson_fc_vec(ts_C)
    fc_BC = _pearson_fc_vec(np.concatenate([ts_B, ts_C], axis=0))

    # Strong-connection mask based on |FC_C|
    strong_mask = np.abs(fc_C) >= fc_thresh
    n_strong = int(strong_mask.sum())

    # Reference FC vector for BS-NET (LW shrinkage on B)
    fc_B_lw = get_fc_matrix(ts_B, vectorized=True, use_shrinkage=True)

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

        try:
            result = run_bootstrap_prediction(
                short_obs=ts_A,
                fc_reference=fc_B_lw,
                config=config,
                correction_method=CORRECTION_METHOD,
            )
            rho_hat_T = result.rho_hat_T
            ci_lower = result.ci_lower
            ci_upper = result.ci_upper
        except Exception as exc:
            logger.warning(f"{sub_id} seed={seed}: BS-NET failed — {exc}")
            rho_hat_T = float("nan")
            ci_lower = float("nan")
            ci_upper = float("nan")

        # Spearman correlations (all-pairs)
        r_fc_AC   = _spearman(fc_A, fc_C)
        r_fc_BC   = _spearman(fc_B, fc_C)
        r_fc_full = _spearman(fc_A, fc_BC)

        # Strong-connection subset
        if n_strong > 10:
            r_fc_AC_strong   = _spearman(fc_A[strong_mask],   fc_C[strong_mask])
            r_fc_BC_strong   = _spearman(fc_B[strong_mask],   fc_C[strong_mask])
            r_fc_full_strong = _spearman(fc_A[strong_mask],   fc_BC[strong_mask])
        else:
            r_fc_AC_strong   = float("nan")
            r_fc_BC_strong   = float("nan")
            r_fc_full_strong = float("nan")

        records.append({
            "sub_id":          sub_id,
            "seed":            seed,
            "n_rois":          n_rois,
            "sec_A":           sec_A,
            "sec_B":           sec_B,
            "sec_C":           sec_C,
            "total_sec":       total_sec,
            "n_pairs":         len(fc_A),
            "n_strong":        n_strong,
            "fc_thresh":       fc_thresh,
            # All-pairs
            "r_fc_AC":         r_fc_AC,
            "r_fc_BC":         r_fc_BC,
            "r_fc_full":       r_fc_full,
            "rho_hat_T":       rho_hat_T,
            "ci_lower":        ci_lower,
            "ci_upper":        ci_upper,
            # Strong-FC subset
            "r_fc_AC_strong":   r_fc_AC_strong,
            "r_fc_BC_strong":   r_fc_BC_strong,
            "r_fc_full_strong": r_fc_full_strong,
        })

    logger.info(
        f"{sub_id}: {total_sec:.0f}s | "
        f"r_AC={r_fc_AC:.3f} → ρ̂T={rho_hat_T:.3f} | "
        f"r_BC={r_fc_BC:.3f} r_full={r_fc_full:.3f}"
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
    """Run held-out prediction validation across all subjects.

    Args:
        atlas: Atlas directory name under CACHE_DIR.
        short_sec: Duration of short segment A in seconds.
        n_seeds: Number of random seeds.
        fc_thresh: |FC| threshold for strong-connection subset.
        n_jobs: Number of parallel workers.
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
    cols = ["r_fc_AC", "rho_hat_T", "r_fc_BC", "r_fc_full",
            "r_fc_AC_strong", "r_fc_full_strong"]
    agg = df.groupby("sub_id")[cols].mean()

    print(f"\n{'='*60}")
    print(f"Held-out Prediction Validation — atlas={atlas}")
    print(f"  Subjects   : {agg.shape[0]}")
    print(f"  Short scan : {short_sec:.0f}s (segment A)")
    print(f"  Threshold  : |FC| >= {fc_thresh}")
    print(f"{'='*60}")
    print(f"  All-pairs (mean across subjects × seeds):")
    print(f"    r_FC(A,C)   = {agg['r_fc_AC'].mean():.3f}  ← uncorrected baseline")
    print(f"    ρ̂T(A,B→C)  = {agg['rho_hat_T'].mean():.3f}  ← BS-NET correction")
    print(f"    r_FC(B,C)   = {agg['r_fc_BC'].mean():.3f}  ← same-subject ceiling")
    print(f"    r_FC(A,B+C) = {agg['r_fc_full'].mean():.3f}  ← ground truth (more data)")
    print(f"  Strong-FC subset (|FC_C|>={fc_thresh}):")
    print(f"    r_FC(A,C)   = {agg['r_fc_AC_strong'].mean():.3f}")
    print(f"    r_FC(A,B+C) = {agg['r_fc_full_strong'].mean():.3f}")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="BS-NET held-out prediction validation (3-segment design)"
    )
    parser.add_argument("--atlas",     default="4s256parcels",
                        help="Atlas key under data/ds000243/timeseries_cache_xcpd/")
    parser.add_argument("--short-sec", type=float, default=120.0,
                        help="Duration of short scan segment A in seconds (default: 120)")
    parser.add_argument("--n-seeds",   type=int,   default=10,
                        help="Number of random seeds per subject (default: 10)")
    parser.add_argument("--fc-thresh", type=float, default=0.20,
                        help="|FC| threshold for strong-connection subset (default: 0.20)")
    parser.add_argument("--n-jobs",    type=int,   default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--out-csv",   type=Path,  default=None,
                        help="Output CSV path (default: data/ds000243/results/held_out_validation_{atlas}.csv)")
    args = parser.parse_args()

    if args.out_csv is None:
        args.out_csv = (
            RESULTS_DIR / f"held_out_validation_{args.atlas}.csv"
        )

    run_held_out_validation(
        atlas=args.atlas,
        short_sec=args.short_sec,
        n_seeds=args.n_seeds,
        fc_thresh=args.fc_thresh,
        n_jobs=args.n_jobs,
        out_csv=args.out_csv,
    )
