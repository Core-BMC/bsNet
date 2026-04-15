#!/usr/bin/env python3
"""Convergence validation: r_FC(τ_ref) → ρ̂T as τ_ref increases.

Proves ρ̂T is a valid reliability estimate by showing that the observed
FC similarity (r_FC) asymptotically approaches ρ̂T as the reference
scan duration grows.

Experimental design (correct):
  - **Fix τ_short** (e.g., 60s, 120s, 180s)
  - **Vary τ_ref** from 30s to (total − τ_short)s, NON-OVERLAPPING
  - ts_short = ts[0 : short_vols]              — always the same
  - ts_ref   = ts[short_vols : short_vols + ref_vols]  — grows

Two-phase execution:
  Phase 1 (fast, ~5 min): r_FC at all (τ_short, τ_ref) combinations
      — deterministic, no bootstrap, no seeds
  Phase 2 (~30 min):      BS-NET ρ̂T at each τ_short (using max τ_ref)
      — bootstrap, multi-seed; computes the asymptotic ceiling

Expected plot:
  X-axis = τ_ref (reference duration)
  Y-axis = r_FC (observed FC similarity)
  Horizontal band = ρ̂T ± SD (BS-NET predicted ceiling)
  → r_FC curve rises toward ρ̂T as τ_ref → ∞

Usage:
    # Full run (49 subjects, ~30 min)
    python src/scripts/run_convergence_validation.py \\
        --dataset ds000243_xcpd --atlas 4s256parcels --n-jobs 8

    # Quick pilot (5 subjects, ~5 min)
    python src/scripts/run_convergence_validation.py \\
        --dataset ds000243_xcpd --atlas 4s256parcels \\
        --max-subjects 5 --n-jobs 4
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.data.data_loader import get_fc_matrix
from src.scripts.run_duration_sweep import (
    CORRECTION_METHOD,
    discover_subjects,
    _resolve_n_jobs,
)

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

# τ_short values to fix (seconds) — each generates one convergence curve
TAU_SHORT_LIST = [60, 120, 180]

# τ_ref grid (seconds) — reference durations to sweep
# Fine-grained at short refs, coarser at long refs
TAU_REF_LIST = [
    30, 60, 90, 120, 150, 180, 240, 300, 360, 450, 540, 660, 780,
]

N_BOOTSTRAPS = 100
DEFAULT_SEEDS = list(range(42, 52))  # 10 seeds for publication-quality


# ── Phase 1: r_FC sweep (no bootstrap) ──────────────────────────────────


def compute_rfc_sweep_fixed_short(
    sub: dict,
    tau_short_list: list[int],
    tau_ref_list: list[int],
) -> list[dict]:
    """Compute r_FC at all (τ_short, τ_ref) combinations for one subject.

    Design: τ_short fixed, τ_ref varies (non-overlapping, contiguous).
      ts_short = ts[0 : short_vols]
      ts_ref   = ts[short_vols : short_vols + ref_vols]

    Args:
        sub: Subject dict from discover_subjects().
        tau_short_list: List of short-scan durations to fix (seconds).
        tau_ref_list: List of reference durations to sweep (seconds).

    Returns:
        List of result dicts.
    """
    ts = np.load(sub["ts_path"]).astype(np.float64)
    tr = sub["tr"]
    n_vols = ts.shape[0]
    total_sec = n_vols * tr

    # Remove zero-variance ROIs
    valid_cols = np.std(ts, axis=0) > 1e-8
    if not np.all(valid_cols):
        ts = ts[:, valid_cols]
    n_rois = ts.shape[1]

    results = []
    for tau_short in tau_short_list:
        short_vols = int(tau_short / tr)
        if short_vols >= n_vols:
            continue

        # Compute fc_short ONCE per τ_short (it's always the same segment)
        ts_short = ts[:short_vols, :]
        fc_short_vec = get_fc_matrix(
            ts_short, vectorized=True, use_shrinkage=True, fisher_z=True,
        )

        max_ref_vols = n_vols - short_vols
        for tau_ref in tau_ref_list:
            ref_vols = int(tau_ref / tr)

            # Reference must fit within remaining data
            if ref_vols > max_ref_vols:
                continue
            # Minimum reference: 12 volumes
            if ref_vols < 12:
                continue

            ts_ref = ts[short_vols : short_vols + ref_vols, :]
            fc_ref_vec = get_fc_matrix(
                ts_ref, vectorized=True, use_shrinkage=True, fisher_z=True,
            )

            r_fc = float(np.corrcoef(fc_short_vec, fc_ref_vec)[0, 1])

            results.append({
                "sub_id": sub["sub_id"],
                "site": sub["site"],
                "tr": tr,
                "total_sec": total_sec,
                "tau_short_sec": tau_short,
                "tau_ref_sec": tau_ref,
                "ref_actual_sec": round(ref_vols * tr, 1),
                "n_rois": n_rois,
                "r_fc": round(r_fc, 6),
            })

    return results


def _rfc_worker(args: tuple) -> list[dict]:
    """Worker for Phase 1."""
    sub, tau_short_list, tau_ref_list = args
    return compute_rfc_sweep_fixed_short(sub, tau_short_list, tau_ref_list)


# ── Phase 2: BS-NET ρ̂T at each τ_short (max reference) ─────────────────


def compute_bsnet_ceiling(
    sub: dict,
    tau_short: int,
    seed: int,
    target_duration_min: float = 15.0,
) -> dict | None:
    """Run BS-NET for one τ_short using the maximum available reference.

    Reference = ts[short_vols:] (all remaining data after short segment).
    This gives the best possible ρ̂T estimate for this τ_short.

    Args:
        sub: Subject dict.
        tau_short: Short-scan duration in seconds.
        seed: Bootstrap random seed.
        target_duration_min: Target extrapolation duration (minutes).

    Returns:
        Result dict or None on failure.
    """
    ts = np.load(sub["ts_path"]).astype(np.float64)
    tr = sub["tr"]
    n_vols = ts.shape[0]

    valid_cols = np.std(ts, axis=0) > 1e-8
    if not np.all(valid_cols):
        ts = ts[:, valid_cols]
    n_rois = ts.shape[1]

    short_vols = int(tau_short / tr)
    if short_vols + 12 > n_vols:
        return None

    ts_short = ts[:short_vols, :]
    # Use ALL remaining data as reference (maximum τ_ref)
    ts_ref = ts[short_vols:, :]
    ref_vols = ts_ref.shape[0]
    ref_sec = round(ref_vols * tr, 1)

    fc_ref_vec = get_fc_matrix(
        ts_ref, vectorized=True, use_shrinkage=True, fisher_z=True,
    )

    config = BSNetConfig(
        n_rois=n_rois,
        tr=tr,
        short_duration_sec=tau_short,
        target_duration_min=target_duration_min,
        n_bootstraps=N_BOOTSTRAPS,
        seed=seed,
    )

    try:
        result = run_bootstrap_prediction(
            ts_short, fc_ref_vec, config,
            correction_method=CORRECTION_METHOD,
            fisher_z_fc=True,
        )
    except Exception as e:
        logger.debug(
            f"BS-NET failed: sub={sub['sub_id']}, tau_short={tau_short}s, "
            f"seed={seed}: {e}"
        )
        return None

    return {
        "sub_id": sub["sub_id"],
        "site": sub["site"],
        "tr": tr,
        "total_sec": round(n_vols * tr, 1),
        "tau_short_sec": tau_short,
        "ref_duration_sec": ref_sec,
        "seed": seed,
        "n_rois": n_rois,
        "rho_hat_T": round(float(result.rho_hat_T), 6),
        "ci_lower": round(float(result.ci_lower), 6),
        "ci_upper": round(float(result.ci_upper), 6),
    }


def _bsnet_worker(args: tuple) -> dict | None:
    """Worker for Phase 2."""
    sub, tau_short, seed, target_min = args
    return compute_bsnet_ceiling(sub, tau_short, seed, target_min)


# ── Main pipeline ────────────────────────────────────────────────────────


def run_convergence_validation(
    dataset: str = "ds000243_xcpd",
    atlas: str = "4s256parcels",
    tau_short_list: list[int] | None = None,
    tau_ref_list: list[int] | None = None,
    seeds: list[int] | None = None,
    min_total_sec: float = 600.0,
    max_subjects: int = 0,
    n_jobs: int = 1,
    target_duration_min: float = 15.0,
) -> tuple[Path, Path]:
    """Run two-phase convergence validation.

    Phase 1: r_FC at all (τ_short, τ_ref) — deterministic, fast.
    Phase 2: ρ̂T at each τ_short (max reference) — bootstrap, multi-seed.

    Args:
        dataset: Dataset name.
        atlas: Atlas name.
        tau_short_list: Short durations to fix (seconds).
        tau_ref_list: Reference durations to sweep (seconds).
        seeds: Random seeds for BS-NET.
        min_total_sec: Minimum total scan for inclusion.
        max_subjects: Limit (0 = all).
        n_jobs: Parallel workers.
        target_duration_min: Target extrapolation duration (minutes).

    Returns:
        Tuple of (rfc_csv_path, bsnet_csv_path).
    """
    if tau_short_list is None:
        tau_short_list = TAU_SHORT_LIST
    if tau_ref_list is None:
        tau_ref_list = TAU_REF_LIST
    if seeds is None:
        seeds = DEFAULT_SEEDS

    subjects = discover_subjects(dataset, atlas, min_total_sec, max_subjects)
    if not subjects:
        logger.error("No subjects found.")
        return Path("."), Path(".")

    n_sub = len(subjects)
    actual_jobs = _resolve_n_jobs(n_jobs)

    results_dataset = "ds000243" if dataset == "ds000243_xcpd" else dataset
    results_dir = Path(f"data/{results_dataset}/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════════
    # Phase 1: r_FC sweep (no bootstrap)
    # ════════════════════════════════════════════════════════════════════
    n_combos = len(tau_short_list) * len(tau_ref_list)
    print(f"\n{'='*65}")
    print(f"Phase 1: r_FC sweep")
    print(f"  {n_sub} subjects × {len(tau_short_list)} τ_short "
          f"× ≤{len(tau_ref_list)} τ_ref = ≤{n_sub * n_combos} records")
    print(f"{'='*65}")

    rfc_tasks = [(sub, tau_short_list, tau_ref_list) for sub in subjects]
    rfc_results: list[dict] = []
    t0 = time.time()

    if actual_jobs <= 1:
        for task in tqdm(rfc_tasks, desc="Phase 1 [r_FC]", unit="sub"):
            rows = _rfc_worker(task)
            rfc_results.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=actual_jobs) as pool:
            futures = {pool.submit(_rfc_worker, t): t for t in rfc_tasks}
            for future in tqdm(
                as_completed(futures),
                total=len(rfc_tasks),
                desc="Phase 1 [r_FC]",
                unit="sub",
            ):
                rows = future.result()
                rfc_results.extend(rows)

    t1 = time.time()
    print(f"Phase 1 done: {len(rfc_results)} records in {t1 - t0:.0f}s\n")

    # Save r_FC CSV
    rfc_csv = results_dir / f"{dataset}_convergence_rfc_{atlas}.csv"
    rfc_fields = [
        "sub_id", "site", "tr", "total_sec",
        "tau_short_sec", "tau_ref_sec", "ref_actual_sec",
        "n_rois", "r_fc",
    ]
    with open(rfc_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rfc_fields)
        writer.writeheader()
        writer.writerows(sorted(
            rfc_results,
            key=lambda r: (r["tau_short_sec"], r["tau_ref_sec"], r["sub_id"]),
        ))
    print(f"  r_FC CSV: {rfc_csv}")

    # ════════════════════════════════════════════════════════════════════
    # Phase 2: BS-NET ρ̂T at each τ_short (max reference)
    # ════════════════════════════════════════════════════════════════════
    n_bsnet = n_sub * len(tau_short_list) * len(seeds)
    print(f"\n{'='*65}")
    print(f"Phase 2: BS-NET ρ̂T (max reference)")
    print(f"  {n_sub} subjects × {len(tau_short_list)} τ_short "
          f"× {len(seeds)} seeds = {n_bsnet} runs")
    print(f"{'='*65}")

    bsnet_tasks = [
        (sub, tau_short, seed, target_duration_min)
        for sub in subjects
        for tau_short in tau_short_list
        for seed in seeds
    ]

    bsnet_results: list[dict] = []
    t2 = time.time()

    if actual_jobs <= 1:
        for task in tqdm(bsnet_tasks, desc="Phase 2 [BS-NET]", unit="run"):
            row = _bsnet_worker(task)
            if row is not None:
                bsnet_results.append(row)
    else:
        with ProcessPoolExecutor(max_workers=actual_jobs) as pool:
            futures = {pool.submit(_bsnet_worker, t): t for t in bsnet_tasks}
            for future in tqdm(
                as_completed(futures),
                total=len(bsnet_tasks),
                desc="Phase 2 [BS-NET]",
                unit="run",
            ):
                row = future.result()
                if row is not None:
                    bsnet_results.append(row)

    t3 = time.time()
    print(f"Phase 2 done: {len(bsnet_results)} records in {t3 - t2:.0f}s\n")

    # Save BS-NET CSV
    bsnet_csv = results_dir / f"{dataset}_convergence_bsnet_{atlas}.csv"
    bsnet_fields = [
        "sub_id", "site", "tr", "total_sec",
        "tau_short_sec", "ref_duration_sec", "seed", "n_rois",
        "rho_hat_T", "ci_lower", "ci_upper",
    ]
    with open(bsnet_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bsnet_fields)
        writer.writeheader()
        writer.writerows(sorted(
            bsnet_results,
            key=lambda r: (r["tau_short_sec"], r["sub_id"], r["seed"]),
        ))
    print(f"  BS-NET CSV: {bsnet_csv}")

    # ── Print summary ───────────────────────────────────────────────────
    total_elapsed = t3 - t0
    print(f"\n{'='*65}")
    print(f"Convergence Validation — {dataset} / {atlas}")
    print(f"  Total: {total_elapsed / 60:.1f} min "
          f"(Phase 1: {(t1 - t0) / 60:.1f}, Phase 2: {(t3 - t2) / 60:.1f})")
    print(f"{'='*65}")

    # r_FC summary per τ_short
    for tau_short in tau_short_list:
        print(f"\n  τ_short = {tau_short}s:")
        print(f"  {'τ_ref':>8} {'N':>5} {'r_FC mean':>10} {'r_FC SD':>9}")
        print(f"  {'-'*36}")
        for tau_ref in tau_ref_list:
            rows = [
                r for r in rfc_results
                if r["tau_short_sec"] == tau_short
                and r["tau_ref_sec"] == tau_ref
            ]
            if not rows:
                continue
            vals = [r["r_fc"] for r in rows]
            print(
                f"  {tau_ref:>6}s {len(rows):>5} "
                f"{np.mean(vals):>9.4f} {np.std(vals):>9.4f}"
            )

        # ρ̂T ceiling for this τ_short
        bsnet_rows = [
            r for r in bsnet_results
            if r["tau_short_sec"] == tau_short
        ]
        if bsnet_rows:
            by_sub: dict[str, list[float]] = {}
            for r in bsnet_rows:
                by_sub.setdefault(r["sub_id"], []).append(r["rho_hat_T"])
            sub_means = [float(np.mean(v)) for v in by_sub.values()]
            print(
                f"  {'ρ̂T':>8} {len(by_sub):>5} "
                f"{np.mean(sub_means):>9.4f} {np.std(sub_means):>9.4f}  "
                f"← ceiling"
            )

    return rfc_csv, bsnet_csv


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Convergence validation: fix τ_short, vary τ_ref, "
            "show r_FC → ρ̂T as τ_ref → ∞"
        ),
    )
    parser.add_argument(
        "--dataset", default="ds000243_xcpd",
        choices=["abide", "ds007535", "ds000243", "ds000243_xcpd"],
    )
    parser.add_argument("--atlas", default=None)
    parser.add_argument(
        "--tau-short", nargs="+", type=int, default=None,
        help=f"Fixed short durations (default: {TAU_SHORT_LIST})",
    )
    parser.add_argument(
        "--tau-ref", nargs="+", type=int, default=None,
        help=f"Reference durations to sweep (default: {TAU_REF_LIST})",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=3,
        help="Seeds for BS-NET (default: 3)",
    )
    parser.add_argument("--min-total-sec", type=float, default=600.0)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--target-duration-min", type=float, default=15.0)
    args = parser.parse_args()

    atlas = args.atlas
    if atlas is None:
        from src.scripts.run_duration_sweep import ATLAS_CHOICES
        atlas = ATLAS_CHOICES[args.dataset][0]

    seeds = list(range(42, 42 + args.n_seeds))

    run_convergence_validation(
        dataset=args.dataset,
        atlas=atlas,
        tau_short_list=args.tau_short,
        tau_ref_list=args.tau_ref,
        seeds=seeds,
        min_total_sec=args.min_total_sec,
        max_subjects=args.max_subjects,
        n_jobs=args.n_jobs,
        target_duration_min=args.target_duration_min,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
