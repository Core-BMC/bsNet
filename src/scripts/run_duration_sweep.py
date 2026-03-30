#!/usr/bin/env python3
"""Duration sweep: BS-NET performance across observation durations.

Runs BS-NET at multiple short-scan durations and records r_FC (raw) and
ρ̂T (BS-NET) vs full-scan reference FC.  Supports multiple datasets:

  - **abide**: ABIDE PCP (CC200/CC400), variable TR per site
  - **ds007535**: OpenNeuro SpeechHemi (Schaefer 200/400), TR=2.0s, ~15 min

Design:
  - Common subset: only subjects with total scan ≥ min_total_sec
  - Reference FC: full timeseries (constant across durations)
  - Multi-seed: 10 seeds for cross-seed ±SD estimation
  - Correction: fisher_z (confirmed method)

Usage:
    # ABIDE (same as previous run_abide_duration_sweep.py)
    python src/scripts/run_duration_sweep.py --dataset abide --atlas cc200

    # ds007535 (15-min task-residual, Schaefer 200)
    python src/scripts/run_duration_sweep.py --dataset ds007535 --atlas schaefer200

    # Quick test
    python src/scripts/run_duration_sweep.py --dataset ds007535 --max-subjects 5 --n-seeds 2

    # Parallel
    python src/scripts/run_duration_sweep.py --dataset ds007535 --n-jobs 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

# Default durations per dataset (seconds)
DURATIONS_ABIDE = [30, 60, 90, 120, 150, 180]
DURATIONS_DS007535 = [30, 60, 90, 120, 180, 240, 300, 450]

DEFAULT_SEEDS = list(range(42, 52))  # 10 seeds: 42–51
CORRECTION_METHOD = "fisher_z"
N_BOOTSTRAPS = 100

# ── Dataset configurations ──────────────────────────────────────────────

# ABIDE: site-specific TR (same as run_abide_bsnet.py)
TR_MAP_ABIDE = {
    "Caltech": 2.0, "CMU": 2.0, "KKI": 2.5, "Leuven": 1.667,
    "MaxMun": 3.0, "NYU": 2.0, "OHSU": 2.5, "Olin": 1.5,
    "Pitt": 1.5, "SBL": 2.2, "SDSU": 2.0, "Stanford": 2.0,
    "Trinity": 2.0, "UCLA": 3.0, "UM": 2.0, "USM": 2.0, "Yale": 2.0,
}

# ds007535: fixed TR (from dataset JSON sidecar)
TR_DS007535 = 2.0

# Atlas mappings per dataset
ATLAS_CHOICES = {
    "abide": ["cc200", "cc400"],
    "ds007535": ["schaefer200", "schaefer400"],
}


def _get_tr_abide(site_id: str) -> float:
    """Get approximate TR for ABIDE site."""
    for key, tr in TR_MAP_ABIDE.items():
        if key.lower() in site_id.lower():
            return tr
    return 2.0


# ── Subject discovery ────────────────────────────────────────────────────


def discover_subjects_abide(
    atlas: str,
    min_total_sec: float,
    max_subjects: int,
) -> list[dict]:
    """Discover cached ABIDE subjects meeting minimum duration.

    Args:
        atlas: Atlas name (cc200/cc400).
        min_total_sec: Minimum total scan duration in seconds.
        max_subjects: Limit (0 = all).

    Returns:
        List of subject dicts.
    """
    cache_dir = Path("data/abide/timeseries_cache") / atlas
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return []

    meta_path = Path("data/abide/results") / f"abide_subjects_{atlas}.json"
    site_map: dict[str, str] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        for entry in meta:
            site_map[str(entry.get("sub_id", ""))] = str(
                entry.get("site", "unknown")
            )

    subjects = []
    for npy_file in sorted(cache_dir.glob(f"*_{atlas}.npy")):
        sub_id = npy_file.stem.replace(f"_{atlas}", "")
        ts = np.load(npy_file)
        n_vols = ts.shape[0]
        site = site_map.get(sub_id, "unknown")
        tr = _get_tr_abide(site)
        total_sec = n_vols * tr

        if total_sec < min_total_sec:
            continue

        subjects.append({
            "sub_id": sub_id,
            "ts_path": str(npy_file),
            "site": site,
            "tr": tr,
            "n_vols": n_vols,
            "total_sec": total_sec,
            "n_rois": ts.shape[1],
        })

    if max_subjects > 0:
        subjects = subjects[:max_subjects]

    logger.info(
        f"[ABIDE] {len(subjects)} subjects with ≥{min_total_sec}s "
        f"(atlas={atlas})"
    )
    return subjects


def discover_subjects_ds007535(
    atlas: str,
    min_total_sec: float,
    max_subjects: int,
) -> list[dict]:
    """Discover preprocessed ds007535 subjects.

    Args:
        atlas: Atlas name (schaefer200/schaefer400).
        min_total_sec: Minimum total scan duration in seconds.
        max_subjects: Limit (0 = all).

    Returns:
        List of subject dicts.
    """
    cache_dir = Path("data/ds007535/timeseries_cache") / atlas
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return []

    tr = TR_DS007535
    subjects = []
    for npy_file in sorted(cache_dir.glob(f"*_{atlas}.npy")):
        sub_id = npy_file.stem.replace(f"_{atlas}", "")
        ts = np.load(npy_file)
        n_vols = ts.shape[0]
        total_sec = n_vols * tr

        if total_sec < min_total_sec:
            continue

        subjects.append({
            "sub_id": sub_id,
            "ts_path": str(npy_file),
            "site": "ds007535",
            "tr": tr,
            "n_vols": n_vols,
            "total_sec": total_sec,
            "n_rois": ts.shape[1],
        })

    if max_subjects > 0:
        subjects = subjects[:max_subjects]

    logger.info(
        f"[ds007535] {len(subjects)} subjects with ≥{min_total_sec}s "
        f"(atlas={atlas})"
    )
    return subjects


def discover_subjects(
    dataset: str,
    atlas: str,
    min_total_sec: float,
    max_subjects: int = 0,
) -> list[dict]:
    """Route to dataset-specific discovery function.

    Args:
        dataset: Dataset name ("abide" or "ds007535").
        atlas: Atlas name.
        min_total_sec: Minimum total scan duration.
        max_subjects: Limit (0 = all).

    Returns:
        List of subject dicts.
    """
    if dataset == "abide":
        return discover_subjects_abide(atlas, min_total_sec, max_subjects)
    if dataset == "ds007535":
        return discover_subjects_ds007535(atlas, min_total_sec, max_subjects)
    raise ValueError(f"Unknown dataset: {dataset}")


# ── Single-subject sweep ─────────────────────────────────────────────────


def sweep_single_subject(
    sub: dict,
    durations: list[int],
    seed: int,
    target_duration_min: float = 15.0,
) -> list[dict]:
    """Run BS-NET at multiple durations for one subject × one seed.

    Args:
        sub: Subject dict from discover_subjects().
        durations: List of short-scan durations in seconds.
        seed: Bootstrap random seed.
        target_duration_min: Target extrapolation duration (minutes).

    Returns:
        List of result dicts (one per duration).
    """
    ts = np.load(sub["ts_path"]).astype(np.float64)
    tr = sub["tr"]
    n_vols = ts.shape[0]
    n_rois = ts.shape[1]

    # Remove zero-variance ROIs
    valid_cols = np.std(ts, axis=0) > 1e-8
    if not np.all(valid_cols):
        ts = ts[:, valid_cols]
        n_rois = ts.shape[1]

    # Reference FC: full timeseries
    fc_full_vec = get_fc_matrix(ts, vectorized=True, use_shrinkage=True)

    results = []
    for dur_sec in durations:
        short_vols = int(dur_sec / tr)

        if short_vols + 10 > n_vols:
            continue

        ts_short = ts[:short_vols, :]
        fc_short_vec = get_fc_matrix(
            ts_short, vectorized=True, use_shrinkage=True
        )

        # Raw FC similarity
        r_fc_raw = float(np.corrcoef(fc_short_vec, fc_full_vec)[0, 1])

        # BS-NET pipeline
        config = BSNetConfig(
            n_rois=n_rois,
            tr=tr,
            short_duration_sec=dur_sec,
            target_duration_min=target_duration_min,
            n_bootstraps=N_BOOTSTRAPS,
            seed=seed,
        )

        try:
            result = run_bootstrap_prediction(
                ts_short, fc_full_vec, config,
                correction_method=CORRECTION_METHOD,
            )
            rho_hat_T = float(result.rho_hat_T)
            ci_lower = float(result.ci_lower)
            ci_upper = float(result.ci_upper)
        except Exception as e:
            logger.debug(
                f"BS-NET failed: sub={sub['sub_id']}, dur={dur_sec}s, "
                f"seed={seed}: {e}"
            )
            continue

        results.append({
            "sub_id": sub["sub_id"],
            "site": sub["site"],
            "tr": tr,
            "total_sec": sub["total_sec"],
            "duration_sec": dur_sec,
            "seed": seed,
            "n_rois": n_rois,
            "r_fc_raw": round(r_fc_raw, 6),
            "rho_hat_T": round(rho_hat_T, 6),
            "ci_lower": round(ci_lower, 6),
            "ci_upper": round(ci_upper, 6),
            "improvement": round(rho_hat_T - r_fc_raw, 6),
        })

    return results


# ── Parallel worker ──────────────────────────────────────────────────────


def _worker(args: tuple) -> list[dict]:
    """Worker for ProcessPoolExecutor."""
    sub, durations, seed, target_min = args
    return sweep_single_subject(sub, durations, seed, target_min)


def _resolve_n_jobs(n_jobs: int) -> int:
    """Resolve n_jobs to actual worker count."""
    cpu_count = os.cpu_count() or 1
    if n_jobs == -1:
        return cpu_count
    if n_jobs <= 0:
        return max(1, cpu_count + n_jobs)
    return min(n_jobs, cpu_count)


# ── Aggregation ──────────────────────────────────────────────────────────


def _save_aggregated(
    all_results: list[dict],
    path: Path,
    durations: list[int],
    seeds: list[int],
) -> None:
    """Save duration × seed aggregated statistics.

    For each (duration, seed): mean across subjects.
    Then for each duration: mean ± SD across seeds.
    """
    seed_level: dict[tuple[int, int], dict[str, list[float]]] = {}
    for r in all_results:
        key = (r["duration_sec"], r["seed"])
        if key not in seed_level:
            seed_level[key] = {
                "r_fc": [], "rho": [], "improvement": [], "ci_width": [],
            }
        seed_level[key]["r_fc"].append(r["r_fc_raw"])
        seed_level[key]["rho"].append(r["rho_hat_T"])
        seed_level[key]["improvement"].append(r["improvement"])
        seed_level[key]["ci_width"].append(r["ci_upper"] - r["ci_lower"])

    rows = []
    for dur in durations:
        seed_means: dict[str, list[float]] = {
            "r_fc": [], "rho": [], "improvement": [], "ci_width": [],
        }
        n_subjects = 0
        for seed in seeds:
            key = (dur, seed)
            if key not in seed_level:
                continue
            for metric in seed_means:
                seed_means[metric].append(
                    float(np.mean(seed_level[key][metric]))
                )
            n_subjects = len(seed_level[key]["r_fc"])

        if not seed_means["rho"]:
            continue

        rows.append({
            "duration_sec": dur,
            "n_subjects": n_subjects,
            "n_seeds": len(seed_means["rho"]),
            "r_fc_mean": round(float(np.mean(seed_means["r_fc"])), 6),
            "r_fc_sd": round(float(np.std(seed_means["r_fc"])), 6),
            "rho_hat_T_mean": round(float(np.mean(seed_means["rho"])), 6),
            "rho_hat_T_sd": round(float(np.std(seed_means["rho"])), 6),
            "improvement_mean": round(
                float(np.mean(seed_means["improvement"])), 6
            ),
            "improvement_sd": round(
                float(np.std(seed_means["improvement"])), 6
            ),
            "ci_width_mean": round(
                float(np.mean(seed_means["ci_width"])), 6
            ),
            "ci_width_sd": round(
                float(np.std(seed_means["ci_width"])), 6
            ),
        })

    fieldnames = [
        "duration_sec", "n_subjects", "n_seeds",
        "r_fc_mean", "r_fc_sd", "rho_hat_T_mean", "rho_hat_T_sd",
        "improvement_mean", "improvement_sd",
        "ci_width_mean", "ci_width_sd",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── Main pipeline ────────────────────────────────────────────────────────


def run_duration_sweep(
    dataset: str = "ds007535",
    atlas: str = "schaefer200",
    durations: list[int] | None = None,
    seeds: list[int] | None = None,
    min_total_sec: float = 360.0,
    max_subjects: int = 0,
    n_jobs: int = 1,
    target_duration_min: float = 15.0,
) -> Path:
    """Run duration sweep experiment.

    Args:
        dataset: Dataset name ("abide" or "ds007535").
        atlas: Atlas name.
        durations: List of short-scan durations (seconds).
        seeds: List of random seeds.
        min_total_sec: Minimum total scan for inclusion.
        max_subjects: Limit (0 = all).
        n_jobs: Parallel workers (1=sequential, -1=all cores).
        target_duration_min: Target extrapolation duration (minutes).

    Returns:
        Path to per-record CSV.
    """
    if durations is None:
        durations = (
            DURATIONS_DS007535 if dataset == "ds007535" else DURATIONS_ABIDE
        )
    if seeds is None:
        seeds = DEFAULT_SEEDS

    subjects = discover_subjects(dataset, atlas, min_total_sec, max_subjects)
    if not subjects:
        logger.error("No subjects found.")
        return Path(".")

    n_sub = len(subjects)
    n_seeds = len(seeds)
    n_dur = len(durations)
    total_tasks = n_sub * n_seeds
    logger.info(
        f"Duration sweep [{dataset}]: {n_sub} subjects × {n_seeds} seeds × "
        f"{n_dur} durations = {n_sub * n_seeds * n_dur} runs"
    )

    tasks = [
        (sub, durations, seed, target_duration_min)
        for sub in subjects
        for seed in seeds
    ]

    all_results: list[dict] = []
    t0 = time.time()
    actual_jobs = _resolve_n_jobs(n_jobs)

    if actual_jobs <= 1:
        for idx, task in enumerate(tasks):
            rows = _worker(task)
            all_results.extend(rows)
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                pct = (idx + 1) / total_tasks * 100
                logger.info(
                    f"  [{idx + 1}/{total_tasks}] {pct:.0f}% "
                    f"({elapsed:.0f}s elapsed)"
                )
    else:
        logger.info(f"Using {actual_jobs} parallel workers")
        with ProcessPoolExecutor(max_workers=actual_jobs) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for done_count, future in enumerate(
                as_completed(futures), start=1
            ):
                rows = future.result()
                all_results.extend(rows)
                if done_count % 100 == 0:
                    elapsed = time.time() - t0
                    pct = done_count / total_tasks * 100
                    logger.info(
                        f"  [{done_count}/{total_tasks}] {pct:.0f}% "
                        f"({elapsed:.0f}s elapsed)"
                    )

    elapsed_total = time.time() - t0
    logger.info(
        f"Completed {len(all_results)} records in {elapsed_total:.0f}s"
    )

    # ── Save per-record CSV ──────────────────────────────────────────────
    results_dir = Path(f"data/{dataset}/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"{dataset}_duration_sweep_{atlas}.csv"
    fieldnames = [
        "sub_id", "site", "tr", "total_sec", "duration_sec", "seed",
        "n_rois", "r_fc_raw", "rho_hat_T", "ci_lower", "ci_upper",
        "improvement",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(all_results, key=lambda r: (
            r["duration_sec"], r["sub_id"], r["seed"]
        )))
    logger.info(f"Per-record CSV: {csv_path}")

    # ── Aggregate ────────────────────────────────────────────────────────
    agg_path = (
        results_dir / f"{dataset}_duration_sweep_{atlas}_aggregated.csv"
    )
    _save_aggregated(all_results, agg_path, durations, seeds)
    logger.info(f"Aggregated CSV: {agg_path}")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Duration Sweep — {dataset.upper()} / {atlas.upper()}")
    print(f"  Subjects: {n_sub}, Seeds: {n_seeds}, Durations: {durations}")
    print(f"  Target: {target_duration_min} min, Elapsed: {elapsed_total:.0f}s")
    print(f"{'='*65}")
    print(f"{'Duration':>10} {'N':>5} {'r_FC':>12} {'ρ̂T':>12} "
          f"{'Δ':>10} {'CI_width':>10}")
    print(f"{'-'*65}")

    for dur in durations:
        dur_rows = [r for r in all_results if r["duration_sec"] == dur]
        if not dur_rows:
            continue
        r_fc_vals = [r["r_fc_raw"] for r in dur_rows]
        rho_vals = [r["rho_hat_T"] for r in dur_rows]
        imp_vals = [r["improvement"] for r in dur_rows]
        ci_widths = [r["ci_upper"] - r["ci_lower"] for r in dur_rows]
        n_unique_subs = len({r["sub_id"] for r in dur_rows})
        print(
            f"{dur:>8}s {n_unique_subs:>5} "
            f"{np.mean(r_fc_vals):>6.3f}±{np.std(r_fc_vals):.3f} "
            f"{np.mean(rho_vals):>6.3f}±{np.std(rho_vals):.3f} "
            f"{np.mean(imp_vals):>+7.3f} "
            f"{np.mean(ci_widths):>7.3f}"
        )

    return csv_path


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Duration sweep for BS-NET Figure 1 (multi-dataset)"
    )
    parser.add_argument(
        "--dataset", default="ds007535",
        choices=["abide", "ds007535"],
        help="Dataset to use (default: ds007535)",
    )
    parser.add_argument(
        "--atlas", default=None,
        help="Atlas name (default: per-dataset, e.g. schaefer200 for ds007535)",
    )
    parser.add_argument(
        "--durations", nargs="+", type=int, default=None,
        help="Short-scan durations in seconds (default: per-dataset)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of bootstrap seeds (default: 10)",
    )
    parser.add_argument(
        "--min-total-sec", type=float, default=0,
        help="Minimum total scan duration (default: per-dataset)",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=0,
        help="Limit subjects (0 = all)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel workers (1=sequential, -1=all cores)",
    )
    parser.add_argument(
        "--target-duration-min", type=float, default=15.0,
        help="Target extrapolation duration in minutes (default: 15)",
    )
    args = parser.parse_args()

    # Per-dataset defaults
    atlas = args.atlas
    if atlas is None:
        atlas = ATLAS_CHOICES[args.dataset][0]  # first default
    min_total_sec = args.min_total_sec
    if min_total_sec <= 0:
        min_total_sec = 360.0 if args.dataset == "abide" else 600.0

    # Validate atlas
    valid = ATLAS_CHOICES[args.dataset]
    if atlas not in valid:
        parser.error(f"Atlas '{atlas}' not valid for {args.dataset}. "
                     f"Choose from: {valid}")

    seeds = list(range(42, 42 + args.n_seeds))

    run_duration_sweep(
        dataset=args.dataset,
        atlas=atlas,
        durations=args.durations,
        seeds=seeds,
        min_total_sec=min_total_sec,
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
