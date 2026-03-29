"""
Component Necessity Analysis for BS-NET.

Leave-one-out experiment that measures the degradation caused by removing
one component at a time from the full BS-NET pipeline. Conditions tested:
- L_full: Full pipeline (baseline)
- L_no_sb: Remove Spearman-Brown scaling
- L_no_lw: Remove Ledoit-Wolf shrinkage
- L_no_boot: Remove bootstrap resampling
- L_no_prior: Remove Bayesian prior
- L_no_atten: Remove attenuation correction entirely

Supports:
- Single .npy input (--input-npy)
- Batch directory input (--input-dir) with optional subject sampling
- Synthetic data (default, when no input specified)
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.core.bootstrap import (
    block_bootstrap_indices,
    correct_attenuation,
    estimate_optimal_block_length,
    fisher_z,
    fisher_z_inv,
    spearman_brown,
)
from src.core.pipeline import compute_split_half_reliability
from src.data.data_loader import get_fc_matrix, load_timeseries_data

logger = logging.getLogger(__name__)


class ConditionResult(NamedTuple):
    """Result for a single condition and seed."""

    condition: str
    seed: int
    rho_hat_T: float
    subject_id: str = "synthetic"


# ---------------------------------------------------------------------------
# Pipeline variants (each accepts k as parameter)
# ---------------------------------------------------------------------------


def run_full_pipeline(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    block_size: int,
    k: float = 7.5,
    n_bootstraps: int = 50,
    method: str = "fisher_z",
) -> float:
    """Full BS-NET pipeline (L_full)."""
    t_samples = short_obs.shape[0]
    rho_hat_b = []
    empirical_prior = (0.25, 0.05)

    for _ in range(n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs_t = np.corrcoef(fc_reference, fc_obs_t)[0, 1]
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=True)
        rho_est_T = correct_attenuation(
            r_obs_t, 0.98, r_split_t, k=k,
            empirical_prior=empirical_prior, method=method,
        )
        rho_hat_b.append(fisher_z(rho_est_T))

    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


def run_no_sb_pipeline(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    block_size: int,
    k: float = 7.5,
    n_bootstraps: int = 50,
    method: str = "fisher_z",
) -> float:
    """Remove Spearman-Brown scaling (L_no_sb): k=1.0."""
    t_samples = short_obs.shape[0]
    rho_hat_b = []
    empirical_prior = (0.25, 0.05)

    for _ in range(n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs_t = np.corrcoef(fc_reference, fc_obs_t)[0, 1]
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=True)
        rho_est_T = correct_attenuation(
            r_obs_t, 0.98, r_split_t, k=1.0,
            empirical_prior=empirical_prior, method=method,
        )
        rho_hat_b.append(fisher_z(rho_est_T))

    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


def run_no_lw_pipeline(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    block_size: int,
    k: float = 7.5,
    n_bootstraps: int = 50,
    method: str = "fisher_z",
) -> float:
    """Remove Ledoit-Wolf shrinkage (L_no_lw): use_shrinkage=False."""
    t_samples = short_obs.shape[0]
    rho_hat_b = []
    empirical_prior = (0.25, 0.05)

    for _ in range(n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=False)
        r_obs_t = np.corrcoef(fc_reference, fc_obs_t)[0, 1]
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=False)
        rho_est_T = correct_attenuation(
            r_obs_t, 0.98, r_split_t, k=k,
            empirical_prior=empirical_prior, method=method,
        )
        rho_hat_b.append(fisher_z(rho_est_T))

    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


def run_no_boot_pipeline(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    k: float = 7.5,
    method: str = "fisher_z",
) -> float:
    """Remove bootstrap resampling (L_no_boot): single sample."""
    empirical_prior = (0.25, 0.05)
    fc_obs_t = get_fc_matrix(short_obs, vectorized=True, use_shrinkage=True)
    r_obs_t = np.corrcoef(fc_reference, fc_obs_t)[0, 1]
    r_split_t = compute_split_half_reliability(short_obs, use_shrinkage=True)
    rho_est_T = correct_attenuation(
        r_obs_t, 0.98, r_split_t, k=k,
        empirical_prior=empirical_prior, method=method,
    )
    z = fisher_z(rho_est_T)
    return float(fisher_z_inv(z))


def run_no_prior_pipeline(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    block_size: int,
    k: float = 7.5,
    n_bootstraps: int = 50,
    method: str = "fisher_z",
) -> float:
    """Remove Bayesian prior (L_no_prior): empirical_prior=None."""
    t_samples = short_obs.shape[0]
    rho_hat_b = []

    for _ in range(n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs_t = np.corrcoef(fc_reference, fc_obs_t)[0, 1]
        r_split_t = compute_split_half_reliability(ts_b, use_shrinkage=True)
        rho_est_T = correct_attenuation(
            r_obs_t, 0.98, r_split_t, k=k,
            empirical_prior=None, method=method,
        )
        rho_hat_b.append(fisher_z(rho_est_T))

    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


def run_no_atten_pipeline(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    block_size: int,
    k: float = 7.5,
    n_bootstraps: int = 50,
) -> float:
    """Remove attenuation correction entirely (L_no_atten): raw + SB only."""
    t_samples = short_obs.shape[0]
    rho_hat_b = []

    for _ in range(n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]
        fc_obs_t = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_raw = np.corrcoef(fc_reference, fc_obs_t)[0, 1]
        r_scaled = spearman_brown(r_raw, k)
        rho_hat_b.append(fisher_z(r_scaled))

    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


# ---------------------------------------------------------------------------
# Worker for multiprocessing (batch mode)
# ---------------------------------------------------------------------------

CONDITIONS = [
    "L_full", "L_no_sb", "L_no_lw", "L_no_boot", "L_no_prior", "L_no_atten",
]


def _subject_worker(args: tuple) -> list[ConditionResult]:
    """Process one subject × one seed across all 6 conditions.

    Args:
        args: (npy_path, seed, short_samples, n_bootstraps, correction_method)

    Returns:
        List of 6 ConditionResult (one per condition).
    """
    npy_path, seed, short_samples, n_bootstraps, correction_method = args

    subject_id = Path(npy_path).stem  # e.g., "50030_cc200"

    # Set seed for reproducibility
    np.random.seed(seed)

    # Load data
    ts_full, short_obs, ts_signal = load_timeseries_data(
        input_npy=str(npy_path),
        short_samples=short_samples,
    )

    # Reference FC from full time series (with shrinkage for real data)
    fc_reference = get_fc_matrix(ts_signal, vectorized=True, use_shrinkage=True)

    # Dynamic k: total TRs / short TRs
    k = ts_full.shape[0] / short_samples

    # Estimate block size
    block_size = estimate_optimal_block_length(short_obs)

    results = []

    # L_full
    rho = run_full_pipeline(
        short_obs, fc_reference, block_size, k=k,
        n_bootstraps=n_bootstraps, method=correction_method,
    )
    results.append(ConditionResult("L_full", seed, rho, subject_id))

    # L_no_sb
    rho = run_no_sb_pipeline(
        short_obs, fc_reference, block_size, k=k,
        n_bootstraps=n_bootstraps, method=correction_method,
    )
    results.append(ConditionResult("L_no_sb", seed, rho, subject_id))

    # L_no_lw
    rho = run_no_lw_pipeline(
        short_obs, fc_reference, block_size, k=k,
        n_bootstraps=n_bootstraps, method=correction_method,
    )
    results.append(ConditionResult("L_no_lw", seed, rho, subject_id))

    # L_no_boot
    rho = run_no_boot_pipeline(
        short_obs, fc_reference, k=k, method=correction_method,
    )
    results.append(ConditionResult("L_no_boot", seed, rho, subject_id))

    # L_no_prior
    rho = run_no_prior_pipeline(
        short_obs, fc_reference, block_size, k=k,
        n_bootstraps=n_bootstraps, method=correction_method,
    )
    results.append(ConditionResult("L_no_prior", seed, rho, subject_id))

    # L_no_atten
    rho = run_no_atten_pipeline(
        short_obs, fc_reference, block_size, k=k,
        n_bootstraps=n_bootstraps,
    )
    results.append(ConditionResult("L_no_atten", seed, rho, subject_id))

    return results


def _synthetic_worker(args: tuple) -> list[ConditionResult]:
    """Process one seed with synthetic data across all 6 conditions.

    Args:
        args: (seed, n_rois, n_samples, short_samples, noise_level, ar1,
               n_bootstraps, correction_method)

    Returns:
        List of 6 ConditionResult (one per condition).
    """
    (
        seed, n_rois, n_samples, short_samples,
        noise_level, ar1, n_bootstraps, correction_method,
    ) = args

    ts_full, short_obs, ts_signal = load_timeseries_data(
        input_npy=None,
        n_samples=n_samples,
        n_rois=n_rois,
        noise_level=noise_level,
        ar1=ar1,
        short_samples=short_samples,
        seed=seed,
    )

    fc_reference = get_fc_matrix(ts_signal, vectorized=True, use_shrinkage=False)
    block_size = estimate_optimal_block_length(short_obs)
    k = n_samples / short_samples

    results = []

    rho = run_full_pipeline(
        short_obs, fc_reference, block_size, k=k,
        n_bootstraps=n_bootstraps, method=correction_method,
    )
    results.append(ConditionResult("L_full", seed, rho, "synthetic"))

    rho = run_no_sb_pipeline(
        short_obs, fc_reference, block_size, k=k,
        n_bootstraps=n_bootstraps, method=correction_method,
    )
    results.append(ConditionResult("L_no_sb", seed, rho, "synthetic"))

    rho = run_no_lw_pipeline(
        short_obs, fc_reference, block_size, k=k,
        n_bootstraps=n_bootstraps, method=correction_method,
    )
    results.append(ConditionResult("L_no_lw", seed, rho, "synthetic"))

    rho = run_no_boot_pipeline(
        short_obs, fc_reference, k=k, method=correction_method,
    )
    results.append(ConditionResult("L_no_boot", seed, rho, "synthetic"))

    rho = run_no_prior_pipeline(
        short_obs, fc_reference, block_size, k=k,
        n_bootstraps=n_bootstraps, method=correction_method,
    )
    results.append(ConditionResult("L_no_prior", seed, rho, "synthetic"))

    rho = run_no_atten_pipeline(
        short_obs, fc_reference, block_size, k=k,
        n_bootstraps=n_bootstraps,
    )
    results.append(ConditionResult("L_no_atten", seed, rho, "synthetic"))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse CLI arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Component necessity analysis (leave-one-out)",
    )
    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-npy",
        type=str,
        default=None,
        help="Single .npy timeseries (n_samples, n_rois).",
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory of .npy timeseries files for batch processing.",
    )

    parser.add_argument(
        "--n-subjects",
        type=int,
        default=50,
        help="Max subjects to sample from --input-dir (default: 50). "
        "Use 0 for all subjects.",
    )
    parser.add_argument(
        "--short-samples",
        type=int,
        default=None,
        help="Number of TRs for short observation. "
        "Default: 120 (synthetic), 60 (real data, ~120s at TR=2).",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=50,
        help="Bootstrap iterations per condition (default: 50).",
    )
    parser.add_argument(
        "--correction-method",
        type=str,
        choices=["original", "fisher_z", "partial", "soft_clamp"],
        default="fisher_z",
        help="Attenuation correction method (default: fisher_z).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: artifacts/reports/component_necessity.csv",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Execute component necessity analysis across all conditions."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    seeds = [42, 123, 777, 2026, 9999, 314, 628, 1414, 2718, 3141]
    n_bootstraps = args.n_bootstraps
    correction_method = args.correction_method

    output_dir = Path("artifacts/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ConditionResult] = []

    # -----------------------------------------------------------------------
    # Mode 1: Batch directory (--input-dir)
    # -----------------------------------------------------------------------
    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
        npy_files = sorted(input_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files in {input_dir}")

        # Sample subjects
        n_subjects = args.n_subjects if args.n_subjects > 0 else len(npy_files)
        if n_subjects < len(npy_files):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(npy_files), size=n_subjects, replace=False)
            npy_files = [npy_files[i] for i in sorted(indices)]

        short_samples = args.short_samples if args.short_samples else 60
        total_tasks = len(npy_files) * len(seeds)

        logger.info(
            f"Batch mode: {len(npy_files)} subjects × {len(seeds)} seeds "
            f"× 6 conditions = {total_tasks * 6} runs"
        )
        logger.info(
            f"short_samples={short_samples}, n_bootstraps={n_bootstraps}, "
            f"method={correction_method}, n_jobs={args.n_jobs}"
        )

        # Build task list: (npy_path, seed, short_samples, n_bootstraps, method)
        tasks = [
            (npy_path, seed, short_samples, n_bootstraps, correction_method)
            for npy_path in npy_files
            for seed in seeds
        ]

        if args.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
                futures = {
                    executor.submit(_subject_worker, task): task
                    for task in tasks
                }
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        results.extend(future.result())
                    except Exception as e:
                        task = futures[future]
                        logger.error(f"Failed: {task[0].name} seed={task[1]}: {e}")
                    if i % 50 == 0:
                        logger.info(f"Progress: {i}/{total_tasks} tasks")
        else:
            for i, task in enumerate(tasks):
                try:
                    results.extend(_subject_worker(task))
                except Exception as e:
                    logger.error(f"Failed: {task[0].name} seed={task[1]}: {e}")
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{total_tasks} tasks")

        # Output filename
        if args.output:
            output_file = Path(args.output)
        else:
            atlas_tag = input_dir.name  # e.g., "cc200"
            output_file = (
                output_dir
                / f"component_necessity_ABIDE_{atlas_tag}_N{len(npy_files)}.csv"
            )

    # -----------------------------------------------------------------------
    # Mode 2: Single .npy (--input-npy)
    # -----------------------------------------------------------------------
    elif args.input_npy is not None:
        short_samples = args.short_samples if args.short_samples else 60
        logger.info(
            f"Single-file mode: {args.input_npy}, "
            f"short_samples={short_samples}, {len(seeds)} seeds × 6 conditions"
        )

        tasks = [
            (args.input_npy, seed, short_samples, n_bootstraps, correction_method)
            for seed in seeds
        ]

        for task in tasks:
            results.extend(_subject_worker(task))

        if args.output:
            output_file = Path(args.output)
        else:
            stem = Path(args.input_npy).stem
            output_file = output_dir / f"component_necessity_{stem}.csv"

    # -----------------------------------------------------------------------
    # Mode 3: Synthetic (default)
    # -----------------------------------------------------------------------
    else:
        n_rois = 50
        short_samples = args.short_samples if args.short_samples else 120
        n_samples = 900
        noise_level = 0.25
        ar1 = 0.6

        logger.info(
            f"Synthetic mode: n_rois={n_rois}, n_samples={n_samples}, "
            f"short={short_samples}, {len(seeds)} seeds × 6 conditions"
        )

        tasks = [
            (
                seed, n_rois, n_samples, short_samples,
                noise_level, ar1, n_bootstraps, correction_method,
            )
            for seed in seeds
        ]

        if args.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
                futures = {
                    executor.submit(_synthetic_worker, task): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    try:
                        results.extend(future.result())
                    except Exception as e:
                        logger.error(f"Synthetic worker failed: {e}")
        else:
            for task in tasks:
                results.extend(_synthetic_worker(task))

        if args.output:
            output_file = Path(args.output)
        else:
            output_file = output_dir / "component_necessity.csv"

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    full_by_key: dict[tuple[str, int], float] = {}
    for r in results:
        if r.condition == "L_full":
            full_by_key[(r.subject_id, r.seed)] = r.rho_hat_T

    with open(output_file, "w") as f:
        f.write("subject_id,condition,seed,rho_hat_T,delta_from_full\n")
        for r in results:
            full_val = full_by_key.get((r.subject_id, r.seed), np.nan)
            delta = r.rho_hat_T - full_val
            f.write(
                f"{r.subject_id},{r.condition},{r.seed},"
                f"{r.rho_hat_T:.6f},{delta:.6f}\n"
            )

    logger.info(f"Results saved to {output_file}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Component Necessity Analysis Summary")
    print("=" * 80)

    n_subjects_actual = len({r.subject_id for r in results})
    n_seeds_actual = len({r.seed for r in results})
    print(
        f"Subjects: {n_subjects_actual}, Seeds: {n_seeds_actual}, "
        f"Method: {correction_method}"
    )
    print("-" * 80)
    print(f"{'Condition':<15} {'Mean ρ̂T':<14} {'Std':<10} {'Δ from Full':<12}")
    print("-" * 80)

    full_values = [r.rho_hat_T for r in results if r.condition == "L_full"]
    mean_full = np.mean(full_values) if full_values else 0.0

    for condition in CONDITIONS:
        cond_values = [r.rho_hat_T for r in results if r.condition == condition]
        if not cond_values:
            continue
        mean_rho = np.mean(cond_values)
        std_rho = np.std(cond_values)
        delta = mean_rho - mean_full
        print(f"{condition:<15} {mean_rho:.4f} ± {std_rho:.4f}   {delta:+.4f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
