"""
Progressive Ablation Study for BS-NET (Real Data).

Cumulative component addition experiment showing how each pipeline stage
incrementally improves the reliability estimate. Complements the leave-one-out
analysis (run_component_necessity.py) by showing the *building-up* perspective.

Levels:
    L0 (Raw):           r_FC = corr(FC_short, FC_ref), no shrinkage
    L1 (+LW):           r_FC with Ledoit-Wolf shrinkage
    L2 (+Bootstrap):    Median of block-bootstrap r_FC samples (with LW)
    L3 (+SB+Atten):     Attenuation correction with SB prophecy (k=actual, no prior)
    L4 (+Prior):        Add Bayesian empirical prior
    L5 (Full BS-NET):   Fisher z-space aggregation (full pipeline)

Supports:
    --input-dir     Batch directory of .npy timeseries
    --input-npy     Single .npy file
    --dry-run       Quick test: 2 subjects × 2 seeds × 3 bootstraps
    --n-jobs        Parallel workers (set BLAS threading vars!)

Output CSV columns:
    subject_id, level, level_name, seed, rho_hat_T, delta_from_raw
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tqdm import tqdm

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


# ============================================================================
# Result container
# ============================================================================

class LevelResult(NamedTuple):
    """Result for a single level × subject × seed."""

    subject_id: str
    level: str
    level_name: str
    seed: int
    rho_hat_T: float


# ============================================================================
# Level names (for CSV and figures)
# ============================================================================

LEVELS = [
    ("L0", "Raw"),
    ("L1", "+LW"),
    ("L2", "+Bootstrap"),
    ("L3", "+SB+Atten"),
    ("L4", "+Prior"),
    ("L5", "Full BS-NET"),
]


# ============================================================================
# Progressive ablation functions
# ============================================================================

def level_l0_raw(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
) -> float:
    """L0 (Raw): Pearson correlation between short FC and reference FC.

    No shrinkage, no corrections. Baseline r_FC.
    """
    fc_short = get_fc_matrix(short_obs, vectorized=True, use_shrinkage=False)
    return float(np.corrcoef(fc_short, fc_reference)[0, 1])


def level_l1_lw(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
) -> float:
    """L1 (+LW): r_FC with Ledoit-Wolf shrinkage on short FC.

    Reference FC already computed with shrinkage.
    """
    fc_short = get_fc_matrix(short_obs, vectorized=True, use_shrinkage=True)
    return float(np.corrcoef(fc_short, fc_reference)[0, 1])


def level_l2_bootstrap(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    block_size: int,
    n_bootstraps: int = 50,
) -> float:
    """L2 (+Bootstrap): Median of block-bootstrap r_FC samples with LW.

    Variance reduction via resampling. No reliability correction.
    """
    t_samples = short_obs.shape[0]
    r_fc_list = []

    for _ in range(n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]
        fc_b = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r = np.corrcoef(fc_b, fc_reference)[0, 1]
        r_fc_list.append(r)

    return float(np.nanmedian(r_fc_list))


def level_l3_sb_atten(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    block_size: int,
    k: float,
    n_bootstraps: int = 50,
) -> float:
    """L3 (+SB+Atten): Attenuation correction with SB prophecy, no prior.

    Core innovation: reliability-based correction using split-half + SB.
    Aggregation in raw space (median of corrected values).
    """
    t_samples = short_obs.shape[0]
    rho_list = []

    for _ in range(n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]
        fc_obs = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs = np.corrcoef(fc_obs, fc_reference)[0, 1]
        r_split = compute_split_half_reliability(ts_b, use_shrinkage=True)

        rho_est = correct_attenuation(
            r_obs, 0.98, r_split, k=k,
            empirical_prior=None, method="fisher_z",
        )
        rho_list.append(rho_est)

    return float(np.nanmedian(rho_list))


def level_l4_prior(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    block_size: int,
    k: float,
    n_bootstraps: int = 50,
) -> float:
    """L4 (+Prior): Add Bayesian empirical prior to split-half estimate.

    Regularizes extreme split-half values via shrinkage toward prior mean.
    Aggregation in raw space.
    """
    t_samples = short_obs.shape[0]
    empirical_prior = (0.25, 0.05)
    rho_list = []

    for _ in range(n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]
        fc_obs = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs = np.corrcoef(fc_obs, fc_reference)[0, 1]
        r_split = compute_split_half_reliability(ts_b, use_shrinkage=True)

        rho_est = correct_attenuation(
            r_obs, 0.98, r_split, k=k,
            empirical_prior=empirical_prior, method="fisher_z",
        )
        rho_list.append(rho_est)

    return float(np.nanmedian(rho_list))


def level_l5_full(
    short_obs: np.ndarray,
    fc_reference: np.ndarray,
    block_size: int,
    k: float,
    n_bootstraps: int = 50,
) -> float:
    """L5 (Full BS-NET): Full pipeline with Fisher z-space aggregation.

    Identical to L4 except bootstrap estimates are aggregated in Fisher z
    space (median of z-transformed values → inverse transform), which is
    more robust to outlier bootstrap samples near ±1.
    """
    t_samples = short_obs.shape[0]
    empirical_prior = (0.25, 0.05)
    z_list = []

    for _ in range(n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = short_obs[idx, :]
        fc_obs = get_fc_matrix(ts_b, vectorized=True, use_shrinkage=True)
        r_obs = np.corrcoef(fc_obs, fc_reference)[0, 1]
        r_split = compute_split_half_reliability(ts_b, use_shrinkage=True)

        rho_est = correct_attenuation(
            r_obs, 0.98, r_split, k=k,
            empirical_prior=empirical_prior, method="fisher_z",
        )
        z_list.append(fisher_z(rho_est))

    return float(fisher_z_inv(np.nanmedian(z_list)))


# ============================================================================
# Worker function (one subject × one seed → 6 LevelResults)
# ============================================================================

def _subject_worker(args: tuple) -> list[LevelResult]:
    """Process one subject × one seed across all 6 progressive levels.

    Args:
        args: (npy_path, seed, short_samples, n_bootstraps)

    Returns:
        List of 6 LevelResult (one per level).
    """
    npy_path, seed, short_samples, n_bootstraps = args

    subject_id = Path(npy_path).stem  # e.g., "sub-001_cc200" or "50030_cc200"

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

    # Estimate optimal block size
    block_size = estimate_optimal_block_length(short_obs)

    results = []

    # L0: Raw
    rho = level_l0_raw(short_obs, fc_reference)
    results.append(LevelResult(subject_id, "L0", "Raw", seed, rho))

    # L1: +LW
    rho = level_l1_lw(short_obs, fc_reference)
    results.append(LevelResult(subject_id, "L1", "+LW", seed, rho))

    # L2: +Bootstrap
    rho = level_l2_bootstrap(short_obs, fc_reference, block_size, n_bootstraps)
    results.append(LevelResult(subject_id, "L2", "+Bootstrap", seed, rho))

    # L3: +SB+Atten
    rho = level_l3_sb_atten(
        short_obs, fc_reference, block_size, k, n_bootstraps,
    )
    results.append(LevelResult(subject_id, "L3", "+SB+Atten", seed, rho))

    # L4: +Prior
    rho = level_l4_prior(
        short_obs, fc_reference, block_size, k, n_bootstraps,
    )
    results.append(LevelResult(subject_id, "L4", "+Prior", seed, rho))

    # L5: Full BS-NET
    rho = level_l5_full(
        short_obs, fc_reference, block_size, k, n_bootstraps,
    )
    results.append(LevelResult(subject_id, "L5", "Full BS-NET", seed, rho))

    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Progressive ablation study for BS-NET: cumulative component "
            "addition (L0→L5) on real fMRI data."
        ),
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Directory of .npy timeseries files for batch processing.",
    )
    input_group.add_argument(
        "--input-npy",
        type=str,
        help="Single .npy timeseries file (n_samples, n_rois).",
    )

    parser.add_argument(
        "--short-samples",
        type=int,
        default=None,
        help=(
            "Number of TRs for short observation. "
            "Default: 48 (ds000243, TR=2.5s → 120s) or 60 (ABIDE, TR=2s → 120s)."
        ),
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=0,
        help="Max subjects from --input-dir (0 = all, default: 0).",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=100,
        help="Bootstrap iterations per level (default: 100).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,777,2026,9999,314,628,1414,2718,3141",
        help="Comma-separated random seeds (default: 10 seeds).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers (default: 1). Set BLAS threading env vars!",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: auto-generated in artifacts/reports/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test: 2 subjects × 2 seeds × 3 bootstraps.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Execute progressive ablation study and save results."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse seeds
    seeds = [int(s) for s in args.seeds.split(",")]

    # Dry-run overrides
    if args.dry_run:
        seeds = seeds[:2]
        n_bootstraps = 3
        n_subjects_limit = 2
        logger.info("=== DRY-RUN MODE: 2 subjects × 2 seeds × 3 bootstraps ===")
    else:
        n_bootstraps = args.n_bootstraps
        n_subjects_limit = args.n_subjects

    output_dir = Path("artifacts/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[LevelResult] = []

    # -------------------------------------------------------------------
    # Batch directory mode
    # -------------------------------------------------------------------
    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
        npy_files = sorted(input_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files in {input_dir}")

        # Subject sampling
        if n_subjects_limit > 0 and n_subjects_limit < len(npy_files):
            rng = np.random.default_rng(42)
            indices = rng.choice(len(npy_files), size=n_subjects_limit, replace=False)
            npy_files = [npy_files[i] for i in sorted(indices)]

        # Auto-detect short_samples from first file
        if args.short_samples is None:
            sample = np.load(str(npy_files[0]))
            total_trs = sample.shape[0]
            # Heuristic: ds000243 TR=2.5s, ABIDE TR=2s → aim for ~120s
            if total_trs > 200:  # ds000243 has 480 TRs
                short_samples = 48  # 48 × 2.5s = 120s
            else:
                short_samples = 60  # 60 × 2s = 120s
            logger.info(
                f"Auto short_samples={short_samples} "
                f"(total_trs={total_trs} in {npy_files[0].name})"
            )
        else:
            short_samples = args.short_samples

        n_subj = len(npy_files)
        n_seeds = len(seeds)
        total_tasks = n_subj * n_seeds
        total_runs = total_tasks * len(LEVELS)

        logger.info("=" * 70)
        logger.info("BS-NET PROGRESSIVE ABLATION STUDY")
        logger.info("=" * 70)
        logger.info(f"Input: {input_dir}")
        logger.info(
            f"Subjects: {n_subj}, Seeds: {n_seeds}, Levels: {len(LEVELS)}, "
            f"Total runs: {total_runs}"
        )
        logger.info(
            f"short_samples={short_samples}, n_bootstraps={n_bootstraps}, "
            f"n_jobs={args.n_jobs}"
        )
        logger.info("=" * 70)

        # Build task list
        tasks = [
            (npy_path, seed, short_samples, n_bootstraps)
            for npy_path in npy_files
            for seed in seeds
        ]

        t0 = time.time()

        if args.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
                futures = {
                    executor.submit(_subject_worker, task): task
                    for task in tasks
                }
                with tqdm(
                    total=total_tasks,
                    desc="Progressive ablation",
                    unit="task",
                ) as pbar:
                    for future in as_completed(futures):
                        try:
                            results.extend(future.result())
                        except Exception as e:
                            task = futures[future]
                            logger.error(
                                f"Failed: {Path(task[0]).name} seed={task[1]}: {e}"
                            )
                        pbar.update(1)
        else:
            for task in tqdm(tasks, desc="Progressive ablation", unit="task"):
                try:
                    results.extend(_subject_worker(task))
                except Exception as e:
                    logger.error(
                        f"Failed: {Path(task[0]).name} seed={task[1]}: {e}"
                    )

        elapsed = time.time() - t0
        logger.info(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

        # Output filename
        if args.output:
            output_file = Path(args.output)
        else:
            # Infer dataset tag from directory structure
            dir_name = input_dir.name  # e.g., "cc200"
            parent_name = input_dir.parent.name  # e.g., "timeseries_cache"
            grandparent = input_dir.parent.parent.name  # e.g., "ds000243"
            tag = f"{grandparent}_{dir_name}" if grandparent != "." else dir_name
            dry_tag = "_dryrun" if args.dry_run else ""
            output_file = (
                output_dir
                / f"progressive_ablation_{tag}_N{n_subj}{dry_tag}.csv"
            )

    # -------------------------------------------------------------------
    # Single file mode
    # -------------------------------------------------------------------
    elif args.input_npy is not None:
        short_samples = args.short_samples if args.short_samples else 60

        logger.info("=" * 70)
        logger.info("BS-NET PROGRESSIVE ABLATION STUDY (single file)")
        logger.info(f"Input: {args.input_npy}")
        logger.info(f"short_samples={short_samples}, seeds={seeds}")
        logger.info("=" * 70)

        tasks = [
            (args.input_npy, seed, short_samples, n_bootstraps)
            for seed in seeds
        ]

        for task in tqdm(tasks, desc="Progressive ablation", unit="seed"):
            results.extend(_subject_worker(task))

        if args.output:
            output_file = Path(args.output)
        else:
            stem = Path(args.input_npy).stem
            dry_tag = "_dryrun" if args.dry_run else ""
            output_file = output_dir / f"progressive_ablation_{stem}{dry_tag}.csv"

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    # Compute delta from L0 (raw) for each subject × seed
    raw_by_key: dict[tuple[str, int], float] = {}
    for r in results:
        if r.level == "L0":
            raw_by_key[(r.subject_id, r.seed)] = r.rho_hat_T

    with open(output_file, "w") as f:
        f.write("subject_id,level,level_name,seed,rho_hat_T,delta_from_raw\n")
        for r in results:
            raw_val = raw_by_key.get((r.subject_id, r.seed), np.nan)
            delta = r.rho_hat_T - raw_val
            f.write(
                f"{r.subject_id},{r.level},{r.level_name},{r.seed},"
                f"{r.rho_hat_T:.6f},{delta:.6f}\n"
            )

    logger.info(f"\nResults saved to {output_file}")

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Progressive Ablation Summary")
    print("=" * 80)

    n_subjects_actual = len({r.subject_id for r in results})
    n_seeds_actual = len({r.seed for r in results})
    print(f"Subjects: {n_subjects_actual}, Seeds: {n_seeds_actual}")
    print("-" * 80)
    print(
        f"{'Level':<8} {'Name':<16} {'Mean':<14} {'Std':<10} "
        f"{'Δ from Raw':<12} {'Δ from Prev':<12}"
    )
    print("-" * 80)

    prev_mean = None
    for level_code, level_name in LEVELS:
        values = [r.rho_hat_T for r in results if r.level == level_code]
        if not values:
            continue
        mean_val = np.mean(values)
        std_val = np.std(values)

        raw_values = [r.rho_hat_T for r in results if r.level == "L0"]
        raw_mean = np.mean(raw_values) if raw_values else 0.0
        delta_raw = mean_val - raw_mean

        delta_prev = mean_val - prev_mean if prev_mean is not None else 0.0

        print(
            f"{level_code:<8} {level_name:<16} "
            f"{mean_val:.4f}±{std_val:.4f}  "
            f"{delta_raw:+.4f}       {delta_prev:+.4f}"
        )
        prev_mean = mean_val

    print("=" * 80)
    print(f"\nOutput: {output_file}")


if __name__ == "__main__":
    main()
