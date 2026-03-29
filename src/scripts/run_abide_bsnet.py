#!/usr/bin/env python3
"""
BS-NET validation on ABIDE Preprocessed Connectomes Project (PCP) data.

Phase 1: Download pre-extracted ROI time series (CC200/CC400) from ABIDE PCP
         via nilearn and run BS-NET pipeline directly — no fMRIPrep/XCP-D needed.

Phase 2 (future): After fMRIPrep/XCP-D completes, re-parcellate with Schaefer
         100/400 and compare across atlases.

Usage:
    # Download + validate with CC200 (default)
    python src/scripts/run_abide_bsnet.py --atlas cc200

    # CC400
    python src/scripts/run_abide_bsnet.py --atlas cc400

    # Limit subjects (for testing)
    python src/scripts/run_abide_bsnet.py --atlas cc200 --max-subjects 10

    # Specific pipeline and strategy
    python src/scripts/run_abide_bsnet.py --atlas cc200 --pipeline cpac --strategy filt_noglobal

    # Only healthy controls (for BS-NET validation)
    python src/scripts/run_abide_bsnet.py --atlas cc200 --group controls

    # Verbose output
    python src/scripts/run_abide_bsnet.py --atlas cc200 --max-subjects 5 --verbose

    # Dry run (download only, no BS-NET)
    python src/scripts/run_abide_bsnet.py --atlas cc200 --max-subjects 5 --download-only
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---- Project imports ----
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class ABIDEConfig:
    """ABIDE PCP download and validation configuration."""

    atlas: str = "cc200"  # cc200 | cc400
    pipeline: str = "cpac"  # cpac | ccs | dparsf | niak
    strategy: str = "filt_noglobal"  # filt_global | filt_noglobal | nofilt_global | nofilt_noglobal
    group: str = "both"  # controls | autism | both
    max_subjects: int = 0  # 0 = all
    min_duration_sec: float = 150.0  # minimum total scan (short_sec + 30s buffer)
    short_duration_sec: int = 120  # BS-NET short window
    output_dir: str = "data/abide"
    download_only: bool = False
    verbose: bool = False
    n_jobs: int = 1  # 1=sequential, -1=all cores, N=N cores
    correction_method: str = "original"  # original | fisher_z | partial | soft_clamp

    @property
    def n_rois(self) -> int:
        return {"cc200": 200, "cc400": 392}[self.atlas]

    @property
    def nilearn_derivative(self) -> str:
        return f"rois_{self.atlas}"


# ============================================================================
# Download ABIDE PCP via nilearn
# ============================================================================
def download_abide_timeseries(cfg: ABIDEConfig) -> list[dict]:
    """Download ABIDE PCP ROI time series via nilearn.

    Args:
        cfg: ABIDE configuration.

    Returns:
        List of subject dicts with keys:
            - sub_id: str (e.g., '0050002')
            - site: str (e.g., 'Caltech')
            - group: str ('Autism' or 'Control')
            - age: float
            - ts_path: str (path to downloaded .1D file)
            - tr: float (repetition time)
    """
    try:
        from nilearn.datasets import fetch_abide_pcp
    except ImportError:
        logger.error("nilearn is required. Install: pip install nilearn --break-system-packages")
        sys.exit(1)

    logger.info(f"Downloading ABIDE PCP: atlas={cfg.atlas}, pipeline={cfg.pipeline}, "
                f"strategy={cfg.strategy}, group={cfg.group}")

    # Build fetch kwargs
    fetch_kwargs: dict = {
        "pipeline": cfg.pipeline,
        "band_pass_filtering": "filt" in cfg.strategy,
        "global_signal_regression": "global" in cfg.strategy and "noglobal" not in cfg.strategy,
        "derivatives": [cfg.nilearn_derivative],
        "data_dir": cfg.output_dir,
    }

    if cfg.group == "controls":
        fetch_kwargs["DX_GROUP"] = 2  # 2 = Control in ABIDE
    elif cfg.group == "autism":
        fetch_kwargs["DX_GROUP"] = 1  # 1 = Autism

    if cfg.max_subjects > 0:
        fetch_kwargs["n_subjects"] = cfg.max_subjects

    abide = fetch_abide_pcp(**fetch_kwargs)

    # Extract subject metadata
    # nilearn returns either:
    #   - list of file paths (str) → load later
    #   - list of numpy arrays (already loaded) → save to disk
    subjects = []
    phenotypic = abide.phenotypic
    ts_data = abide[cfg.nilearn_derivative]

    # Determine if data is paths or arrays
    sample = ts_data[0] if len(ts_data) > 0 else None
    is_array = isinstance(sample, np.ndarray)
    if is_array:
        logger.info("nilearn returned pre-loaded numpy arrays (saving to disk)")

    cache_dir = Path(cfg.output_dir) / "timeseries_cache" / cfg.atlas
    if is_array:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for i, ts_item in enumerate(ts_data):
        row = phenotypic.iloc[i] if hasattr(phenotypic, "iloc") else phenotypic[i]
        sub_id = str(row.get("SUB_ID", row.get("subject", f"sub_{i:05d}")))

        if is_array:
            # ts_item is a numpy array — save to .npy for consistent handling
            if ts_item is None or (isinstance(ts_item, np.ndarray) and ts_item.size == 0):
                continue
            ts_path = str(cache_dir / f"{sub_id}_{cfg.atlas}.npy")
            np.save(ts_path, ts_item)
        else:
            # ts_item is a file path (str or Path-like)
            ts_path = str(ts_item)
            if not Path(ts_path).exists():
                continue

        sub_info = {
            "sub_id": sub_id,
            "site": str(row.get("SITE_ID", "unknown")),
            "group": "Autism" if row.get("DX_GROUP", 0) == 1 else "Control",
            "age": float(row.get("AGE_AT_SCAN", 0)),
            "ts_path": ts_path,
            # ABIDE TR은 사이트별로 다름. phenotypic에 없으면 기본값 사용
            "tr": _get_abide_tr(str(row.get("SITE_ID", ""))),
        }
        subjects.append(sub_info)

    logger.info(f"Downloaded {len(subjects)} subjects "
                f"(Controls: {sum(1 for s in subjects if s['group'] == 'Control')}, "
                f"Autism: {sum(1 for s in subjects if s['group'] == 'Autism')})")

    return subjects


def _get_abide_tr(site_id: str) -> float:
    """Get approximate TR for ABIDE site.

    Most ABIDE sites use TR=2.0s, with some exceptions.
    """
    tr_map = {
        "Caltech": 2.0, "CMU": 2.0, "KKI": 2.5, "Leuven": 1.667,
        "MaxMun": 3.0, "NYU": 2.0, "OHSU": 2.5, "Olin": 1.5,
        "Pitt": 1.5, "SBL": 2.2, "SDSU": 2.0, "Stanford": 2.0,
        "Trinity": 2.0, "UCLA": 3.0, "UM": 2.0, "USM": 2.0, "Yale": 2.0,
    }
    # Fuzzy match: site_id may contain extra info
    for key, tr in tr_map.items():
        if key.lower() in site_id.lower():
            return tr
    return 2.0  # default


# ============================================================================
# Load time series from .1D file
# ============================================================================
def load_timeseries(ts_path: str) -> np.ndarray | None:
    """Load ROI time series from ABIDE .1D file.

    Args:
        ts_path: Path to .1D or .npy file.

    Returns:
        numpy array of shape (n_timepoints, n_rois), or None if loading fails.
    """
    path = Path(ts_path)

    try:
        if path.suffix == ".npy":
            ts = np.load(path)
        elif path.suffix == ".1D":
            ts = np.loadtxt(path)
        else:
            # Try generic loading
            ts = np.loadtxt(path)

        # Validate shape
        if ts.ndim == 1:
            logger.warning(f"1D array loaded from {path.name}, skipping")
            return None
        if ts.shape[0] < 10:
            logger.warning(f"Too few timepoints ({ts.shape[0]}) in {path.name}")
            return None

        # Remove any all-zero columns (invalid ROIs)
        valid_cols = np.std(ts, axis=0) > 1e-8
        n_invalid = np.sum(~valid_cols)
        if n_invalid > 0:
            logger.debug(f"Removing {n_invalid} zero-variance ROIs from {path.name}")
            ts = ts[:, valid_cols]

        return ts.astype(np.float64)

    except Exception as e:
        logger.warning(f"Failed to load {ts_path}: {e}")
        return None


# ============================================================================
# Run BS-NET on a single subject
# ============================================================================
def run_bsnet_single(
    ts: np.ndarray,
    tr: float,
    short_sec: int = 120,
    target_min: int = 15,
    n_bootstraps: int = 100,
    seed: int = 42,
    correction_method: str = "original",
) -> dict | None:
    """Run BS-NET pipeline on pre-extracted time series.

    Args:
        ts: Time series array (n_timepoints, n_rois).
        tr: Repetition time in seconds.
        short_sec: Short observation window in seconds.
        target_min: Target extrapolation duration in minutes.
        n_bootstraps: Number of bootstrap iterations.
        seed: Random seed.
        correction_method: Attenuation correction method (see bootstrap.py).

    Returns:
        Dict with results, or None if validation fails.
    """
    n_vols, n_rois = ts.shape
    short_vols = int(short_sec / tr)
    total_sec = n_vols * tr

    # Validation
    if n_vols < short_vols + 10:
        logger.warning(f"Insufficient volumes: {n_vols} < {short_vols + 10}")
        return None

    # Config
    config = BSNetConfig(
        n_rois=n_rois,
        tr=tr,
        short_duration_sec=short_sec,
        target_duration_min=target_min,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # Full-scan FC (reference)
    fc_full_vec = get_fc_matrix(ts, vectorized=True, use_shrinkage=True)

    # Short-scan FC (first 2 minutes)
    ts_short = ts[:short_vols, :]
    fc_short_vec = get_fc_matrix(ts_short, vectorized=True, use_shrinkage=True)

    # Baseline: raw correlation between short and full FC
    r_fc_raw = float(np.corrcoef(fc_short_vec, fc_full_vec)[0, 1])

    # BS-NET prediction
    result = run_bootstrap_prediction(
        ts_short, fc_full_vec, config,
        correction_method=correction_method,
    )

    # BS-NET enhanced FC correlation
    # rho_hat_T is a scalar reliability estimate, not a full FC vector.
    # The improvement metric is rho_hat_T vs r_fc_raw.

    return {
        "n_vols": n_vols,
        "n_rois": n_rois,
        "total_sec": round(total_sec, 1),
        "tr": tr,
        "r_fc_raw": round(r_fc_raw, 4),
        "rho_hat_T": round(float(result.rho_hat_T), 4),
        "ci_lower": round(float(result.ci_lower), 4),
        "ci_upper": round(float(result.ci_upper), 4),
        "improvement": round(float(result.rho_hat_T) - r_fc_raw, 4),
    }


# ============================================================================
# Parallel worker
# ============================================================================
def _process_subject(args: tuple) -> dict | None:
    """Worker function for parallel BS-NET processing.

    Args:
        args: Tuple of (index, subject_dict, short_duration_sec, correction_method).

    Returns:
        Result dict with metadata, or None on failure.
    """
    i, sub, short_sec, corr_method = args
    sub_id = sub["sub_id"]

    ts = load_timeseries(sub["ts_path"])
    if ts is None:
        return {"status": "fail", "sub_id": sub_id, "reason": "load_failed"}

    total_sec = ts.shape[0] * sub["tr"]
    if total_sec < 150.0:
        return {"status": "fail", "sub_id": sub_id, "reason": f"too_short_{total_sec:.0f}s"}

    try:
        res = run_bsnet_single(
            ts=ts, tr=sub["tr"], short_sec=short_sec,
            correction_method=corr_method,
        )
    except Exception as e:
        return {"status": "fail", "sub_id": sub_id, "reason": str(e)}

    if res is None:
        return {"status": "fail", "sub_id": sub_id, "reason": "bsnet_returned_none"}

    res["sub_id"] = sub_id
    res["site"] = sub["site"]
    res["group"] = sub["group"]
    res["age"] = sub["age"]
    res["status"] = "ok"
    return res


def _resolve_n_jobs(n_jobs: int) -> int:
    """Resolve n_jobs to actual worker count."""
    cpu_count = os.cpu_count() or 1
    if n_jobs == -1:
        return cpu_count
    if n_jobs <= 0:
        return max(1, cpu_count + n_jobs)
    return min(n_jobs, cpu_count)


# ============================================================================
# Batch pipeline
# ============================================================================
def run_abide_validation(cfg: ABIDEConfig) -> Path:
    """Run full ABIDE validation pipeline.

    1. Download ABIDE PCP time series via nilearn.
    2. For each subject: load time series → run BS-NET → collect results.
    3. Save results CSV and summary JSON.

    Args:
        cfg: ABIDE configuration.

    Returns:
        Path to results CSV file.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    logger.info("=" * 60)
    logger.info("Phase 1: Downloading ABIDE PCP time series")
    logger.info("=" * 60)

    subjects = download_abide_timeseries(cfg)
    if not subjects:
        logger.error("No subjects downloaded. Exiting.")
        sys.exit(1)

    if cfg.download_only:
        logger.info("Download-only mode. Exiting.")
        # Save subject list
        subj_path = results_dir / f"abide_subjects_{cfg.atlas}.json"
        with open(subj_path, "w") as f:
            json.dump(subjects, f, indent=2)
        logger.info(f"Subject list saved: {subj_path}")
        return subj_path

    # Step 2: Run BS-NET per subject
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 2: Running BS-NET validation")
    logger.info("=" * 60)

    results = []
    n_success = 0
    n_fail = 0
    t_start = time.time()

    n_workers = _resolve_n_jobs(cfg.n_jobs)
    use_parallel = n_workers > 1

    if use_parallel:
        logger.info(f"Parallel mode: {n_workers} workers")
        worker_args = [
            (i, sub, cfg.short_duration_sec, cfg.correction_method)
            for i, sub in enumerate(subjects)
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_subject, arg): arg[0]
                for arg in worker_args
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    logger.error(f"Worker exception for idx={idx}: {e}")
                    n_fail += 1
                    continue

                if res is None or res.get("status") != "ok":
                    reason = res.get("reason", "unknown") if res else "none"
                    sub_id = res.get("sub_id", f"idx_{idx}") if res else f"idx_{idx}"
                    logger.debug(f"{sub_id}: {reason}")
                    n_fail += 1
                    continue

                res["atlas"] = cfg.atlas
                res["pipeline"] = cfg.pipeline
                res["strategy"] = cfg.strategy
                del res["status"]
                results.append(res)
                n_success += 1

                if cfg.verbose:
                    logger.info(
                        f"[{n_success + n_fail}/{len(subjects)}] "
                        f"{res['sub_id']} ({res['site']}): "
                        f"r_fc_raw={res['r_fc_raw']:.3f} → ρ̂T={res['rho_hat_T']:.3f} "
                        f"(Δ={res['improvement']:+.3f})"
                    )
                elif (n_success + n_fail) % 20 == 0:
                    logger.info(
                        f"[{n_success + n_fail}/{len(subjects)}] "
                        f"({n_success} ok, {n_fail} fail)"
                    )
    else:
        logger.info("Sequential mode (use --n-jobs to parallelize)")
        for i, sub in enumerate(subjects):
            sub_id = sub["sub_id"]
            prefix = f"[{i+1}/{len(subjects)}]"

            ts = load_timeseries(sub["ts_path"])
            if ts is None:
                logger.warning(f"{prefix} {sub_id}: load failed, skipping")
                n_fail += 1
                continue

            total_sec = ts.shape[0] * sub["tr"]
            if total_sec < cfg.min_duration_sec:
                logger.warning(
                    f"{prefix} {sub_id}: too short "
                    f"({total_sec:.0f}s < {cfg.min_duration_sec}s)"
                )
                n_fail += 1
                continue

            try:
                res = run_bsnet_single(
                    ts=ts, tr=sub["tr"], short_sec=cfg.short_duration_sec,
                    correction_method=cfg.correction_method,
                )
            except Exception as e:
                logger.error(f"{prefix} {sub_id}: BS-NET error: {e}")
                n_fail += 1
                continue

            if res is None:
                n_fail += 1
                continue

            res["sub_id"] = sub_id
            res["site"] = sub["site"]
            res["group"] = sub["group"]
            res["age"] = sub["age"]
            res["atlas"] = cfg.atlas
            res["pipeline"] = cfg.pipeline
            res["strategy"] = cfg.strategy
            results.append(res)
            n_success += 1

            if cfg.verbose:
                logger.info(
                    f"{prefix} {sub_id} ({sub['site']}): "
                    f"r_fc_raw={res['r_fc_raw']:.3f} → ρ̂T={res['rho_hat_T']:.3f} "
                    f"(Δ={res['improvement']:+.3f})"
                )
            elif (i + 1) % 20 == 0:
                logger.info(f"{prefix} processed ({n_success} ok, {n_fail} fail)")

    elapsed = time.time() - t_start

    # Step 3: Save results
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 3: Saving results")
    logger.info("=" * 60)

    csv_path = results_dir / f"abide_bsnet_{cfg.atlas}_{cfg.pipeline}_{cfg.strategy}.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results CSV: {csv_path}")

    # Summary statistics
    if results:
        r_fc_raw_arr = np.array([r["r_fc_raw"] for r in results])
        rho_hat_arr = np.array([r["rho_hat_T"] for r in results])
        improvement_arr = np.array([r["improvement"] for r in results])

        summary = {
            "config": {
                "atlas": cfg.atlas,
                "pipeline": cfg.pipeline,
                "strategy": cfg.strategy,
                "group": cfg.group,
                "short_duration_sec": cfg.short_duration_sec,
            },
            "counts": {
                "total_downloaded": len(subjects),
                "success": n_success,
                "failed": n_fail,
                "controls": sum(1 for r in results if r["group"] == "Control"),
                "autism": sum(1 for r in results if r["group"] == "Autism"),
            },
            "r_fc_raw": {
                "mean": round(float(np.mean(r_fc_raw_arr)), 4),
                "std": round(float(np.std(r_fc_raw_arr)), 4),
                "median": round(float(np.median(r_fc_raw_arr)), 4),
            },
            "rho_hat_T": {
                "mean": round(float(np.mean(rho_hat_arr)), 4),
                "std": round(float(np.std(rho_hat_arr)), 4),
                "median": round(float(np.median(rho_hat_arr)), 4),
            },
            "improvement": {
                "mean": round(float(np.mean(improvement_arr)), 4),
                "std": round(float(np.std(improvement_arr)), 4),
                "median": round(float(np.median(improvement_arr)), 4),
                "pct_improved": round(
                    float(np.mean(improvement_arr > 0)) * 100, 1
                ),
            },
            "elapsed_sec": round(elapsed, 1),
        }

        summary_path = results_dir / f"abide_summary_{cfg.atlas}_{cfg.pipeline}_{cfg.strategy}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary JSON: {summary_path}")

        # Print summary
        logger.info("")
        logger.info("━" * 50)
        logger.info(f" ABIDE BS-NET Validation Summary ({cfg.atlas.upper()})")
        logger.info("━" * 50)
        logger.info(f"  Subjects: {n_success} success / {n_fail} failed")
        logger.info(f"  Atlas: {cfg.atlas} ({results[0]['n_rois']} ROIs)")
        logger.info(f"  Pipeline: {cfg.pipeline}, Strategy: {cfg.strategy}")
        logger.info("")
        logger.info("  r_FC (raw, 2min vs full):")
        logger.info(f"    mean={summary['r_fc_raw']['mean']:.3f} ± {summary['r_fc_raw']['std']:.3f}")
        logger.info("  ρ̂T (BS-NET predicted):")
        logger.info(f"    mean={summary['rho_hat_T']['mean']:.3f} ± {summary['rho_hat_T']['std']:.3f}")
        logger.info("  Improvement (ρ̂T - r_FC):")
        logger.info(f"    mean={summary['improvement']['mean']:+.3f} ± {summary['improvement']['std']:.3f}")
        logger.info(f"    {summary['improvement']['pct_improved']}% of subjects improved")
        logger.info(f"  Elapsed: {elapsed:.0f}s ({elapsed/max(n_success,1):.1f}s/subject)")
        logger.info("━" * 50)

    return csv_path


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> ABIDEConfig:
    parser = argparse.ArgumentParser(
        description="BS-NET validation on ABIDE Preprocessed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--atlas", choices=["cc200", "cc400"], default="cc200",
        help="Parcellation atlas (default: cc200)",
    )
    parser.add_argument(
        "--pipeline", choices=["cpac", "ccs", "dparsf", "niak"], default="cpac",
        help="Preprocessing pipeline (default: cpac)",
    )
    parser.add_argument(
        "--strategy",
        choices=["filt_global", "filt_noglobal", "nofilt_global", "nofilt_noglobal"],
        default="filt_noglobal",
        help="Denoising strategy (default: filt_noglobal)",
    )
    parser.add_argument(
        "--group", choices=["controls", "autism", "both"], default="controls",
        help="Subject group (default: controls)",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=0,
        help="Max subjects to process (0=all, default: 0)",
    )
    parser.add_argument(
        "--short-sec", type=int, default=120,
        help="Short scan duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/abide",
        help="Output directory (default: data/abide)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel workers (1=sequential, -1=all cores, default: 1)",
    )
    parser.add_argument(
        "--correction-method",
        choices=["original", "fisher_z", "partial", "soft_clamp"],
        default="original",
        help="Attenuation correction method (default: original)",
    )
    parser.add_argument("--download-only", action="store_true", help="Download only, skip BS-NET")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    return ABIDEConfig(
        atlas=args.atlas,
        pipeline=args.pipeline,
        strategy=args.strategy,
        group=args.group,
        max_subjects=args.max_subjects,
        short_duration_sec=args.short_sec,
        output_dir=args.output_dir,
        download_only=args.download_only,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
        correction_method=args.correction_method,
    )


def main() -> None:
    cfg = parse_args()

    log_level = logging.DEBUG if cfg.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    run_abide_validation(cfg)


if __name__ == "__main__":
    main()
