#!/usr/bin/env python3
"""
BS-NET validation on nilearn ADHD resting-state dataset.

ADHD-200 subset (40 subjects: 20 ADHD + 20 controls) provides:
  - 4D preprocessed resting-state fMRI NIfTI
  - Confound regressors
  - Phenotypic data

Unlike ABIDE PCP (pre-extracted ROI time series), this dataset requires
atlas-based parcellation, allowing direct comparison across atlases:
  - CC200 (Craddock 200, cross-validation with ABIDE)
  - CC400 (Craddock 400)

Usage:
    # CC200 (default), all subjects
    python src/scripts/run_nilearn_adhd_bsnet.py

    # CC400
    python src/scripts/run_nilearn_adhd_bsnet.py --atlas cc400

    # Limit subjects + verbose
    python src/scripts/run_nilearn_adhd_bsnet.py --max-subjects 5 -v

    # Controls only
    python src/scripts/run_nilearn_adhd_bsnet.py --group controls

    # Multi-atlas comparison (run all atlases sequentially)
    python src/scripts/run_nilearn_adhd_bsnet.py --atlas all
    python src/scripts/run_nilearn_adhd_bsnet.py --atlas all --correction-method fisher_z --n-jobs 8
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
# Atlas definitions
# ============================================================================
ATLAS_REGISTRY: dict[str, dict] = {
    "cc200": {
        "fetch": "craddock",
        "n_rois": 200,
        "volume_idx": 19,  # max_label=200, n_labels=195
        "params": {},
        "label": "Craddock CC200",
    },
    "cc400": {
        "fetch": "craddock",
        "n_rois": 400,
        "volume_idx": 31,  # max_label=400, n_labels=356
        "params": {},
        "label": "Craddock CC400",
    },
}


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class ADHDConfig:
    """ADHD dataset validation configuration."""

    atlas: str = "cc200"
    group: str = "both"  # controls | adhd | both
    max_subjects: int = 0  # 0 = all (40)
    short_duration_sec: int = 120
    target_duration_min: int = 15
    n_bootstraps: int = 100
    output_dir: str = "data/adhd"
    verbose: bool = False
    n_jobs: int = 1  # 1=sequential, -1=all cores
    correction_method: str = "original"  # original | fisher_z | partial | soft_clamp
    multi_seed: int = 0  # 0=single seed, >0=N seeds for stability analysis

    @property
    def n_rois(self) -> int:
        return ATLAS_REGISTRY[self.atlas]["n_rois"]


# ============================================================================
# Fetch atlas
# ============================================================================
def _fetch_craddock_with_ssl_fallback():
    """Fetch Craddock 2012 atlas with SSL fallback for NITRC certificate issues.

    NITRC server (cluster_roi.projects.nitrc.org) has recurring SSL certificate
    hostname mismatch. If the standard fetch fails with SSLError, retry with
    verification disabled.

    Returns:
        Path to the atlas NIfTI file (scorr_mean / maps).
    """
    import nilearn.datasets

    def _do_fetch():
        return nilearn.datasets.fetch_atlas_craddock_2012()

    try:
        atlas = _do_fetch()
    except Exception as exc:
        if "SSL" not in str(type(exc).__name__) and "SSL" not in str(exc):
            raise
        logger.warning(
            "SSL certificate error from NITRC server — retrying with "
            "verification disabled (NITRC hostname mismatch known issue)"
        )
        # nilearn uses requests/urllib3, so patch HTTPAdapter.send
        import requests.adapters

        _orig_send = requests.adapters.HTTPAdapter.send

        def _no_verify_send(self, request, **kwargs):
            kwargs["verify"] = False
            return _orig_send(self, request, **kwargs)

        requests.adapters.HTTPAdapter.send = _no_verify_send
        try:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            atlas = _do_fetch()
        finally:
            requests.adapters.HTTPAdapter.send = _orig_send

    # nilearn >=0.10: Atlas with .maps; older: Bunch with .scorr_mean
    if hasattr(atlas, "maps"):
        return atlas.maps
    return atlas.scorr_mean


def fetch_atlas(atlas_name: str, cache_dir: str | None = None) -> tuple:
    """Fetch atlas image and labels.

    Args:
        atlas_name: Key in ATLAS_REGISTRY.
        cache_dir: Directory to cache extracted atlas NIfTI (for multiprocessing).

    Returns:
        Tuple of (atlas_img, labels_list, atlas_nii_path).
        atlas_nii_path is a file path for passing to worker processes.
    """
    spec = ATLAS_REGISTRY[atlas_name]

    if spec["fetch"] == "craddock":
        import nibabel as nib

        atlas_nii = _fetch_craddock_with_ssl_fallback()
        vol_idx = spec["volume_idx"]
        atlas_img = nib.load(atlas_nii)
        parcel_data = atlas_img.get_fdata()[..., vol_idx]
        parcel_img = nib.Nifti1Image(parcel_data, atlas_img.affine)
        n_labels = int(parcel_data.max())
        prefix = atlas_name.upper()
        labels = [f"{prefix}_{i:03d}" for i in range(1, n_labels + 1)]

        # Save to disk for multiprocessing workers
        cache_path = Path(cache_dir) if cache_dir else Path("data/adhd/atlas_cache")
        cache_path.mkdir(parents=True, exist_ok=True)
        nii_path = str(cache_path / f"{atlas_name}_parcels.nii.gz")
        nib.save(parcel_img, nii_path)

        return parcel_img, labels, nii_path
    else:
        raise ValueError(f"Unknown atlas fetch type: {spec['fetch']}")


# ============================================================================
# Download ADHD dataset
# ============================================================================
def download_adhd(cfg: ADHDConfig) -> dict:
    """Download ADHD resting-state dataset via nilearn.

    Args:
        cfg: Configuration.

    Returns:
        Dict with keys: func (list of paths), confounds (list of paths),
        phenotypic (recarray).
    """
    from nilearn.datasets import fetch_adhd

    n_subjects = cfg.max_subjects if cfg.max_subjects > 0 else 40

    logger.info(f"Downloading ADHD dataset (n={n_subjects})...")
    adhd = fetch_adhd(n_subjects=n_subjects, data_dir=cfg.output_dir)

    logger.info(f"Downloaded {len(adhd.func)} subjects")
    return adhd


# ============================================================================
# Extract time series with atlas parcellation
# ============================================================================
def extract_timeseries(
    func_path: str,
    confounds_path: str | None,
    atlas_img,
    t_r: float = 2.0,
) -> np.ndarray | None:
    """Extract ROI time series from 4D NIfTI using atlas parcellation.

    Applies: detrending, high-pass 0.01 Hz, low-pass 0.1 Hz, z-score.
    Confounds regressed if provided.

    Args:
        func_path: Path to 4D preprocessed BOLD NIfTI.
        confounds_path: Path to confound regressors (TSV/CSV), or None.
        atlas_img: Atlas image (NIfTI or path).
        t_r: Repetition time in seconds.

    Returns:
        Time series array (n_timepoints, n_rois), or None on failure.
    """
    from nilearn.maskers import NiftiLabelsMasker

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize="zscore_sample",
        detrend=True,
        high_pass=0.01,
        low_pass=0.1,
        t_r=t_r,
        verbose=0,
    )

    try:
        # Load confounds if available (ADHD-200 TSV has header row)
        confounds = None
        if confounds_path is not None and Path(confounds_path).exists():
            # Try with header skip; fall back to raw load
            try:
                confounds = np.loadtxt(confounds_path, skiprows=1)
            except ValueError:
                confounds = np.loadtxt(confounds_path)
            if confounds.ndim == 1:
                confounds = confounds[:, np.newaxis]

        ts = masker.fit_transform(func_path, confounds=confounds)

        if ts.shape[0] < 20:
            logger.warning(f"Too few timepoints ({ts.shape[0]})")
            return None

        # Remove zero-variance ROIs
        valid = np.std(ts, axis=0) > 1e-8
        n_invalid = np.sum(~valid)
        if n_invalid > ts.shape[1] * 0.1:
            logger.warning(f"Too many invalid ROIs ({n_invalid}/{ts.shape[1]})")
            return None
        if n_invalid > 0:
            logger.debug(f"Removing {n_invalid} zero-variance ROIs")
            ts = ts[:, valid]

        return ts.astype(np.float64)

    except Exception as e:
        logger.warning(f"Time series extraction failed: {e}")
        return None


# ============================================================================
# Run BS-NET on single subject
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
    """Run BS-NET pipeline on extracted time series.

    Args:
        ts: Time series (n_timepoints, n_rois).
        tr: Repetition time.
        short_sec: Short observation window.
        target_min: Target extrapolation duration.
        n_bootstraps: Bootstrap iterations.
        seed: Random seed.
        correction_method: Attenuation correction method (see bootstrap.py).

    Returns:
        Results dict or None.
    """
    n_vols, n_rois = ts.shape
    short_vols = int(short_sec / tr)

    if n_vols < short_vols + 10:
        logger.warning(f"Insufficient volumes: {n_vols} < {short_vols + 10}")
        return None

    config = BSNetConfig(
        n_rois=n_rois,
        tr=tr,
        short_duration_sec=short_sec,
        target_duration_min=target_min,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    # Reference FC from full scan
    fc_full_vec = get_fc_matrix(ts, vectorized=True, use_shrinkage=True)

    # Short scan (first 2 min)
    ts_short = ts[:short_vols, :]
    fc_short_vec = get_fc_matrix(ts_short, vectorized=True, use_shrinkage=True)

    # Baseline correlation
    r_fc_raw = float(np.corrcoef(fc_short_vec, fc_full_vec)[0, 1])

    # BS-NET
    result = run_bootstrap_prediction(
        ts_short, fc_full_vec, config,
        correction_method=correction_method,
    )

    return {
        "n_vols": n_vols,
        "n_rois": n_rois,
        "total_sec": round(n_vols * tr, 1),
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
def _process_adhd_subject(args: tuple) -> dict | None:
    """Worker for parallel ADHD BS-NET processing.

    Args:
        args: (index, func_path, conf_path, atlas_nii_path, t_r, short_sec,
               target_min, n_bootstraps, atlas_name, correction_method,
               group_label, ts_cache_dir).

    Returns:
        Result dict or failure dict.
    """
    (i, func_path, conf_path, atlas_nii_path, t_r,
     short_sec, target_min, n_bootstraps, atlas_name, corr_method,
     group_label, ts_cache_dir) = args

    # Load atlas in worker process (avoids pickle issues with nibabel)
    import nibabel as nib
    atlas_img = nib.load(atlas_nii_path)

    ts = extract_timeseries(func_path, conf_path, atlas_img, t_r=t_r)
    if ts is None:
        return {"status": "fail", "sub_idx": i, "reason": "extraction_failed"}

    # Cache time series for multi-seed reuse
    if ts_cache_dir:
        cache_path = Path(ts_cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path / f"sub_{i:03d}_ts.npy"), ts)

    total_sec = ts.shape[0] * t_r
    if total_sec < short_sec + 20:
        return {"status": "fail", "sub_idx": i, "reason": f"too_short_{total_sec:.0f}s"}

    try:
        res = run_bsnet_single(
            ts=ts, tr=t_r, short_sec=short_sec,
            target_min=target_min, n_bootstraps=n_bootstraps,
            correction_method=corr_method,
        )
    except Exception as e:
        return {"status": "fail", "sub_idx": i, "reason": str(e)}

    if res is None:
        return {"status": "fail", "sub_idx": i, "reason": "bsnet_returned_none"}

    res["sub_idx"] = i
    res["func_path"] = str(Path(func_path).name)
    res["atlas"] = atlas_name
    res["group"] = group_label
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
# Batch validation
# ============================================================================
def run_adhd_validation(cfg: ADHDConfig) -> Path:
    """Run ADHD validation pipeline for a single atlas.

    Args:
        cfg: Configuration.

    Returns:
        Path to results CSV.
    """
    output_dir = Path(cfg.output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    logger.info("=" * 60)
    logger.info(f"ADHD BS-NET Validation — Atlas: {cfg.atlas}")
    logger.info("=" * 60)

    adhd = download_adhd(cfg)

    # Step 1b: Extract phenotypic info and apply group filter
    func_list = list(adhd.func)
    confounds_list = (
        list(adhd.confounds) if hasattr(adhd, "confounds")
        else [None] * len(func_list)
    )

    # Load group labels from local phenotypic CSV (adhd column: 0=control, 1=ADHD)
    pheno_csv = (
        Path(cfg.output_dir) / "adhd"
        / "ADHD200_40subs_motion_parameters_and_phenotypics.csv"
    )
    group_labels: list[str] = ["unknown"] * len(func_list)
    if pheno_csv.exists():
        # Build Subject ID → group mapping
        sub_group_map: dict[str, str] = {}
        with open(pheno_csv) as f:
            for row in csv.DictReader(f):
                sub_id = row.get("Subject", "")
                adhd_val = row.get("adhd", "")
                if sub_id and adhd_val in ("0", "1"):
                    sub_group_map[sub_id] = "adhd" if adhd_val == "1" else "control"

        # Match func paths to Subject IDs (filename contains subject ID)
        # NIfTI filenames have leading zeros (e.g. "0010042"), CSV does not ("10042")
        for i, func_path in enumerate(func_list):
            fname = Path(func_path).stem  # e.g. "0010042_rest_tshift_RPI_voreg_mni"
            sub_id_raw = fname.split("_")[0]
            sub_id_norm = str(int(sub_id_raw))  # strip leading zeros
            group_labels[i] = sub_group_map.get(
                sub_id_raw, sub_group_map.get(sub_id_norm, "unknown")
            )

        n_ctrl = sum(1 for g in group_labels if g == "control")
        n_adhd = sum(1 for g in group_labels if g == "adhd")
        logger.info(f"Phenotypic loaded: {n_ctrl} controls, {n_adhd} ADHD")
    else:
        logger.warning(f"Phenotypic CSV not found: {pheno_csv}")

    # Apply group filter
    if cfg.group != "both" and group_labels[0] != "unknown":
        target = "control" if cfg.group == "controls" else "adhd"
        indices = [i for i, g in enumerate(group_labels) if g == target]
        func_list = [func_list[i] for i in indices]
        confounds_list = [confounds_list[i] for i in indices]
        group_labels = [group_labels[i] for i in indices]
        logger.info(f"Group filter '{cfg.group}': {len(indices)} subjects selected")

    # Step 2: Fetch atlas
    logger.info(f"Fetching atlas: {ATLAS_REGISTRY[cfg.atlas]['label']}...")
    atlas_cache = str(Path(cfg.output_dir) / "atlas_cache")
    atlas_img, atlas_labels, atlas_nii_path = fetch_atlas(cfg.atlas, cache_dir=atlas_cache)
    logger.info(f"Atlas loaded: {len(atlas_labels)} ROIs")

    # Step 3: Process subjects
    logger.info("")
    logger.info("Processing subjects...")

    # ADHD dataset TR = 2.0s (NYU site)
    t_r = 2.0

    results = []
    n_success = 0
    n_fail = 0
    t_start = time.time()

    n_workers = _resolve_n_jobs(cfg.n_jobs)
    use_parallel = n_workers > 1

    # Cache dir for time series (enables multi-seed reuse)
    ts_cache_dir = str(Path(cfg.output_dir) / "ts_cache" / cfg.atlas)

    if use_parallel:
        logger.info(f"Parallel mode: {n_workers} workers")
        worker_args = [
            (i, func_path, conf_path, atlas_nii_path, t_r,
             cfg.short_duration_sec, cfg.target_duration_min,
             cfg.n_bootstraps, cfg.atlas, cfg.correction_method, grp,
             ts_cache_dir)
            for i, (func_path, conf_path, grp) in enumerate(
                zip(func_list, confounds_list, group_labels)
            )
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_adhd_subject, arg): arg[0]
                for arg in worker_args
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    logger.error(f"Worker exception idx={idx}: {e}")
                    n_fail += 1
                    continue

                if res is None or res.get("status") != "ok":
                    reason = res.get("reason", "unknown") if res else "none"
                    logger.debug(f"idx={idx}: {reason}")
                    n_fail += 1
                    continue

                del res["status"]
                results.append(res)
                n_success += 1

                if cfg.verbose:
                    logger.info(
                        f"[{n_success + n_fail}/{len(func_list)}] "
                        f"r_fc_raw={res['r_fc_raw']:.3f} → "
                        f"ρ̂T={res['rho_hat_T']:.3f} (Δ={res['improvement']:+.3f})"
                    )
    else:
        logger.info("Sequential mode (use --n-jobs to parallelize)")
        for i, (func_path, conf_path) in enumerate(zip(func_list, confounds_list)):
            prefix = f"[{i+1}/{len(func_list)}]"

            ts = extract_timeseries(func_path, conf_path, atlas_img, t_r=t_r)
            if ts is None:
                logger.warning(f"{prefix} extraction failed, skipping")
                n_fail += 1
                continue

            # Cache time series for multi-seed reuse
            _ts_cache = Path(ts_cache_dir)
            _ts_cache.mkdir(parents=True, exist_ok=True)
            np.save(str(_ts_cache / f"sub_{i:03d}_ts.npy"), ts)

            total_sec = ts.shape[0] * t_r
            if total_sec < cfg.short_duration_sec + 20:
                logger.warning(f"{prefix} too short ({total_sec:.0f}s)")
                n_fail += 1
                continue

            try:
                res = run_bsnet_single(
                    ts=ts, tr=t_r,
                    short_sec=cfg.short_duration_sec,
                    target_min=cfg.target_duration_min,
                    n_bootstraps=cfg.n_bootstraps,
                    correction_method=cfg.correction_method,
                )
            except Exception as e:
                logger.error(f"{prefix} BS-NET error: {e}")
                n_fail += 1
                continue

            if res is None:
                n_fail += 1
                continue

            res["sub_idx"] = i
            res["func_path"] = str(Path(func_path).name)
            res["atlas"] = cfg.atlas
            res["group"] = group_labels[i]
            results.append(res)
            n_success += 1

            if cfg.verbose:
                logger.info(
                    f"{prefix} r_fc_raw={res['r_fc_raw']:.3f} → "
                    f"ρ̂T={res['rho_hat_T']:.3f} (Δ={res['improvement']:+.3f})"
                )

    elapsed = time.time() - t_start

    # Step 4: Save results
    csv_path = results_dir / f"adhd_bsnet_{cfg.atlas}.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Results CSV: {csv_path}")

    # Summary
    if results:
        r_raw = np.array([r["r_fc_raw"] for r in results])
        rho_hat = np.array([r["rho_hat_T"] for r in results])
        improvement = np.array([r["improvement"] for r in results])

        summary = {
            "config": {
                "atlas": cfg.atlas,
                "atlas_label": ATLAS_REGISTRY[cfg.atlas]["label"],
                "n_rois": cfg.n_rois,
                "short_duration_sec": cfg.short_duration_sec,
                "target_duration_min": cfg.target_duration_min,
                "n_bootstraps": cfg.n_bootstraps,
            },
            "counts": {
                "total": len(func_list),
                "success": n_success,
                "failed": n_fail,
            },
            "r_fc_raw": {
                "mean": round(float(np.mean(r_raw)), 4),
                "std": round(float(np.std(r_raw)), 4),
                "median": round(float(np.median(r_raw)), 4),
            },
            "rho_hat_T": {
                "mean": round(float(np.mean(rho_hat)), 4),
                "std": round(float(np.std(rho_hat)), 4),
                "median": round(float(np.median(rho_hat)), 4),
            },
            "improvement": {
                "mean": round(float(np.mean(improvement)), 4),
                "std": round(float(np.std(improvement)), 4),
                "median": round(float(np.median(improvement)), 4),
                "pct_improved": round(float(np.mean(improvement > 0)) * 100, 1),
            },
            "elapsed_sec": round(elapsed, 1),
        }

        # Group-level statistics
        groups_in_results = set(r.get("group", "unknown") for r in results)
        group_stats = {}
        for grp in sorted(groups_in_results):
            grp_res = [r for r in results if r.get("group") == grp]
            if not grp_res:
                continue
            grp_r = np.array([r["r_fc_raw"] for r in grp_res])
            grp_rho = np.array([r["rho_hat_T"] for r in grp_res])
            grp_imp = np.array([r["improvement"] for r in grp_res])
            group_stats[grp] = {
                "n": len(grp_res),
                "r_fc_raw": {"mean": round(float(np.mean(grp_r)), 4),
                             "std": round(float(np.std(grp_r)), 4)},
                "rho_hat_T": {"mean": round(float(np.mean(grp_rho)), 4),
                              "std": round(float(np.std(grp_rho)), 4)},
                "improvement": {"mean": round(float(np.mean(grp_imp)), 4),
                                "pct_improved": round(
                                    float(np.mean(grp_imp > 0)) * 100, 1)},
            }
        summary["group_stats"] = group_stats

        summary_path = results_dir / f"adhd_summary_{cfg.atlas}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("")
        logger.info("━" * 50)
        logger.info(f" ADHD BS-NET Results — {ATLAS_REGISTRY[cfg.atlas]['label']}")
        logger.info("━" * 50)
        logger.info(f"  Subjects: {n_success}/{len(func_list)}")
        logger.info(f"  ROIs: {results[0]['n_rois']}")
        logger.info("")
        logger.info("  r_FC (raw, 2min vs full):")
        logger.info(f"    mean={summary['r_fc_raw']['mean']:.3f} ± {summary['r_fc_raw']['std']:.3f}")
        logger.info("  ρ̂T (BS-NET predicted):")
        logger.info(f"    mean={summary['rho_hat_T']['mean']:.3f} ± {summary['rho_hat_T']['std']:.3f}")
        logger.info("  Improvement (ρ̂T - r_FC):")
        logger.info(f"    mean={summary['improvement']['mean']:+.3f} ± {summary['improvement']['std']:.3f}")
        logger.info(f"    {summary['improvement']['pct_improved']}% improved")
        # Group breakdown
        if len(group_stats) > 1:
            logger.info("")
            logger.info("  By group:")
            for grp, gs in group_stats.items():
                logger.info(
                    f"    {grp:>8s} (n={gs['n']:>2d}): "
                    f"r_FC={gs['r_fc_raw']['mean']:.3f} → "
                    f"ρ̂T={gs['rho_hat_T']['mean']:.3f} "
                    f"(Δ={gs['improvement']['mean']:+.3f}, "
                    f"{gs['improvement']['pct_improved']}%↑)"
                )
        logger.info(f"  Time: {elapsed:.0f}s ({elapsed/max(n_success,1):.1f}s/subject)")
        logger.info("━" * 50)

    return csv_path


# ============================================================================
# Multi-seed stability analysis
# ============================================================================
def _multiseed_adhd_worker(args: tuple) -> dict:
    """Worker: run BS-NET with multiple seeds on cached time series.

    Args:
        args: (sub_idx, ts_path, tr, short_sec, n_seeds, n_bootstraps,
               correction_method, group_label, atlas_name).

    Returns:
        Dict with per-seed rho_hat_T, ci bounds, and metadata.
    """
    (sub_idx, ts_path, tr, short_sec, n_seeds, n_bootstraps,
     corr_method, group_label, atlas_name) = args

    ts = np.load(ts_path).astype(np.float64)
    valid = np.std(ts, axis=0) > 1e-8
    ts = ts[:, valid]
    n_rois = ts.shape[1]

    short_vols = int(short_sec / tr)
    ts_short = ts[:short_vols, :]
    fc_full_vec = get_fc_matrix(ts, vectorized=True, use_shrinkage=True)
    fc_short_vec = get_fc_matrix(ts_short, vectorized=True, use_shrinkage=True)
    r_fc_raw = float(np.corrcoef(fc_short_vec, fc_full_vec)[0, 1])

    rho_arr = np.zeros(n_seeds)
    ci_lo_arr = np.zeros(n_seeds)
    ci_hi_arr = np.zeros(n_seeds)

    for s in range(n_seeds):
        seed = 42 + s * 7
        config = BSNetConfig(
            n_rois=n_rois, tr=tr,
            short_duration_sec=short_sec,
            target_duration_min=15,
            n_bootstraps=n_bootstraps, seed=seed,
        )
        result = run_bootstrap_prediction(
            ts_short, fc_full_vec, config,
            correction_method=corr_method,
        )
        rho_arr[s] = float(result.rho_hat_T)
        ci_lo_arr[s] = float(result.ci_lower)
        ci_hi_arr[s] = float(result.ci_upper)

    return {
        "sub_idx": sub_idx,
        "r_fc_raw": round(r_fc_raw, 4),
        "rho_mean": round(float(np.mean(rho_arr)), 4),
        "rho_std": round(float(np.std(rho_arr)), 4),
        "rho_min": round(float(np.min(rho_arr)), 4),
        "rho_max": round(float(np.max(rho_arr)), 4),
        "ci_lower_mean": round(float(np.mean(ci_lo_arr)), 4),
        "ci_upper_mean": round(float(np.mean(ci_hi_arr)), 4),
        "n_rois": n_rois,
        "group": group_label,
        "atlas": atlas_name,
    }


def run_multi_seed_adhd(cfg: ADHDConfig) -> Path:
    """Run multi-seed stability analysis on ADHD dataset.

    Requires a prior single-seed run that cached time series .npy files.
    Re-runs BS-NET with N different seeds per subject.

    Args:
        cfg: Configuration with multi_seed > 0.

    Returns:
        Path to multi-seed results CSV.
    """
    n_seeds = cfg.multi_seed
    results_dir = Path(cfg.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts_cache = Path(cfg.output_dir) / "ts_cache" / cfg.atlas

    if not ts_cache.exists():
        logger.error(
            f"Time series cache not found: {ts_cache}\n"
            "Run single-seed first to build cache."
        )
        return Path()

    # Enumerate cached subjects
    ts_files = sorted(ts_cache.glob("sub_*_ts.npy"))
    if not ts_files:
        logger.error(f"No cached time series in {ts_cache}")
        return Path()

    logger.info(f"Multi-seed analysis: {len(ts_files)} subjects × {n_seeds} seeds")

    # Extract group labels from single-seed CSV
    csv_path = results_dir / f"adhd_bsnet_{cfg.atlas}.csv"
    group_map: dict[int, str] = {}
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                group_map[int(row["sub_idx"])] = row.get("group", "unknown")

    worker_args = []
    for ts_file in ts_files:
        # filename: sub_003_ts.npy
        sub_idx = int(ts_file.stem.split("_")[1])
        grp = group_map.get(sub_idx, "unknown")
        worker_args.append((
            sub_idx, str(ts_file), 2.0, cfg.short_duration_sec,
            n_seeds, cfg.n_bootstraps, cfg.correction_method,
            grp, cfg.atlas,
        ))

    n_workers = _resolve_n_jobs(cfg.n_jobs)
    ms_results = []

    if n_workers > 1:
        logger.info(f"Parallel mode: {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_multiseed_adhd_worker, arg): arg[0]
                for arg in worker_args
            }
            for done_count, future in enumerate(as_completed(futures), 1):
                res = future.result()
                ms_results.append(res)
                if done_count % 10 == 0 or done_count == 1:
                    logger.info(
                        f"  [{done_count}/{len(worker_args)}] sub_{res['sub_idx']:03d}: "
                        f"ρ̂T={res['rho_mean']:.3f}±{res['rho_std']:.3f}"
                    )
    else:
        for arg in worker_args:
            res = _multiseed_adhd_worker(arg)
            ms_results.append(res)

    ms_results.sort(key=lambda x: x["sub_idx"])

    # Save CSV
    ms_csv = results_dir / f"adhd_multiseed_{cfg.atlas}_{n_seeds}seeds.csv"
    if ms_results:
        fieldnames = list(ms_results[0].keys())
        with open(ms_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ms_results)

    # Summary
    rho_means = np.array([r["rho_mean"] for r in ms_results])
    rho_stds = np.array([r["rho_std"] for r in ms_results])
    ceiling_count = int(np.sum(rho_means > 0.999))

    logger.info("")
    logger.info("━" * 60)
    logger.info(f" ADHD Multi-Seed Results — {cfg.atlas} ({n_seeds} seeds)")
    logger.info("━" * 60)
    logger.info(f"  N={len(ms_results)}")
    logger.info(f"  ρ̂T mean={np.mean(rho_means):.3f}±{np.std(rho_means):.3f}")
    logger.info(f"  seed std mean={np.mean(rho_stds):.4f}")
    logger.info(f"  ceiling (>0.999): {ceiling_count}/{len(ms_results)} "
                f"({100*ceiling_count/max(len(ms_results),1):.1f}%)")

    # Group breakdown
    groups = set(r["group"] for r in ms_results)
    if len(groups) > 1:
        logger.info("")
        logger.info("  By group:")
        for grp in sorted(groups):
            grp_res = [r for r in ms_results if r["group"] == grp]
            grp_rho = np.array([r["rho_mean"] for r in grp_res])
            grp_raw = np.array([r["r_fc_raw"] for r in grp_res])
            logger.info(
                f"    {grp:>8s} (n={len(grp_res):>2d}): "
                f"r_FC={np.mean(grp_raw):.3f} → "
                f"ρ̂T={np.mean(grp_rho):.3f}±{np.std(grp_rho):.3f}"
            )

    logger.info("━" * 60)
    logger.info(f"Multi-seed CSV: {ms_csv}")
    return ms_csv


# ============================================================================
# Multi-atlas comparison
# ============================================================================
def run_multi_atlas_comparison(cfg: ADHDConfig) -> None:
    """Run BS-NET across all registered atlases and print comparison table.

    Args:
        cfg: Base configuration (atlas will be overridden per iteration).
    """
    atlas_names = list(ATLAS_REGISTRY.keys())
    all_summaries = {}

    for atlas_name in atlas_names:
        logger.info("")
        logger.info(f"{'='*60}")
        logger.info(f"  Atlas: {atlas_name}")
        logger.info(f"{'='*60}")

        cfg_copy = ADHDConfig(
            atlas=atlas_name,
            group=cfg.group,
            max_subjects=cfg.max_subjects,
            short_duration_sec=cfg.short_duration_sec,
            target_duration_min=cfg.target_duration_min,
            n_bootstraps=cfg.n_bootstraps,
            output_dir=cfg.output_dir,
            verbose=cfg.verbose,
            n_jobs=cfg.n_jobs,
            correction_method=cfg.correction_method,
        )

        run_adhd_validation(cfg_copy)

        # Load summary
        summary_path = Path(cfg.output_dir) / "results" / f"adhd_summary_{atlas_name}.json"
        if summary_path.exists():
            with open(summary_path) as f:
                all_summaries[atlas_name] = json.load(f)

    # Print comparison table
    if len(all_summaries) > 1:
        logger.info("")
        logger.info("=" * 70)
        logger.info(" Multi-Atlas Comparison")
        logger.info("=" * 70)
        logger.info(f"  {'Atlas':<16} {'ROIs':>5} {'r_FC_raw':>10} {'ρ̂T':>10} {'Δ':>10} {'%↑':>6}")
        logger.info("  " + "-" * 60)
        for name, s in all_summaries.items():
            logger.info(
                f"  {name:<16} {s['config']['n_rois']:>5} "
                f"{s['r_fc_raw']['mean']:>10.3f} "
                f"{s['rho_hat_T']['mean']:>10.3f} "
                f"{s['improvement']['mean']:>+10.3f} "
                f"{s['improvement']['pct_improved']:>5.1f}%"
            )
        logger.info("=" * 70)

        # Save comparison
        comp_path = Path(cfg.output_dir) / "results" / "adhd_atlas_comparison.json"
        with open(comp_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        logger.info(f"Comparison saved: {comp_path}")


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> ADHDConfig:
    parser = argparse.ArgumentParser(
        description="BS-NET validation on ADHD resting-state dataset (nilearn)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    atlas_choices = list(ATLAS_REGISTRY.keys()) + ["all"]
    parser.add_argument(
        "--atlas", choices=atlas_choices, default="cc200",
        help="Parcellation atlas (default: cc200). 'all' runs all atlases.",
    )
    parser.add_argument(
        "--group", choices=["controls", "adhd", "both"], default="both",
        help="Subject group filter (default: both)",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=0,
        help="Max subjects (0=all 40, default: 0)",
    )
    parser.add_argument(
        "--short-sec", type=int, default=120,
        help="Short scan duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--n-bootstraps", type=int, default=100,
        help="Bootstrap iterations (default: 100)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/adhd",
        help="Output directory (default: data/adhd)",
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
    parser.add_argument(
        "--multi-seed", type=int, default=0,
        help="Run N seeds for stability analysis (0=single seed, default: 0). "
             "Requires prior single-seed run for time series cache.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    return ADHDConfig(
        atlas=args.atlas if args.atlas != "all" else "cc200",
        group=args.group,
        max_subjects=args.max_subjects,
        short_duration_sec=args.short_sec,
        target_duration_min=15,
        n_bootstraps=args.n_bootstraps,
        output_dir=args.output_dir,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
        correction_method=args.correction_method,
        multi_seed=args.multi_seed,
    ), args.atlas == "all"


def main() -> None:
    cfg, run_all = parse_args()

    log_level = logging.DEBUG if cfg.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if cfg.multi_seed > 0:
        run_multi_seed_adhd(cfg)
    elif run_all:
        run_multi_atlas_comparison(cfg)
    else:
        run_adhd_validation(cfg)


if __name__ == "__main__":
    main()
