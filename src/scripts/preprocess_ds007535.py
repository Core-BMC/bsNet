#!/usr/bin/env python3
"""Preprocess ds007535 (SpeechHemi) for BS-NET duration sweep.

Pipeline:
  1. Load fMRIPrep preprocessed BOLD (MNI space, 2mm)
  2. Load confounds (36P: 6 motion params + derivatives + quadratics + WM/CSF)
  3. Task regression: regress out block-design HRF-convolved task regressors
  4. Bandpass filter: 0.01–0.1 Hz
  5. Parcellation: Schaefer 200 (7-network) → (n_timepoints × n_rois) timeseries
  6. Save as .npy per subject

Input:
  ds007535/ (BIDS derivative from OpenNeuro, fMRIPrep preprocessed)
    sub-XX/func/sub-XX_task-SpeechHemi_bold.nii.gz
    sub-XX/func/sub-XX_task-SpeechHemi_desc-confounds_timeseries.tsv
    sub-XX/func/sub-XX_task-SpeechHemi_bold.json

Output:
  data/ds007535/timeseries_cache/schaefer200/sub-XX_schaefer200.npy

References:
  - Cole et al. (2014): Intrinsic and task-evoked network architectures.
    DOI: 10.1016/j.neuron.2014.05.014
  - Gratton et al. (2018): FC dominated by stable individual factors.
    DOI: 10.1016/j.neuron.2018.03.035

Usage:
    # Process all subjects (requires ds007535 downloaded)
    python src/scripts/preprocess_ds007535.py --input-dir data/ds007535/raw

    # Limit subjects
    python src/scripts/preprocess_ds007535.py --input-dir data/ds007535/raw --max-subjects 5

    # Custom parcellation
    python src/scripts/preprocess_ds007535.py --input-dir data/ds007535/raw --atlas schaefer400
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

# 36P confound columns (Ciric et al., 2017; Satterthwaite et al., 2013)
CONFOUND_COLS_BASE = [
    "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
    "csf", "white_matter",
]

# Task design for SpeechHemi (from dataset description)
# 9 blocks per condition, each block = 8 pairs × 2.5s = 20s
# Rest between blocks = 24s
# Two conditions: semantic, control (alternating)
TASK_BLOCK_DURATION = 20.0  # seconds
TASK_REST_DURATION = 24.0  # seconds
TASK_N_BLOCKS = 9  # per condition, 18 total
TASK_CONDITIONS = ["semantic", "control"]

# Atlas options
ATLAS_MAP = {
    "schaefer200": {
        "nilearn_name": "schaefer_2018",
        "n_rois": 200,
        "kwargs": {"n_rois": 200, "yeo_networks": 7, "resolution_mm": 2},
    },
    "schaefer400": {
        "nilearn_name": "schaefer_2018",
        "n_rois": 400,
        "kwargs": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 2},
    },
}


# ── Confounds loading ────────────────────────────────────────────────────


def load_confounds_36p(confounds_path: Path) -> np.ndarray:
    """Load 36P confound matrix from fMRIPrep TSV.

    36P = 8 base signals (6 motion + WM + CSF)
        + 8 temporal derivatives
        + 16 quadratic terms (8 base² + 8 deriv²)
        = 32 columns (some implementations use 36 with global signal)

    Args:
        confounds_path: Path to desc-confounds_timeseries.tsv.

    Returns:
        (n_timepoints, n_confounds) array with NaN-filled first row.
    """
    import pandas as pd

    df = pd.read_csv(confounds_path, sep="\t")

    cols = []
    for base_col in CONFOUND_COLS_BASE:
        if base_col in df.columns:
            cols.append(base_col)
            # Derivative
            deriv_col = f"{base_col}_derivative1"
            if deriv_col in df.columns:
                cols.append(deriv_col)
            # Power2
            power_col = f"{base_col}_power2"
            if power_col in df.columns:
                cols.append(power_col)
            # Derivative power2
            deriv_power_col = f"{base_col}_derivative1_power2"
            if deriv_power_col in df.columns:
                cols.append(deriv_power_col)

    if not cols:
        logger.warning(f"No confound columns found in {confounds_path}")
        return np.zeros((len(df), 1))

    confounds = df[cols].values.astype(np.float64)
    # Replace NaN (first row of derivatives) with 0
    confounds = np.nan_to_num(confounds, nan=0.0)

    logger.debug(f"Loaded {len(cols)} confound columns from {confounds_path.name}")
    return confounds


# ── Task regressors ──────────────────────────────────────────────────────


def create_task_regressors(
    n_timepoints: int,
    tr: float,
) -> np.ndarray:
    """Create HRF-convolved task regressors for SpeechHemi design.

    Block design: alternating semantic/control blocks with rest periods.
    Each block = 20s task, followed by 24s rest.

    Args:
        n_timepoints: Number of fMRI volumes.
        tr: Repetition time in seconds.

    Returns:
        (n_timepoints, 2) array: [semantic_regressor, control_regressor].
    """
    total_time = n_timepoints * tr
    time_points = np.arange(0, total_time, tr)

    # Build block onsets for alternating design
    # Pattern: rest(24s) - semantic(20s) - rest(24s) - control(20s) - ...
    block_cycle = TASK_BLOCK_DURATION + TASK_REST_DURATION  # 44s
    regressors = np.zeros((n_timepoints, 2))

    for cond_idx, _cond in enumerate(TASK_CONDITIONS):
        boxcar = np.zeros(n_timepoints)
        for block_i in range(TASK_N_BLOCKS):
            # Alternating: even blocks = semantic, odd = control
            block_global_idx = block_i * 2 + cond_idx
            onset = TASK_REST_DURATION + block_global_idx * block_cycle
            offset = onset + TASK_BLOCK_DURATION
            mask = (time_points >= onset) & (time_points < offset)
            boxcar[mask] = 1.0

        # Convolve with canonical HRF
        hrf = _canonical_hrf(tr, duration=32.0)
        convolved = np.convolve(boxcar, hrf)[:n_timepoints]
        # Normalize
        if np.std(convolved) > 1e-8:
            convolved = (convolved - np.mean(convolved)) / np.std(convolved)
        regressors[:, cond_idx] = convolved

    return regressors


def _canonical_hrf(tr: float, duration: float = 32.0) -> np.ndarray:
    """Generate canonical double-gamma HRF (SPM style).

    Args:
        tr: Repetition time (sampling interval).
        duration: HRF duration in seconds.

    Returns:
        1D array of HRF values sampled at TR intervals.
    """
    from scipy.stats import gamma as gamma_dist

    time = np.arange(0, duration, tr)
    # Double gamma parameters (SPM defaults)
    peak1 = gamma_dist.pdf(time, 6.0)  # positive peak
    peak2 = gamma_dist.pdf(time, 16.0)  # undershoot
    hrf = peak1 - (1.0 / 6.0) * peak2
    return hrf / np.max(np.abs(hrf))


# ── Main processing ──────────────────────────────────────────────────────


def process_single_subject(
    bold_path: Path,
    confounds_path: Path,
    json_path: Path | None,
    atlas: str,
    output_dir: Path,
) -> dict | None:
    """Process one subject: confound regression + task removal + parcellation.

    Args:
        bold_path: Path to preprocessed BOLD NIfTI.
        confounds_path: Path to confounds TSV.
        json_path: Path to BOLD sidecar JSON.
        atlas: Atlas name (schaefer200/schaefer400).
        output_dir: Output directory for .npy files.

    Returns:
        Dict with subject info, or None on failure.
    """
    try:
        import nibabel as nib
        from nilearn.datasets import fetch_atlas_schaefer_2018
        from nilearn.maskers import NiftiLabelsMasker
    except ImportError as e:
        logger.error(f"Required package missing: {e}")
        return None

    sub_id = bold_path.name.split("_")[0]  # sub-XX

    # Load TR from sidecar (fallback to 2.0s for ds007535)
    tr = 2.0
    if json_path and json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        tr = meta.get("RepetitionTime", 2.0)

    # Load BOLD
    img = nib.load(bold_path)
    n_timepoints = img.shape[-1]
    logger.info(f"{sub_id}: {n_timepoints} volumes, TR={tr}s, "
                f"total={n_timepoints * tr:.0f}s")

    # Load confounds (36P)
    confounds_36p = load_confounds_36p(confounds_path)

    # Trim confounds if fMRIPrep included dummy scans not present in BOLD
    if confounds_36p.shape[0] > n_timepoints:
        n_trim = confounds_36p.shape[0] - n_timepoints
        logger.debug(f"  Trimming {n_trim} dummy volumes from confounds head")
        confounds_36p = confounds_36p[n_trim:]

    # Create task regressors (HRF-convolved)
    task_regs = create_task_regressors(n_timepoints, tr)

    # Combine: 36P + 2 task regressors
    all_confounds = np.hstack([confounds_36p, task_regs])
    logger.debug(f"  Confound matrix: {all_confounds.shape}")

    # Fetch atlas
    atlas_info = ATLAS_MAP[atlas]
    atlas_data = fetch_atlas_schaefer_2018(**atlas_info["kwargs"])

    # Extract parcellated timeseries with denoising
    masker = NiftiLabelsMasker(
        labels_img=atlas_data["maps"],
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=tr,
    )

    try:
        timeseries = masker.fit_transform(img, confounds=all_confounds)
    except Exception as e:
        logger.error(f"{sub_id}: Masker failed: {e}")
        return None

    n_rois_actual = timeseries.shape[1]
    logger.info(f"  Parcellated: ({n_timepoints}, {n_rois_actual})")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sub_id}_{atlas}.npy"
    np.save(out_path, timeseries.astype(np.float32))

    return {
        "sub_id": sub_id,
        "n_vols": n_timepoints,
        "tr": tr,
        "total_sec": n_timepoints * tr,
        "n_rois": n_rois_actual,
        "ts_path": str(out_path),
    }


def discover_subjects(input_dir: Path) -> list[dict]:
    """Discover ds007535 subjects with required files.

    Args:
        input_dir: Root directory of ds007535 BIDS derivative.

    Returns:
        List of dicts with bold_path, confounds_path, json_path.
    """
    subjects = []
    for sub_dir in sorted(input_dir.glob("sub-*")):
        func_dir = sub_dir / "func"
        if not func_dir.exists():
            continue

        bold_files = list(func_dir.glob("*_bold.nii.gz"))
        if not bold_files:
            continue

        bold_path = bold_files[0]

        # Skip DataLad broken symlinks (annex objects not yet retrieved)
        # Real NIfTI files are >> 1 MB; symlink stubs are ~200 bytes
        try:
            actual_size = bold_path.stat().st_size
        except OSError:
            logger.debug(f"  Skipping {bold_path.name}: broken symlink or unreadable")
            continue
        if actual_size < 1_000_000:
            logger.debug(
                f"  Skipping {bold_path.name}: file too small ({actual_size} bytes) "
                "— likely unretrieved DataLad annex object"
            )
            continue

        stem = bold_path.name.replace("_bold.nii.gz", "")
        confounds_path = func_dir / f"{stem}_desc-confounds_timeseries.tsv"
        json_path = func_dir / f"{stem}_bold.json"

        if confounds_path.exists():
            subjects.append({
                "bold_path": bold_path,
                "confounds_path": confounds_path,
                "json_path": json_path if json_path.exists() else None,
            })

    logger.info(f"Discovered {len(subjects)} subjects in {input_dir}")
    return subjects


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess ds007535 for BS-NET duration sweep"
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Path to ds007535 BIDS derivative (containing sub-XX/func/)",
    )
    parser.add_argument(
        "--atlas", default="schaefer200",
        choices=list(ATLAS_MAP.keys()),
    )
    parser.add_argument(
        "--output-dir", default=None, type=Path,
        help="Output directory (default: data/ds007535/timeseries_cache/{atlas})",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=0,
        help="Limit subjects (0 = all)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (
        Path("data/ds007535/timeseries_cache") / args.atlas
    )

    subjects = discover_subjects(args.input_dir)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]

    results = []
    for i, sub in enumerate(subjects):
        logger.info(f"[{i + 1}/{len(subjects)}] Processing {sub['bold_path'].parent.parent.name}")
        result = process_single_subject(
            bold_path=sub["bold_path"],
            confounds_path=sub["confounds_path"],
            json_path=sub["json_path"],
            atlas=args.atlas,
            output_dir=output_dir,
        )
        if result:
            results.append(result)

    print(f"\nProcessed {len(results)}/{len(subjects)} subjects")
    print(f"Output: {output_dir}")
    if results:
        durations = [r["total_sec"] for r in results]
        print(f"Duration: {min(durations):.0f}–{max(durations):.0f}s "
              f"(mean={np.mean(durations):.0f}s)")
        print(f"ROIs: {results[0]['n_rois']}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
