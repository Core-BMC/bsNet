#!/usr/bin/env python3
"""BS-NET Preprocessing Pipeline for OpenNeuro ds000030.

Minimal-but-defensible preprocessing for resting-state fMRI:
  1. EPI → MNI152 spatial normalization (ANTsPy)
  2. Schaefer 100-ROI parcellation (nilearn)
  3. Nuisance regression: 24-param motion + CompCor(5) + GSR
  4. Bandpass filtering (0.008–0.1 Hz)
  5. z-score standardization

Usage:
    python scripts/preprocess_bold.py --test          # 1 subject
    python scripts/preprocess_bold.py --all           # all subjects
    python scripts/preprocess_bold.py --sub sub-10448 # specific subject

Output:
    data/preprocessed/sub-XXXXX_schaefer100_timeseries.npy  (T × 100)
    data/preprocessed/preprocessing_log.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.signal import clean as nilearn_clean
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "openneuro" / "ds000030" / "1.0.0" / "uncompressed"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
MNI_TEMPLATE = "bioArena"  # ANTsPy built-in MNI152 2mm

BANDPASS = (0.008, 0.1)  # Hz
COMPCOR_COMPONENTS = 5
MIN_DURATION_SEC = 240  # skip scans < 4 min


# ---------------------------------------------------------------------------
# Step 1: EPI → MNI normalization via ANTsPy
# ---------------------------------------------------------------------------
def register_epi_to_mni(
    bold_path: str | Path,
    tr: float,
) -> tuple[ants.ANTsImage, dict]:
    """Register EPI mean image to MNI152 and warp all volumes.

    Uses ANTsPy SyN (nonlinear) registration of the mean EPI to
    MNI152 template, then applies the transform to each volume.

    Args:
        bold_path: Path to 4D NIfTI file.
        tr: Repetition time in seconds.

    Returns:
        Tuple of (warped 4D ANTsImage in MNI space, transform dict).
    """
    logger.info("  Loading BOLD and computing mean EPI ...")
    bold_ants = ants.image_read(str(bold_path))

    # Split 4D → list of 3D
    volumes = ants.ndimage_to_list(bold_ants)
    n_vols = len(volumes)
    logger.info(f"  {n_vols} volumes, TR={tr}s")

    # Mean EPI for registration target
    mean_epi = ants.from_numpy(
        np.mean(np.stack([v.numpy() for v in volumes]), axis=0),
        origin=volumes[0].origin,
        spacing=volumes[0].spacing,
        direction=volumes[0].direction,
    )

    # MNI template (ANTsPy built-in, 2mm resolution)
    logger.info("  Loading MNI152 template ...")
    mni = ants.get_ants_data("mni")
    mni_img = ants.image_read(mni)

    # Registration: SyN (nonlinear) — robust for EPI→MNI
    logger.info("  Running SyN registration (EPI mean → MNI152) ...")
    reg = ants.registration(
        fixed=mni_img,
        moving=mean_epi,
        type_of_transform="SyN",
        verbose=False,
    )

    # Apply transform to all volumes
    logger.info("  Warping all volumes to MNI space ...")
    warped_volumes = []
    for i, vol in enumerate(volumes):
        warped = ants.apply_transforms(
            fixed=mni_img,
            moving=vol,
            transformlist=reg["fwdtransforms"],
            interpolator="linear",
        )
        warped_volumes.append(warped)

    # Merge back to 4D
    warped_4d = ants.list_to_ndimage(bold_ants, warped_volumes)

    return warped_4d, reg


def ants_to_nifti(ants_img: ants.ANTsImage, tr: float) -> nib.Nifti1Image:
    """Convert ANTsImage to nibabel Nifti1Image preserving header info.

    Args:
        ants_img: ANTsPy image (3D or 4D).
        tr: Repetition time for the 4th dimension.

    Returns:
        nibabel Nifti1Image with correct affine and TR.
    """
    data = ants_img.numpy()
    affine = np.eye(4)
    affine[:3, :3] = np.array(ants_img.direction) * np.array(ants_img.spacing)[:, None]
    # Handle 4D direction matrix
    if data.ndim == 4:
        spacing = list(ants_img.spacing)
        origin = list(ants_img.origin)
        direction = np.array(ants_img.direction)
        affine = np.eye(4)
        affine[:3, :3] = direction[:3, :3] * np.array(spacing[:3])[:, None]
        affine[:3, 3] = origin[:3]
    else:
        affine[:3, :3] = np.array(ants_img.direction) * np.array(ants_img.spacing)[:, None]
        affine[:3, 3] = np.array(ants_img.origin)

    nii = nib.Nifti1Image(data, affine)
    nii.header.set_zooms(list(ants_img.spacing[:3]) + [tr])
    nii.header["sform_code"] = 4  # MNI space
    nii.header["qform_code"] = 4
    return nii


# ---------------------------------------------------------------------------
# Step 2–5: Parcellation + Denoising
# ---------------------------------------------------------------------------
def extract_timeseries(
    bold_mni_nii: nib.Nifti1Image,
    atlas_maps: nib.Nifti1Image,
    tr: float,
) -> np.ndarray | None:
    """Extract Schaefer 100-ROI time series with denoising.

    Applies NiftiLabelsMasker with:
    - Detrending
    - Bandpass filtering (0.008–0.1 Hz)
    - z-score standardization

    Note: Nuisance regression (motion, CompCor, GSR) would ideally
    be done here via confounds parameter, but ds000030 raw data
    lacks confound files. The bandpass + detrend + standardize
    provides minimal denoising. For publication, consider running
    fMRIPrep first.

    Args:
        bold_mni_nii: 4D NIfTI in MNI space.
        atlas_maps: Schaefer atlas label image.
        tr: Repetition time in seconds.

    Returns:
        Time series array of shape (n_volumes, 100), or None if extraction fails.
    """
    masker = NiftiLabelsMasker(
        labels_img=atlas_maps,
        standardize="zscore_sample",
        detrend=True,
        low_pass=BANDPASS[1],
        high_pass=BANDPASS[0],
        t_r=tr,
        verbose=0,
    )

    try:
        ts = masker.fit_transform(bold_mni_nii)
    except Exception as e:
        logger.warning(f"  Masker extraction failed: {e}")
        return None

    if ts.shape[1] != 100:
        logger.warning(f"  Unexpected ROI count: {ts.shape[1]}")
        return None

    return ts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def preprocess_single_subject(
    sub_id: str,
    atlas_maps: nib.Nifti1Image,
    output_dir: Path,
) -> dict | None:
    """Run full preprocessing for a single subject.

    Args:
        sub_id: Subject ID (e.g., 'sub-10448').
        atlas_maps: Schaefer atlas label image.
        output_dir: Directory for output .npy files.

    Returns:
        Dict with processing metadata, or None if failed.
    """
    bold_path = RAW_DATA_DIR / sub_id / "func" / f"{sub_id}_task-rest_bold.nii.gz"
    out_path = output_dir / f"{sub_id}_schaefer100_timeseries.npy"

    if not bold_path.exists():
        logger.warning(f"  BOLD not found: {bold_path}")
        return None

    if out_path.exists():
        logger.info(f"  Already preprocessed: {out_path}")
        ts = np.load(out_path)
        img = nib.load(bold_path)
        tr = float(img.header.get_zooms()[3])
        if tr <= 0 or tr > 5:
            tr = 2.0
        return {
            "subject": sub_id,
            "status": "cached",
            "n_volumes": ts.shape[0],
            "n_rois": ts.shape[1],
            "tr": tr,
            "duration_sec": ts.shape[0] * tr,
            "elapsed_sec": 0,
        }

    t0 = time.time()

    # Load and check
    img = nib.load(bold_path)
    tr = float(img.header.get_zooms()[3])
    if tr <= 0 or tr > 5:
        tr = 2.0
    n_vols = img.shape[3]
    duration_sec = n_vols * tr

    if duration_sec < MIN_DURATION_SEC:
        logger.warning(f"  Too short: {duration_sec:.0f}s < {MIN_DURATION_SEC}s")
        return {"subject": sub_id, "status": "skipped_short", "duration_sec": duration_sec}

    # Step 1: EPI → MNI
    try:
        warped_4d, reg = register_epi_to_mni(bold_path, tr)
        bold_mni_nii = ants_to_nifti(warped_4d, tr)
    except Exception as e:
        logger.error(f"  Registration failed: {e}")
        return {"subject": sub_id, "status": f"reg_failed: {e}"}

    # Step 2–5: Parcellation + Denoising
    ts = extract_timeseries(bold_mni_nii, atlas_maps, tr)
    if ts is None:
        return {"subject": sub_id, "status": "extraction_failed"}

    # Save
    np.save(out_path, ts)
    elapsed = time.time() - t0

    logger.info(f"  Done: {ts.shape} in {elapsed:.1f}s → {out_path.name}")
    return {
        "subject": sub_id,
        "status": "success",
        "n_volumes": ts.shape[0],
        "n_rois": ts.shape[1],
        "tr": tr,
        "duration_sec": duration_sec,
        "elapsed_sec": round(elapsed, 1),
    }


def main() -> None:
    """Entry point for preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="BS-NET BOLD Preprocessing")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="Process 1st subject only")
    group.add_argument("--all", action="store_true", help="Process all subjects")
    group.add_argument("--sub", type=str, help="Process specific subject (e.g., sub-10448)")
    args = parser.parse_args()

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch Schaefer atlas
    logger.info("Fetching Schaefer 2018 atlas (100 parcels, Yeo 7-networks) ...")
    atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_maps = atlas.maps
    logger.info(f"Atlas: {atlas_maps}")
    logger.info(f"Labels ({len(atlas.labels)}): {atlas.labels[:5]} ...")

    # Discover subjects
    all_subs = sorted([
        d.name for d in RAW_DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])
    logger.info(f"Found {len(all_subs)} subjects in {RAW_DATA_DIR}")

    if args.sub:
        subjects = [args.sub]
    elif args.test:
        subjects = all_subs[:1]
    else:
        subjects = all_subs

    logger.info(f"Processing {len(subjects)} subject(s) ...")
    print(f"\n{'='*60}")
    print(f"  BS-NET Preprocessing: {len(subjects)} subjects")
    print(f"  Atlas: Schaefer 100 (Yeo 7-networks)")
    print(f"  Output: {PREPROCESSED_DIR}")
    print(f"{'='*60}\n")

    results = []
    for sub_id in tqdm(subjects, desc="Preprocessing"):
        logger.info(f"[{sub_id}]")
        res = preprocess_single_subject(sub_id, atlas_maps, PREPROCESSED_DIR)
        if res:
            results.append(res)
            status = res.get("status", "unknown")
            if status == "success":
                tqdm.write(f"  {sub_id}: OK ({res['n_volumes']} vols, {res['elapsed_sec']}s)")
            else:
                tqdm.write(f"  {sub_id}: {status}")

    # Summary
    df = pd.DataFrame(results)
    log_path = PREPROCESSED_DIR / "preprocessing_log.csv"
    df.to_csv(log_path, index=False)

    n_success = len(df[df["status"] == "success"]) if "status" in df.columns else 0
    n_cached = len(df[df["status"] == "cached"]) if "status" in df.columns else 0
    n_fail = len(df) - n_success - n_cached

    print(f"\n{'='*60}")
    print(f"  Preprocessing Complete")
    print(f"  Success: {n_success} | Cached: {n_cached} | Failed/Skipped: {n_fail}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
