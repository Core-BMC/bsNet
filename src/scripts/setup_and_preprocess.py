"""Setup atlas + preprocess raw BOLD + extract FC for BS-NET real-data validation.

Usage (local environment):
    # Step 1: Install dependencies
    pip install nilearn nibabel dipy scikit-learn templateflow pandas numpy

    # Step 2: Run setup (downloads atlas) + preprocess 1 subject test
    python -m src.scripts.setup_and_preprocess --test-one

    # Step 3: Run all 100 subjects
    python -m src.scripts.setup_and_preprocess --run-all

Outputs:
    data/atlas/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz
    data/derivatives/sub-{id}/
        sub-{id}_fc_full.npy        (100x100 FC matrix, full scan)
        sub-{id}_fc_short.npy       (100x100 FC matrix, first 2 min)
        sub-{id}_ts_mni.npy         (n_volumes x 100 time series)
        sub-{id}_quality.json       (QC metrics)
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ATLAS_DIR = DATA_DIR / "atlas"
DERIV_DIR = DATA_DIR / "derivatives"
RAW_DIR = DATA_DIR / "openneuro" / "ds000030" / "1.0.0" / "uncompressed"

TR = 2.0  # seconds
SHORT_DURATION_SEC = 120  # 2 minutes
SHORT_VOLUMES = int(SHORT_DURATION_SEC / TR)  # 60 volumes


def setup_atlas() -> Path:
    """Download Schaefer 100-parcel atlas if not present.

    Tries templateflow first, falls back to nilearn, then direct URL.

    Returns:
        Path to atlas NIfTI file.
    """
    atlas_file = ATLAS_DIR / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
    if atlas_file.exists():
        logger.info(f"Atlas already exists: {atlas_file}")
        return atlas_file

    ATLAS_DIR.mkdir(parents=True, exist_ok=True)

    # Method 1: templateflow
    try:
        from templateflow import api as tflow

        path = tflow.get(
            "MNI152NLin6Asym",
            atlas="Schaefer2018",
            desc="100Parcels7Networks",
            resolution=2,
            suffix="dseg",
            extension=".nii.gz",
        )
        import shutil

        shutil.copy2(str(path), str(atlas_file))
        logger.info(f"Atlas downloaded via templateflow: {atlas_file}")
        return atlas_file
    except Exception as e:
        logger.warning(f"templateflow failed: {e}")

    # Method 2: nilearn
    try:
        import nilearn.datasets as ds

        atlas = ds.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
        import shutil

        shutil.copy2(str(atlas.maps), str(atlas_file))
        logger.info(f"Atlas downloaded via nilearn: {atlas_file}")
        return atlas_file
    except Exception as e:
        logger.warning(f"nilearn failed: {e}")

    # Method 3: direct URL
    try:
        import urllib.request

        url = (
            "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/"
            "master/stable_projects/brain_parcellation/"
            "Schaefer2018_LocalGlobal/Parcellations/MNI/"
            "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
        )
        logger.info(f"Downloading from GitHub: {url}")
        urllib.request.urlretrieve(url, str(atlas_file))
        logger.info(f"Atlas downloaded: {atlas_file}")
        return atlas_file
    except Exception as e:
        logger.error(f"All download methods failed: {e}")
        logger.error(
            "Please manually download the Schaefer atlas and place it at:\n"
            f"  {atlas_file}\n"
            "Download from: https://github.com/ThomasYeoLab/CBIG/tree/master/"
            "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI"
        )
        sys.exit(1)

    return atlas_file


def register_epi_to_mni(
    epi_path: str | Path,
    template_img: nib.Nifti1Image,
) -> tuple[np.ndarray, np.ndarray]:
    """Register mean EPI to MNI space using dipy affine registration.

    Args:
        epi_path: Path to 4D BOLD NIfTI.
        template_img: MNI152 template image.

    Returns:
        Tuple of (affine_matrix, registered_mean_epi_data).
    """
    from dipy.align.imaffine import (
        AffineRegistration,
        MutualInformationMetric,
        transform_centers_of_mass,
    )
    from dipy.align.transforms import AffineTransform3D, RigidTransform3D
    from nilearn import image as nimg

    mean_epi = nimg.mean_img(str(epi_path))

    static = template_img.get_fdata()
    static_affine = template_img.affine
    moving = mean_epi.get_fdata()
    moving_affine = mean_epi.affine

    # Center of mass → Rigid (6-DOF) → Affine (12-DOF)
    c_of_mass = transform_centers_of_mass(static, static_affine, moving, moving_affine)

    metric = MutualInformationMetric(nbins=32, sampling_proportion=0.5)
    affreg = AffineRegistration(metric=metric, level_iters=[1000, 100, 10])

    rigid = affreg.optimize(
        static, moving, RigidTransform3D(), None,
        static_affine, moving_affine,
        starting_affine=c_of_mass.affine,
    )
    affine_result = affreg.optimize(
        static, moving, AffineTransform3D(), None,
        static_affine, moving_affine,
        starting_affine=rigid.affine,
    )

    transformed = affine_result.transform(moving)
    return affine_result.affine, transformed


def transform_4d_to_mni(
    epi_path: str | Path,
    affine_matrix: np.ndarray,
    template_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Apply affine transform to all volumes of a 4D EPI.

    Uses memory-efficient volume-by-volume processing.

    Args:
        epi_path: Path to 4D BOLD NIfTI.
        affine_matrix: 4x4 affine transform (EPI → MNI).
        template_img: MNI152 template (defines output grid).

    Returns:
        4D NIfTI image in MNI space.
    """
    from dipy.align.imaffine import AffineMap

    epi_img = nib.load(str(epi_path))
    epi_data = epi_img.get_fdata()
    n_vols = epi_data.shape[-1]

    aff_map = AffineMap(
        affine_matrix,
        template_img.shape, template_img.affine,
        epi_img.shape[:3], epi_img.affine,
    )

    # Process volume by volume to save memory
    out_shape = template_img.shape + (n_vols,)
    registered = np.zeros(out_shape, dtype=np.float32)

    for t in range(n_vols):
        registered[..., t] = aff_map.transform(epi_data[..., t]).astype(np.float32)

    return nib.Nifti1Image(registered, template_img.affine)


def extract_timeseries(
    bold_mni: nib.Nifti1Image,
    atlas_path: str | Path,
) -> np.ndarray:
    """Extract ROI time series using Schaefer atlas.

    Applies: detrend, zscore standardization, bandpass (0.01–0.1 Hz).

    Args:
        bold_mni: 4D BOLD in MNI space.
        atlas_path: Path to atlas NIfTI.

    Returns:
        Time series array (n_volumes, n_rois).
    """
    from nilearn.maskers import NiftiLabelsMasker

    masker = NiftiLabelsMasker(
        labels_img=str(atlas_path),
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=TR,
    )
    ts = masker.fit_transform(bold_mni)
    return ts


def compute_fc(ts: np.ndarray, use_shrinkage: bool = True) -> np.ndarray:
    """Compute functional connectivity matrix.

    Args:
        ts: Time series (n_volumes, n_rois).
        use_shrinkage: Use Ledoit-Wolf shrinkage estimation.

    Returns:
        FC matrix (n_rois, n_rois).
    """
    if use_shrinkage:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf()
        cov = lw.fit(ts).covariance_
        d = np.sqrt(np.diag(cov))
        d[d == 0] = 1e-10
        fc = cov / np.outer(d, d)
        fc = np.clip(fc, -1.0, 1.0)
    else:
        fc = np.corrcoef(ts.T)

    np.fill_diagonal(fc, 0)
    return fc


def compute_quality_metrics(
    ts: np.ndarray,
    fc: np.ndarray,
) -> dict:
    """Compute QC metrics for a subject.

    Args:
        ts: Time series (n_volumes, n_rois).
        fc: FC matrix (n_rois, n_rois).

    Returns:
        Dict of QC metrics.
    """
    triu = np.triu_indices(fc.shape[0], k=1)
    fc_upper = fc[triu]
    return {
        "n_volumes": int(ts.shape[0]),
        "n_rois_nonzero": int((ts.std(axis=0) > 0).sum()),
        "mean_fc": float(np.mean(fc_upper)),
        "std_fc": float(np.std(fc_upper)),
        "median_fc": float(np.median(fc_upper)),
        "ts_mean_std": float(np.mean(ts.std(axis=0))),
    }


def process_single_subject(
    sub_id: str,
    atlas_path: Path,
    template_img: nib.Nifti1Image,
) -> dict | None:
    """Full pipeline for one subject: register → extract → FC.

    Args:
        sub_id: Subject ID (e.g., 'sub-10448').
        atlas_path: Path to Schaefer atlas.
        template_img: MNI152 template.

    Returns:
        QC metrics dict or None on failure.
    """
    bold_path = RAW_DIR / sub_id / "func" / f"{sub_id}_task-rest_bold.nii.gz"
    if not bold_path.exists():
        logger.error(f"BOLD not found: {bold_path}")
        return None

    out_dir = DERIV_DIR / sub_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    if (out_dir / f"{sub_id}_fc_full.npy").exists():
        logger.info(f"  {sub_id}: already processed, skipping")
        qc_file = out_dir / f"{sub_id}_quality.json"
        if qc_file.exists():
            return json.loads(qc_file.read_text())
        return {"status": "previously_completed"}

    try:
        # Step 1: Register mean EPI → MNI
        logger.info(f"  {sub_id}: registering to MNI...")
        aff_matrix, _ = register_epi_to_mni(bold_path, template_img)

        # Step 2: Transform full 4D to MNI
        logger.info(f"  {sub_id}: transforming 4D to MNI...")
        bold_mni = transform_4d_to_mni(bold_path, aff_matrix, template_img)

        # Step 3: Extract time series
        logger.info(f"  {sub_id}: extracting time series...")
        ts = extract_timeseries(bold_mni, atlas_path)
        np.save(out_dir / f"{sub_id}_ts_mni.npy", ts)

        # Step 4: Compute FC (full scan)
        logger.info(f"  {sub_id}: computing FC (full={ts.shape[0]} vols)...")
        fc_full = compute_fc(ts, use_shrinkage=True)
        np.save(out_dir / f"{sub_id}_fc_full.npy", fc_full)

        # Step 5: Compute FC (short = first 2 min)
        n_short = min(SHORT_VOLUMES, ts.shape[0])
        ts_short = ts[:n_short, :]
        fc_short = compute_fc(ts_short, use_shrinkage=True)
        np.save(out_dir / f"{sub_id}_fc_short.npy", fc_short)

        # Step 6: QC metrics
        qc = compute_quality_metrics(ts, fc_full)
        qc["sub_id"] = sub_id
        qc["status"] = "success"
        qc["n_volumes_short"] = n_short
        with open(out_dir / f"{sub_id}_quality.json", "w") as f:
            json.dump(qc, f, indent=2)

        logger.info(
            f"  {sub_id}: DONE — FC mean={qc['mean_fc']:.3f}, "
            f"ROIs={qc['n_rois_nonzero']}/100"
        )
        return qc

    except Exception as e:
        logger.error(f"  {sub_id}: FAILED — {e}")
        qc = {"sub_id": sub_id, "status": "failed", "error": str(e)}
        with open(out_dir / f"{sub_id}_quality.json", "w") as f:
            json.dump(qc, f, indent=2)
        return qc


def get_subject_list() -> list[str]:
    """Get sorted list of subject IDs from raw data directory."""
    return sorted([
        d.name for d in RAW_DIR.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BS-NET real data preprocessing")
    parser.add_argument("--test-one", action="store_true", help="Process only first subject")
    parser.add_argument("--run-all", action="store_true", help="Process all subjects")
    parser.add_argument("--subject", type=str, help="Process specific subject ID")
    args = parser.parse_args()

    # Step 0: Setup atlas
    logger.info("=== Setting up Schaefer 100 atlas ===")
    atlas_path = setup_atlas()

    # Verify atlas
    atlas_img = nib.load(str(atlas_path))
    labels = np.unique(atlas_img.get_fdata())
    logger.info(f"Atlas: {atlas_img.shape}, {len(labels)-1} parcels")

    # Load MNI template
    from nilearn.datasets import load_mni152_template

    template = load_mni152_template(resolution=2)
    logger.info(f"Template: {template.shape}")

    # Get subjects
    subjects = get_subject_list()
    logger.info(f"Found {len(subjects)} subjects")

    if args.subject:
        subjects = [args.subject]
    elif args.test_one:
        subjects = subjects[:1]
    elif not args.run_all:
        logger.info("Use --test-one, --run-all, or --subject SUB_ID")
        return

    # Process
    results = []
    for i, sub in enumerate(subjects):
        logger.info(f"\n[{i+1}/{len(subjects)}] Processing {sub}")
        qc = process_single_subject(sub, atlas_path, template)
        if qc:
            results.append(qc)

    # Summary
    successes = [r for r in results if r.get("status") == "success"]
    failures = [r for r in results if r.get("status") == "failed"]
    logger.info(f"\n=== SUMMARY: {len(successes)} success, {len(failures)} failed ===")

    if successes:
        mean_fcs = [r["mean_fc"] for r in successes]
        logger.info(f"Mean FC across subjects: {np.mean(mean_fcs):.3f} ± {np.std(mean_fcs):.3f}")

    # Save summary
    DERIV_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DERIV_DIR / "preprocessing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
