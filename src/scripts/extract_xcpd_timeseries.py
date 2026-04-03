#!/usr/bin/env python3
"""Extract parcellated timeseries from XCP-D denoised BOLD NIfTI files.

XCP-D crashed at ds_denoised_bold (float32 JSON serialization bug), which
prevented the downstream parcellation step from running.  The denoised BOLD
NIfTIs themselves are intact; this script extracts timeseries using the atlas
NIfTIs from the XCP-D atlases/ directory via nilearn's NiftiLabelsMasker.

Output format is identical to preprocess_ds000243.py and convert_xcpd_to_npy.py:
    data/ds000243/timeseries_cache_xcpd/{atlas_lower}/sub-XXX_{atlas_lower}.npy
    Shape: (n_timepoints, n_rois)  — float32, multi-run concatenated.

Atlas NIfTI discovery (priority order):
    1. --atlas-nifti explicit path per atlas
    2. {xcpd_dir}/atlases/atlas-{name}/*dseg*.nii.gz
    3. {xcpd_dir}/atlases/atlas-{name}/*.nii.gz (first match)

Usage:
    # All subjects, both atlases
    python src/scripts/extract_xcpd_timeseries.py \\
        --xcpd-dir data/derivatives/xcpd \\
        --atlas 4S256Parcels 4S456Parcels \\
        --n-jobs 4

    # Dry-run
    python src/scripts/extract_xcpd_timeseries.py \\
        --xcpd-dir data/derivatives/xcpd \\
        --atlas 4S256Parcels --dry-run

    # Explicit atlas NIfTI path
    python src/scripts/extract_xcpd_timeseries.py \\
        --xcpd-dir data/derivatives/xcpd \\
        --atlas 4S256Parcels \\
        --atlas-nifti 4S256Parcels:/path/to/atlas.nii.gz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# XCP-D CamelCase atlas name → BS-NET lowercase key
ATLAS_NAME_MAP: dict[str, str] = {
    "4S156Parcels":  "4s156parcels",
    "4S256Parcels":  "4s256parcels",
    "4S356Parcels":  "4s356parcels",
    "4S456Parcels":  "4s456parcels",
    "4S556Parcels":  "4s556parcels",
    "4S656Parcels":  "4s656parcels",
    "4S756Parcels":  "4s756parcels",
    "4S856Parcels":  "4s856parcels",
    "4S956Parcels":  "4s956parcels",
    "4S1056Parcels": "4s1056parcels",
    "Schaefer100":   "schaefer100",
    "Schaefer200":   "schaefer200",
    "Schaefer400":   "schaefer400",
    "Craddock200":   "cc200",
    "Craddock400":   "cc400",
}

# ds000243 TR fallback
TR_FALLBACK = 2.5


def _find_atlas_nifti(xcpd_dir: Path, atlas_name: str) -> Path | None:
    """Locate atlas label NIfTI in XCP-D atlases/ directory.

    Args:
        xcpd_dir: XCP-D output root.
        atlas_name: CamelCase atlas name (e.g. ``4S256Parcels``).

    Returns:
        Path to atlas NIfTI, or None if not found.
    """
    atlas_dir = xcpd_dir / "atlases" / f"atlas-{atlas_name}"
    if not atlas_dir.exists():
        logger.error(f"Atlas directory not found: {atlas_dir}")
        return None

    # Prefer dseg (discrete segmentation) NIfTI
    for pat in ["*dseg*.nii.gz", "*dseg*.nii", "*label*.nii.gz", "*.nii.gz", "*.nii"]:
        matches = sorted(atlas_dir.glob(pat))
        if matches:
            logger.debug(f"  Atlas NIfTI: {matches[0]}")
            return matches[0]

    logger.error(f"No NIfTI found in {atlas_dir}")
    return None


def _find_denoised_bolds(func_dir: Path) -> list[Path]:
    """Find all denoised BOLD NIfTIs in a subject func/ directory.

    XCP-D writes:
        sub-XXX_task-rest_run-{N}_space-MNI152NLin6Asym_res-2_desc-denoised_bold.nii.gz

    Args:
        func_dir: XCP-D subject func/ directory.

    Returns:
        Sorted list of denoised BOLD NIfTI paths (run-1 before run-2).
    """
    return sorted(func_dir.glob("*desc-denoised_bold.nii.gz"))


def _extract_timeseries(
    bold_path: Path,
    atlas_nifti: Path,
    sub_id: str,
) -> np.ndarray | None:
    """Extract parcellated timeseries from a single BOLD NIfTI.

    Args:
        bold_path: Denoised BOLD NIfTI path.
        atlas_nifti: Atlas label NIfTI (integer labels, 0 = background).
        sub_id: Subject identifier for logging.

    Returns:
        Array of shape (n_timepoints, n_rois) float32, or None on failure.
    """
    try:
        from nilearn.maskers import NiftiLabelsMasker
    except ImportError:
        logger.error("nilearn required: pip install nilearn --break-system-packages")
        return None

    try:
        masker = NiftiLabelsMasker(
            labels_img=str(atlas_nifti),
            standardize=False,
            resampling_target="data",
            strategy="mean",
            verbose=0,
        )
        ts = masker.fit_transform(str(bold_path))  # (n_tp, n_rois)
        return ts.astype(np.float32)
    except Exception as e:
        logger.error(f"  {sub_id} [{bold_path.name}]: extraction failed — {e}")
        return None


def convert_subject(
    sub_dir: Path,
    atlas_name: str,
    atlas_nifti: Path,
    atlas_lower: str,
    output_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict | None:
    """Extract and save timeseries for one subject.

    Args:
        sub_dir: XCP-D subject directory (``sub-XXX/``).
        atlas_name: CamelCase XCP-D atlas name.
        atlas_nifti: Atlas label NIfTI path.
        atlas_lower: BS-NET lowercase atlas key.
        output_dir: Output directory for .npy files.
        dry_run: Print plan without writing.
        force: Overwrite existing .npy.

    Returns:
        Summary dict, or None on failure.
    """
    sub_id = sub_dir.name
    out_path = output_dir / f"{sub_id}_{atlas_lower}.npy"

    if not force and out_path.exists() and not dry_run:
        logger.info(f"  SKIP {sub_id}: {out_path.name} already exists")
        return {"sub_id": sub_id, "status": "skipped"}

    func_dir = sub_dir / "func"
    if not func_dir.exists():
        logger.debug(f"  SKIP {sub_id}: no func/ directory")
        return None

    bold_files = _find_denoised_bolds(func_dir)
    if not bold_files:
        logger.warning(f"  SKIP {sub_id}: no denoised BOLD found in {func_dir}")
        return None

    if dry_run:
        logger.info(
            f"  DRY-RUN {sub_id}: {len(bold_files)} run(s) → {out_path.name}"
        )
        return {"sub_id": sub_id, "n_runs": len(bold_files), "status": "dry_run"}

    arrays: list[np.ndarray] = []
    for bold in bold_files:
        ts = _extract_timeseries(bold, atlas_nifti, sub_id)
        if ts is not None:
            arrays.append(ts)
            logger.debug(f"    {bold.name}: {ts.shape}")

    if not arrays:
        logger.error(f"  FAIL {sub_id}: all BOLD extractions failed")
        return None

    timeseries = np.concatenate(arrays, axis=0)  # multi-run concat
    n_vols, n_rois = timeseries.shape
    total_sec = n_vols * TR_FALLBACK

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, timeseries)
    logger.info(
        f"  OK {sub_id}: ({n_vols}, {n_rois}), "
        f"total={total_sec:.0f}s [{len(bold_files)} run(s)] → {out_path.name}"
    )
    return {
        "sub_id": sub_id,
        "n_vols": n_vols,
        "n_rois": n_rois,
        "n_runs": len(bold_files),
        "total_sec": total_sec,
        "status": "ok",
    }


def extract_dataset(
    xcpd_dir: Path,
    atlas_name: str,
    atlas_nifti: Path,
    output_base: Path,
    dry_run: bool = False,
    force: bool = False,
    max_subjects: int = 0,
    n_jobs: int = 1,
) -> list[dict]:
    """Extract timeseries for all subjects.

    Args:
        xcpd_dir: XCP-D output root.
        atlas_name: CamelCase XCP-D atlas name.
        atlas_nifti: Atlas label NIfTI path.
        output_base: Base output directory.
        dry_run: Print plan without writing.
        force: Overwrite existing .npy files.
        max_subjects: Limit subjects (0 = all).
        n_jobs: Parallel jobs (joblib if >1).

    Returns:
        List of summary dicts.
    """
    atlas_lower = ATLAS_NAME_MAP.get(atlas_name, atlas_name.lower())
    output_dir = output_base / atlas_lower

    sub_dirs = sorted(
        d for d in xcpd_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    if max_subjects > 0:
        sub_dirs = sub_dirs[:max_subjects]

    logger.info(
        f"[{atlas_name}] {len(sub_dirs)} subjects → {output_dir}"
    )
    logger.info(f"  Atlas NIfTI: {atlas_nifti}")

    def _process(sub_dir: Path) -> dict | None:
        return convert_subject(
            sub_dir, atlas_name, atlas_nifti, atlas_lower,
            output_dir, dry_run=dry_run, force=force,
        )

    results: list[dict] = []

    if n_jobs > 1 and not dry_run:
        try:
            from joblib import Parallel, delayed
            raw = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_process)(d) for d in sub_dirs
            )
            results = [r for r in raw if r is not None]
        except ImportError:
            logger.warning("joblib not found, falling back to serial processing")
            results = [r for d in sub_dirs if (r := _process(d)) is not None]
    else:
        results = [r for d in sub_dirs if (r := _process(d)) is not None]

    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = len(sub_dirs) - len(results)

    print(f"\n[{atlas_name} → {atlas_lower}]")
    print(f"  Extracted : {ok}")
    print(f"  Skipped   : {skipped} (already exist)")
    print(f"  Failed    : {failed}")

    ok_results = [r for r in results if r.get("status") == "ok" and "total_sec" in r]
    if ok_results:
        durations = [r["total_sec"] for r in ok_results]
        rois = ok_results[0]["n_rois"]
        print(
            f"  Duration  : {min(durations):.0f}–{max(durations):.0f}s "
            f"(mean={sum(durations)/len(durations):.0f}s)"
        )
        print(f"  ROIs      : {rois}")
        print(f"  Output    : {output_dir}/")

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract parcellated timeseries from XCP-D denoised BOLD NIfTIs.\n"
            "Use when XCP-D crashed at ds_denoised_bold (float32 JSON bug) "
            "and timeseries TSVs were not generated."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--xcpd-dir", required=True, type=Path,
        help="XCP-D output root (contains sub-XXX/ and atlases/ subdirectories)",
    )
    parser.add_argument(
        "--atlas", nargs="+", default=["4S256Parcels"],
        metavar="ATLAS",
        help=(
            "XCP-D atlas name(s) (CamelCase). "
            f"Known: {list(ATLAS_NAME_MAP.keys())}"
        ),
    )
    parser.add_argument(
        "--atlas-nifti", nargs="+", default=[],
        metavar="NAME:PATH",
        help=(
            "Explicit atlas NIfTI path(s) as NAME:PATH pairs. "
            "Overrides auto-discovery from atlases/. "
            "Example: 4S256Parcels:/path/to/atlas.nii.gz"
        ),
    )
    parser.add_argument(
        "--output-base", type=Path,
        default=Path("data/ds000243/timeseries_cache_xcpd"),
        help="Base output directory (default: data/ds000243/timeseries_cache_xcpd)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without writing files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .npy files",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=0,
        help="Limit to first N subjects (0 = all)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel jobs via joblib (default: 1)",
    )
    args = parser.parse_args()

    if not args.xcpd_dir.exists():
        parser.error(f"XCP-D directory not found: {args.xcpd_dir}")

    # Parse explicit atlas NIfTI overrides
    explicit_niftis: dict[str, Path] = {}
    for item in args.atlas_nifti:
        if ":" not in item:
            parser.error(f"--atlas-nifti must be NAME:PATH, got: {item}")
        name, path_str = item.split(":", 1)
        p = Path(path_str)
        if not p.exists():
            parser.error(f"Atlas NIfTI not found: {p}")
        explicit_niftis[name] = p

    for atlas_name in args.atlas:
        # Resolve atlas NIfTI
        if atlas_name in explicit_niftis:
            atlas_nifti = explicit_niftis[atlas_name]
        else:
            atlas_nifti = _find_atlas_nifti(args.xcpd_dir, atlas_name)

        if atlas_nifti is None:
            logger.error(
                f"Cannot find atlas NIfTI for {atlas_name}. "
                "Use --atlas-nifti to specify explicitly."
            )
            continue

        extract_dataset(
            xcpd_dir=args.xcpd_dir,
            atlas_name=atlas_name,
            atlas_nifti=atlas_nifti,
            output_base=args.output_base,
            dry_run=args.dry_run,
            force=args.force,
            max_subjects=args.max_subjects,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
