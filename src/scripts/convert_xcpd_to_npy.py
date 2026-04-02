#!/usr/bin/env python3
"""Convert XCP-D parcellated timeseries TSV outputs to BS-NET .npy format.

XCP-D outputs parcellated timeseries as TSV files per subject:
    {xcpd_dir}/sub-XXX/func/sub-XXX_task-rest_atlas-Schaefer200_timeseries.tsv

This script converts those TSV files to the .npy format expected by
run_duration_sweep.py and BS-NET pipelines:
    data/ds000243/timeseries_cache_xcpd/{atlas}/sub-XXX_{atlas}.npy

Shape: (n_timepoints, n_rois)  — identical to preprocess_ds000243.py output.

Multi-run handling:
  - With ``--combine-runs y`` in XCP-D: single TSV per subject (no run entity)
  - Without: multiple TSVs per subject → concatenated along time axis

Usage:
    # All subjects, Schaefer200 atlas
    python src/scripts/convert_xcpd_to_npy.py \\
        --xcpd-dir data/ds000243/results/xcpd \\
        --atlas Schaefer200

    # Multiple atlases in one call
    python src/scripts/convert_xcpd_to_npy.py \\
        --xcpd-dir data/ds000243/results/xcpd \\
        --atlas Schaefer200 Schaefer400

    # Dry-run: print what would be done
    python src/scripts/convert_xcpd_to_npy.py \\
        --xcpd-dir data/ds000243/results/xcpd \\
        --atlas Schaefer200 --dry-run

Notes:
    - XCP-D atlas names are CamelCase (e.g. ``Schaefer200``).
      The output .npy uses lowercase snake_case (e.g. ``schaefer200``)
      to match the existing timeseries_cache naming convention.
    - Output directory: ``data/ds000243/timeseries_cache_xcpd/{atlas_lower}/``
    - TR is read from the XCP-D BOLD JSON sidecar if present; falls back to
      ``TR_FALLBACK`` (2.5 s for ds000243).
    - Rows with all-NaN values (censored volumes) are dropped before saving.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ds000243 TR fallback (confirmed from NIfTI header)
TR_FALLBACK = 2.5

# Map: XCP-D CamelCase atlas name → BS-NET lowercase key
# Extend as needed when more atlases are added.
ATLAS_NAME_MAP: dict[str, str] = {
    "Schaefer100":  "schaefer100",
    "Schaefer200":  "schaefer200",
    "Schaefer300":  "schaefer300",
    "Schaefer400":  "schaefer400",
    "Schaefer500":  "schaefer500",
    "Schaefer600":  "schaefer600",
    "Schaefer800":  "schaefer800",
    "Schaefer1000": "schaefer1000",
    "AAL":          "aal",
    "HarvardOxford": "harvard_oxford",
    "Craddock200":  "cc200",
    "Craddock400":  "cc400",
    # XCP-D v26.x built-in 4S series (Schaefer cortical + Tian subcortical)
    # NIfTI 모드에서 Schaefer200/400 대신 이 이름을 사용해야 함
    # 4S256 = Schaefer 200 + 56 subcortical, 4S456 = Schaefer 400 + 56 subcortical
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
}


def _xcpd_ts_glob(func_dir: Path, atlas_xcpd: str) -> list[Path]:
    """Find all timeseries TSV files for the given atlas in a func/ directory.

    XCP-D filenames follow the pattern:
        sub-XXX_[task-YYY_][run-N_]atlas-{atlas}_timeseries.tsv

    Args:
        func_dir: Subject func/ directory from XCP-D output.
        atlas_xcpd: XCP-D atlas name (CamelCase, e.g. ``Schaefer200``).

    Returns:
        Sorted list of matching TSV paths.
    """
    matches = sorted(func_dir.glob(f"*atlas-{atlas_xcpd}_timeseries.tsv"))
    return matches


def _read_timeseries_tsv(tsv_path: Path) -> np.ndarray | None:
    """Read XCP-D timeseries TSV into a (n_timepoints, n_rois) float32 array.

    Censored (all-NaN) rows are dropped.  Partial NaN rows are zero-filled
    with a warning.

    Args:
        tsv_path: Path to timeseries TSV.

    Returns:
        Array of shape (n_timepoints, n_rois), or None on failure.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas required: pip install pandas --break-system-packages")
        return None

    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except Exception as e:
        logger.error(f"Failed to read {tsv_path.name}: {e}")
        return None

    arr = df.values.astype(np.float64)

    # Drop fully-censored rows (all NaN)
    all_nan_rows = np.all(np.isnan(arr), axis=1)
    if all_nan_rows.any():
        n_drop = int(all_nan_rows.sum())
        logger.debug(f"  Dropping {n_drop} censored (all-NaN) rows from {tsv_path.name}")
        arr = arr[~all_nan_rows]

    # Warn and zero-fill partial NaN columns
    partial_nan = np.isnan(arr).any(axis=0)
    if partial_nan.any():
        n_cols = int(partial_nan.sum())
        logger.warning(
            f"  {tsv_path.name}: {n_cols} ROI columns contain partial NaN "
            "— zero-filling. Check atlas coverage."
        )
        arr = np.nan_to_num(arr, nan=0.0)

    if arr.shape[0] == 0:
        logger.error(f"  {tsv_path.name}: no valid timepoints after censoring")
        return None

    return arr.astype(np.float32)


def _resolve_tr(func_dir: Path, sub_id: str) -> float:
    """Try to read TR from an XCP-D BOLD JSON sidecar; fall back to constant.

    Args:
        func_dir: Subject func/ directory.
        sub_id: Subject identifier (for logging).

    Returns:
        TR in seconds.
    """
    json_candidates = sorted(func_dir.glob("*_bold.json"))
    for jp in json_candidates:
        try:
            with open(jp) as f:
                meta = json.load(f)
            tr = float(meta.get("RepetitionTime", 0))
            if tr > 0:
                logger.debug(f"  {sub_id}: TR={tr}s from {jp.name}")
                return tr
        except Exception:
            continue
    logger.debug(f"  {sub_id}: TR sidecar not found, using fallback {TR_FALLBACK}s")
    return TR_FALLBACK


def convert_subject(
    sub_dir: Path,
    atlas_xcpd: str,
    atlas_lower: str,
    output_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict | None:
    """Convert one subject's XCP-D timeseries TSV to .npy.

    Args:
        sub_dir: Subject directory in XCP-D output (e.g. ``xcp_d/sub-001/``).
        atlas_xcpd: XCP-D atlas name (CamelCase).
        atlas_lower: BS-NET lowercase atlas key for output filename.
        output_dir: Destination directory for .npy files.
        dry_run: If True, print plan without writing.
        force: Overwrite existing .npy files.

    Returns:
        Summary dict, or None if conversion failed.
    """
    sub_id = sub_dir.name
    out_path = output_dir / f"{sub_id}_{atlas_lower}.npy"

    if not force and out_path.exists() and not dry_run:
        logger.info(f"  SKIP {sub_id}: already exists ({out_path.name})")
        return {"sub_id": sub_id, "status": "skipped"}

    func_dir = sub_dir / "func"
    if not func_dir.exists():
        logger.debug(f"  SKIP {sub_id}: no func/ directory")
        return None

    tsv_files = _xcpd_ts_glob(func_dir, atlas_xcpd)
    if not tsv_files:
        logger.debug(
            f"  SKIP {sub_id}: no atlas-{atlas_xcpd}_timeseries.tsv found"
        )
        return None

    tr = _resolve_tr(func_dir, sub_id)

    # Read and concatenate across runs (sorted → run-1 before run-2)
    arrays: list[np.ndarray] = []
    for tsv in tsv_files:
        arr = _read_timeseries_tsv(tsv)
        if arr is not None:
            arrays.append(arr)

    if not arrays:
        logger.error(f"  FAIL {sub_id}: all TSV reads failed")
        return None

    timeseries = np.concatenate(arrays, axis=0)
    n_vols, n_rois = timeseries.shape
    total_sec = n_vols * tr

    if dry_run:
        logger.info(
            f"  DRY-RUN {sub_id}: ({n_vols}, {n_rois}), "
            f"total={total_sec:.0f}s → {out_path}"
        )
        return {
            "sub_id": sub_id, "n_vols": n_vols, "n_rois": n_rois,
            "tr": tr, "total_sec": total_sec, "status": "dry_run",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, timeseries)
    logger.info(
        f"  OK {sub_id}: ({n_vols}, {n_rois}), "
        f"total={total_sec:.0f}s → {out_path.name}"
    )
    return {
        "sub_id": sub_id, "n_vols": n_vols, "n_rois": n_rois,
        "tr": tr, "total_sec": total_sec, "status": "ok",
    }


def convert_dataset(
    xcpd_dir: Path,
    atlas_xcpd: str,
    output_base: Path,
    dry_run: bool = False,
    force: bool = False,
    max_subjects: int = 0,
) -> list[dict]:
    """Convert all subjects in an XCP-D output directory.

    Args:
        xcpd_dir: XCP-D output root (contains sub-XXX/ subdirectories).
        atlas_xcpd: XCP-D atlas name (CamelCase, e.g. ``Schaefer200``).
        output_base: Base for output: ``{output_base}/{atlas_lower}/``
        dry_run: Print plan without writing.
        force: Overwrite existing .npy files.
        max_subjects: Limit subjects (0 = all).

    Returns:
        List of summary dicts for processed subjects.
    """
    atlas_lower = ATLAS_NAME_MAP.get(atlas_xcpd)
    if atlas_lower is None:
        logger.warning(
            f"Atlas '{atlas_xcpd}' not in ATLAS_NAME_MAP; "
            "using lowercased name as fallback."
        )
        atlas_lower = atlas_xcpd.lower()

    output_dir = output_base / atlas_lower

    sub_dirs = sorted(d for d in xcpd_dir.iterdir() if d.is_dir() and d.name.startswith("sub-"))
    if max_subjects > 0:
        sub_dirs = sub_dirs[:max_subjects]

    logger.info(
        f"[{atlas_xcpd}] Converting {len(sub_dirs)} subjects → {output_dir}"
    )

    results = []
    for sub_dir in sub_dirs:
        r = convert_subject(
            sub_dir, atlas_xcpd, atlas_lower, output_dir,
            dry_run=dry_run, force=force,
        )
        if r is not None:
            results.append(r)

    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    failed = len(sub_dirs) - len(results)

    print(f"\n[{atlas_xcpd} → {atlas_lower}]")
    print(f"  Converted : {ok}")
    print(f"  Skipped   : {skipped} (already exist)")
    print(f"  Failed    : {failed}")
    if results:
        durations = [r["total_sec"] for r in results if "total_sec" in r]
        if durations:
            print(
                f"  Duration  : {min(durations):.0f}–{max(durations):.0f}s "
                f"(mean={sum(durations)/len(durations):.0f}s)"
            )
        print(f"  Output    : {output_dir}/")

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert XCP-D parcellated timeseries TSV → BS-NET .npy"
    )
    parser.add_argument(
        "--xcpd-dir", required=True, type=Path,
        help=(
            "XCP-D output root directory containing sub-XXX/ subdirectories. "
            "Example: data/ds000243/results/xcpd"
        ),
    )
    parser.add_argument(
        "--atlas", nargs="+", default=["Schaefer200"],
        metavar="ATLAS",
        help=(
            "XCP-D atlas name(s) in CamelCase (default: Schaefer200). "
            f"Known names: {list(ATLAS_NAME_MAP.keys())}"
        ),
    )
    parser.add_argument(
        "--output-base", type=Path, default=None,
        help=(
            "Base output directory. "
            "Default: data/ds000243/timeseries_cache_xcpd"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print conversion plan without writing files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .npy files",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=0,
        help="Limit to first N subjects (0 = all)",
    )
    args = parser.parse_args()

    if not args.xcpd_dir.exists():
        parser.error(f"XCP-D directory not found: {args.xcpd_dir}")

    output_base = args.output_base or Path("data/ds000243/timeseries_cache_xcpd")

    for atlas_xcpd in args.atlas:
        convert_dataset(
            xcpd_dir=args.xcpd_dir,
            atlas_xcpd=atlas_xcpd,
            output_base=output_base,
            dry_run=args.dry_run,
            force=args.force,
            max_subjects=args.max_subjects,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
