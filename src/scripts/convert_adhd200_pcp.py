#!/usr/bin/env python3
"""Convert ADHD-200 PCP CC200 timeseries (.1D) to .npy + build subject JSON.

Reads the extracted ADHD200_CC200_TCs_filtfix.tar.gz output and:
  1. Parses per-site phenotypic CSVs
  2. Converts filtered .1D timeseries (sfnwmrda*) to .npy arrays
  3. Concatenates multi-session runs per subject
  4. Generates BS-NET compatible subject JSON

Input structure (after tar extraction):
  data/adhd/pcp/raw/{site}/{sub_id}/sfnwmrda{sub_id}_session_{N}_rest_{M}_cc200_TCs.1D

Output:
  data/adhd/pcp/timeseries_cache/cc200/{sub_id}_cc200.npy
  data/adhd/pcp/results/adhd200_subjects_cc200.json

File format (.1D from Athena pipeline):
  - Tab-separated, 1 header row
  - Col 0: filename, Col 1: sub-brick index, Cols 2+: ROI timeseries (190 ROIs)
  - sfnwmrda = spatially filtered, nuisance-regressed, de-trended, z-scored
  - snwmrda = same without bandpass filter (we use sfnwmrda)

Phenotypic DX coding:
  0 = TDC (typically developing control)
  1 = ADHD-Combined
  2 = ADHD-Hyperactive/Impulsive
  3 = ADHD-Inattentive

Usage:
    PYTHONPATH=. python src/scripts/convert_adhd200_pcp.py

    # Only first session per subject (no concatenation)
    PYTHONPATH=. python src/scripts/convert_adhd200_pcp.py --first-session-only

    # Verbose
    PYTHONPATH=. python src/scripts/convert_adhd200_pcp.py -v
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

RAW_DIR = Path("data/adhd/pcp/raw")
OUT_DIR = Path("data/adhd/pcp")

SITES = ["KKI", "NeuroIMAGE", "NYU", "OHSU", "Peking_1", "Peking_2", "Peking_3", "Pittsburgh", "WashU"]

# TR per site (seconds) — from ADHD-200 documentation
SITE_TR: dict[str, float] = {
    "KKI": 2.5,
    "NeuroIMAGE": 1.96,
    "NYU": 2.0,
    "OHSU": 2.5,
    "Peking_1": 2.0,
    "Peking_2": 2.0,
    "Peking_3": 2.0,
    "Pittsburgh": 1.5,
    "WashU": 2.5,
}

DX_MAP = {
    "0": "Control",
    "1": "ADHD",    # ADHD-Combined
    "2": "ADHD",    # ADHD-HI
    "3": "ADHD",    # ADHD-Inattentive
}


# ═══════════════════════════════════════════════════════════════════════
# Phenotypic parsing
# ═══════════════════════════════════════════════════════════════════════

def load_all_phenotypic(raw_dir: Path) -> dict[str, dict]:
    """Load phenotypic data from all sites.

    Returns:
        Dict keyed by subject ID with phenotypic metadata.
    """
    pheno = {}
    for site in SITES:
        csv_path = raw_dir / site / f"{site}_phenotypic.csv"
        if not csv_path.exists():
            logger.warning(f"Missing phenotypic: {csv_path}")
            continue

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sub_id = str(row.get("ScanDir ID", "")).strip()
                if not sub_id:
                    continue

                dx = str(row.get("DX", "")).strip()
                group = DX_MAP.get(dx, "Unknown")
                age_str = str(row.get("Age", "")).strip()
                try:
                    age = float(age_str)
                except (ValueError, TypeError):
                    age = -1.0

                gender = str(row.get("Gender", "")).strip()
                sex = "M" if gender == "1" else "F" if gender == "0" else "Unknown"

                # QC flags
                qc_rest = str(row.get("QC_Rest_1", "")).strip()

                pheno[sub_id] = {
                    "sub_id": sub_id,
                    "site": site,
                    "group": group,
                    "dx": int(dx) if dx.isdigit() else -1,
                    "age": age,
                    "sex": sex,
                    "qc_rest_1": qc_rest,
                }

    logger.info(f"Loaded phenotypic data: {len(pheno)} subjects across {len(SITES)} sites")
    return pheno


# ═══════════════════════════════════════════════════════════════════════
# .1D parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_1d_file(path: Path) -> np.ndarray | None:
    """Parse Athena .1D timeseries file.

    Format: tab-separated, 1 header row.
      Col 0: filename, Col 1: sub-brick index, Cols 2+: ROI values

    Returns:
        Array of shape (n_trs, n_rois), or None on failure.
    """
    try:
        lines = path.read_text().strip().split("\n")
        if len(lines) < 2:
            return None

        # Skip header
        data_lines = lines[1:]
        ts = []
        for line in data_lines:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            # Skip first 2 columns (filename, sub-brick)
            vals = [float(v) for v in parts[2:]]
            ts.append(vals)

        arr = np.array(ts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 10:
            return None

        return arr

    except Exception as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return None


def discover_1d_files(raw_dir: Path, site: str, sub_id: str) -> list[Path]:
    """Find all filtered .1D files for a subject, sorted by session/rest.

    Returns list of paths sorted by (session, rest) order.
    """
    sub_dir = raw_dir / site / sub_id
    if not sub_dir.exists():
        return []

    files = sorted(sub_dir.glob(f"sfnwmrda{sub_id}_session_*_rest_*_cc200_TCs.1D"))
    return files


# ═══════════════════════════════════════════════════════════════════════
# Conversion pipeline
# ═══════════════════════════════════════════════════════════════════════

def convert_all(
    raw_dir: Path = RAW_DIR,
    out_dir: Path = OUT_DIR,
    first_session_only: bool = False,
) -> dict:
    """Convert all .1D files to .npy and build subject JSON.

    Args:
        raw_dir: Path to extracted tar contents.
        out_dir: Output directory.
        first_session_only: If True, use only session_1_rest_1 (no concatenation).

    Returns:
        Summary dict.
    """
    cache_dir = out_dir / "timeseries_cache" / "cc200"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load phenotypic
    pheno = load_all_phenotypic(raw_dir)

    # Discover all subject directories
    subjects_success = []
    subjects_failed = []
    n_multi_session = 0

    for site in SITES:
        site_dir = raw_dir / site
        if not site_dir.exists():
            continue

        # List subject directories (numeric IDs)
        sub_dirs = sorted([
            d.name for d in site_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])

        for sub_id in sub_dirs:
            files_1d = discover_1d_files(raw_dir, site, sub_id)
            if not files_1d:
                subjects_failed.append(sub_id)
                continue

            if first_session_only:
                files_1d = files_1d[:1]

            # Parse and optionally concatenate
            arrays = []
            for f in files_1d:
                arr = parse_1d_file(f)
                if arr is not None:
                    arrays.append(arr)

            if not arrays:
                subjects_failed.append(sub_id)
                continue

            if len(arrays) > 1:
                # Verify consistent ROI count
                n_rois = arrays[0].shape[1]
                arrays = [a for a in arrays if a.shape[1] == n_rois]
                if not arrays:
                    subjects_failed.append(sub_id)
                    continue
                ts = np.concatenate(arrays, axis=0)
                n_multi_session += 1
            else:
                ts = arrays[0]

            # Remove zero-variance ROIs
            valid = np.std(ts, axis=0) > 1e-8
            n_invalid = np.sum(~valid)
            if n_invalid > ts.shape[1] * 0.2:
                logger.debug(f"Too many invalid ROIs for {sub_id}: {n_invalid}/{ts.shape[1]}")
                subjects_failed.append(sub_id)
                continue
            if n_invalid > 0:
                ts = ts[:, valid]

            # Save .npy
            npy_path = cache_dir / f"{sub_id}_cc200.npy"
            np.save(npy_path, ts)

            # Build metadata
            meta = pheno.get(sub_id, {
                "sub_id": sub_id,
                "site": site,
                "group": "Unknown",
                "dx": -1,
                "age": -1.0,
                "sex": "Unknown",
            })
            meta["ts_path"] = str(npy_path)
            meta["n_trs"] = int(ts.shape[0])
            meta["n_rois"] = int(ts.shape[1])
            meta["n_sessions"] = len(files_1d)
            meta["tr"] = SITE_TR.get(site, 2.0)
            meta["total_s"] = meta["n_trs"] * meta["tr"]

            subjects_success.append(meta)

    # Sort by site + sub_id
    subjects_success.sort(key=lambda s: (s.get("site", ""), s.get("sub_id", "")))

    # Save subject JSON
    json_path = results_dir / "adhd200_subjects_cc200.json"
    with open(json_path, "w") as f:
        json.dump(subjects_success, f, indent=2, ensure_ascii=False)

    # Summary
    sites = sorted(set(s["site"] for s in subjects_success))
    groups = {}
    for s in subjects_success:
        g = s.get("group", "Unknown")
        groups[g] = groups.get(g, 0) + 1

    site_counts = {}
    for s in subjects_success:
        site = s["site"]
        site_counts[site] = site_counts.get(site, 0) + 1

    trs = [s["n_trs"] for s in subjects_success]
    durations = [s["total_s"] for s in subjects_success]

    summary = {
        "n_total_phenotypic": len(pheno),
        "n_converted": len(subjects_success),
        "n_failed": len(subjects_failed),
        "n_multi_session": n_multi_session,
        "first_session_only": first_session_only,
        "n_sites": len(sites),
        "sites": sites,
        "site_counts": site_counts,
        "groups": groups,
        "n_rois_typical": 190,
        "tr_per_site": SITE_TR,
        "trs_stats": {
            "min": int(np.min(trs)) if trs else 0,
            "max": int(np.max(trs)) if trs else 0,
            "mean": float(np.mean(trs)) if trs else 0,
            "median": float(np.median(trs)) if trs else 0,
        },
        "duration_s_stats": {
            "min": float(np.min(durations)) if durations else 0,
            "max": float(np.max(durations)) if durations else 0,
            "mean": float(np.mean(durations)) if durations else 0,
        },
        "json_path": str(json_path),
    }

    summary_path = results_dir / "adhd200_convert_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Conversion complete!")
    logger.info(f"  Subjects: {len(subjects_success)} / {len(subjects_success) + len(subjects_failed)}")
    logger.info(f"  Multi-session: {n_multi_session}")
    logger.info(f"  Sites: {len(sites)} — {', '.join(sites)}")
    logger.info(f"  Groups: {groups}")
    logger.info(f"  TRs: {summary['trs_stats']}")
    logger.info(f"  Duration: {summary['duration_s_stats']['min']:.0f}–{summary['duration_s_stats']['max']:.0f}s "
                f"(mean {summary['duration_s_stats']['mean']:.0f}s)")
    logger.info(f"  Output: {json_path}")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"{'=' * 60}")

    return summary


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert ADHD-200 PCP .1D timeseries to .npy + subject JSON",
    )
    parser.add_argument(
        "--raw-dir", type=str, default=str(RAW_DIR),
        help=f"Path to extracted tar contents (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUT_DIR),
        help=f"Output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--first-session-only", action="store_true",
        help="Use only session_1_rest_1 per subject (no multi-run concatenation)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    convert_all(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.output_dir),
        first_session_only=args.first_session_only,
    )


if __name__ == "__main__":
    main()
