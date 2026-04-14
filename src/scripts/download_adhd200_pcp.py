#!/usr/bin/env python3
"""Download ADHD-200 Preprocessed Connectomes Project (PCP) CC200/CC400 timeseries.

Downloads ROI timeseries (.1D) from the fcp-indi S3 bucket, converts to .npy,
and generates a subject metadata JSON compatible with BS-NET pipeline.

The ADHD-200 PCP provides ~973 subjects across 8 sites, preprocessed by
multiple pipelines (Athena/CPAC/NIAK). This script downloads the Athena
pipeline results with bandpass filtering + no global signal regression.

S3 URL pattern (same as ABIDE PCP):
  https://s3.amazonaws.com/fcp-indi/data/Projects/ADHD200/Outputs/
    {pipeline}/{strategy}/{derivative}/{file_id}_{derivative}.1D

References:
  - Bellec et al. (2017). The Neuro Bureau ADHD-200 Preprocessed Repository.
    NeuroImage, 144, 275-286. DOI: 10.1016/j.neuroimage.2016.06.034
  - http://preprocessed-connectomes-project.org/adhd200/

Usage:
    # Download CC200 timeseries (default: Athena pipeline, filt_noglobal)
    python src/scripts/download_adhd200_pcp.py

    # CC400
    python src/scripts/download_adhd200_pcp.py --derivative rois_cc400

    # Different pipeline/strategy
    python src/scripts/download_adhd200_pcp.py --pipeline cpac --strategy filt_global

    # Pilot: first 50 subjects only
    python src/scripts/download_adhd200_pcp.py --max-subjects 50

    # Resume interrupted download (skips existing files)
    python src/scripts/download_adhd200_pcp.py --resume
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

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

S3_BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ADHD200"
S3_PHENO_URL = f"{S3_BASE}/Phenotypic_V1_0b_preprocessed1.csv"

# Alternative phenotypic URLs to try
PHENO_URLS = [
    f"{S3_BASE}/Phenotypic_V1_0b_preprocessed1.csv",
    f"{S3_BASE}/phenotypic/adhd200_preprocessed_phenotypics.tsv",
    "https://s3.amazonaws.com/fcp-indi/data/Projects/ADHD200/adhd200_preprocessed_phenotypics.tsv",
]

# ADHD-200 sites and their TR values (seconds)
# Source: ADHD-200 documentation
SITE_TR = {
    "KKI": 2.5,
    "NeuroIMAGE": 1.6,
    "NYU": 2.0,
    "OHSU": 2.5,
    "Peking": 2.0,       # Peking_1/2/3 all use TR=2.0
    "Peking_1": 2.0,
    "Peking_2": 2.0,
    "Peking_3": 2.0,
    "Pittsburgh": 1.5,
    "WashU": 2.5,
    # Alternative names
    "Peking_University": 2.0,
    "Peking University": 2.0,
    "Kennedy_Krieger": 2.5,
    "Oregon_Health": 2.5,
    "OHTSU": 2.5,
    "Brown": 2.0,
}

# Available pipelines and strategies
PIPELINES = ["athena", "cpac", "niak"]
STRATEGIES = ["filt_global", "filt_noglobal", "nofilt_global", "nofilt_noglobal"]
DERIVATIVES = ["rois_cc200", "rois_cc400"]

OUTPUT_DIR = Path("data/adhd/pcp")


# ═══════════════════════════════════════════════════════════════════════
# Phenotypic data
# ═══════════════════════════════════════════════════════════════════════

def download_phenotypic(output_dir: Path) -> Path:
    """Download ADHD-200 phenotypic file from S3.

    Tries multiple known URLs. Falls back to a local file if present.

    Returns:
        Path to the downloaded phenotypic file.
    """
    pheno_dir = output_dir / "phenotypic"
    pheno_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing local file
    for ext in ["csv", "tsv"]:
        local = pheno_dir / f"adhd200_phenotypic.{ext}"
        if local.exists() and local.stat().st_size > 1000:
            logger.info(f"Phenotypic file already exists: {local}")
            return local

    # Try downloading from known URLs
    for url in PHENO_URLS:
        try:
            logger.info(f"Trying phenotypic URL: {url}")
            resp = urlopen(url, timeout=30)
            data = resp.read()
            ext = "tsv" if "tsv" in url else "csv"
            out_path = pheno_dir / f"adhd200_phenotypic.{ext}"
            out_path.write_bytes(data)
            logger.info(f"Downloaded phenotypic file: {out_path} ({len(data)} bytes)")
            return out_path
        except (HTTPError, URLError) as e:
            logger.warning(f"  Failed: {e}")
            continue

    raise RuntimeError(
        "Could not download ADHD-200 phenotypic file. "
        "Please manually download from http://preprocessed-connectomes-project.org/adhd200/ "
        f"and place in {pheno_dir}/"
    )


def parse_phenotypic(pheno_path: Path) -> list[dict]:
    """Parse ADHD-200 phenotypic file into subject metadata.

    Handles both CSV and TSV formats from different PCP sources.

    Returns:
        List of dicts with keys: sub_id, site, group, age, sex.
    """
    text = pheno_path.read_text(encoding="utf-8", errors="replace")

    # Detect delimiter
    first_line = text.split("\n")[0]
    delimiter = "\t" if "\t" in first_line else ","

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)

    # Normalize column names (PCP uses varying names)
    subjects = []
    for row in reader:
        # Subject ID — try multiple column names
        sub_id = (
            row.get("ScanDir ID")
            or row.get("Subject")
            or row.get("SUB_ID")
            or row.get("ScanDirID")
            or row.get("SubID")
            or ""
        ).strip().strip('"')

        if not sub_id or sub_id in ("", "NA"):
            continue

        # Site
        site = (
            row.get("Site")
            or row.get("site")
            or row.get("SITE_ID")
            or ""
        ).strip().strip('"')

        # Diagnosis: DX (1=TDC, 2=ADHD-C, 3=ADHD-I, 0=pending)
        # Or: adhd=1/0, tdc=1/0
        dx = row.get("DX", row.get("dx", "")).strip().strip('"')
        adhd_col = row.get("adhd", row.get("ADHD", "")).strip().strip('"') if "adhd" in row or "ADHD" in row else ""
        tdc_col = row.get("tdc", row.get("TDC", "")).strip().strip('"') if "tdc" in row or "TDC" in row else ""

        if dx:
            try:
                dx_int = int(float(dx))
                group = "Control" if dx_int == 0 else "ADHD"
            except ValueError:
                group = "Unknown"
        elif adhd_col:
            group = "ADHD" if adhd_col == "1" else "Control"
        elif tdc_col:
            group = "Control" if tdc_col == "1" else "ADHD"
        else:
            group = "Unknown"

        # Age
        age_str = (row.get("Age", row.get("age", ""))).strip().strip('"')
        try:
            age = float(age_str)
        except (ValueError, TypeError):
            age = -1.0

        # Sex
        sex = (row.get("Sex", row.get("sex", row.get("Gender", "")))).strip().strip('"')

        subjects.append({
            "sub_id": sub_id,
            "site": site,
            "group": group,
            "age": age,
            "sex": sex,
        })

    logger.info(
        f"Parsed {len(subjects)} subjects from phenotypic file, "
        f"sites: {sorted(set(s['site'] for s in subjects))}"
    )
    return subjects


# ═══════════════════════════════════════════════════════════════════════
# S3 URL construction & download
# ═══════════════════════════════════════════════════════════════════════

def build_s3_url(
    sub_id: str,
    site: str,
    pipeline: str,
    strategy: str,
    derivative: str,
) -> list[str]:
    """Build candidate S3 URLs for a subject's timeseries.

    ADHD-200 PCP file naming varies by pipeline:
      - Athena: {ScanDirID}_{derivative}.1D
      - CPAC:   {site}_{ScanDirID}_{derivative}.1D

    Returns multiple candidate URLs to try.
    """
    base = f"{S3_BASE}/Outputs/{pipeline}/{strategy}/{derivative}"
    candidates = [
        # Pattern 1: plain subject ID
        f"{base}/{sub_id}_{derivative}.1D",
        # Pattern 2: site_subjectID (CPAC style)
        f"{base}/{site}_{sub_id}_{derivative}.1D",
        # Pattern 3: zero-padded 7-digit
        f"{base}/{int(sub_id):07d}_{derivative}.1D",
    ]
    return candidates


def download_timeseries_1d(
    sub: dict,
    pipeline: str,
    strategy: str,
    derivative: str,
    output_dir: Path,
    resume: bool = True,
) -> Path | None:
    """Download a single subject's .1D timeseries file.

    Args:
        sub: Subject metadata dict.
        pipeline: Pipeline name (athena, cpac, niak).
        strategy: Strategy name (filt_noglobal, etc.).
        derivative: Derivative name (rois_cc200, rois_cc400).
        output_dir: Directory to save .1D files.
        resume: Skip if file already exists.

    Returns:
        Path to downloaded .1D file, or None on failure.
    """
    raw_dir = output_dir / "raw_1d" / derivative
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_path = raw_dir / f"{sub['sub_id']}_{derivative}.1D"

    if resume and out_path.exists() and out_path.stat().st_size > 100:
        return out_path

    urls = build_s3_url(sub["sub_id"], sub["site"], pipeline, strategy, derivative)

    for url in urls:
        try:
            resp = urlopen(url, timeout=30)
            data = resp.read()
            if len(data) < 100:
                continue
            out_path.write_bytes(data)
            return out_path
        except (HTTPError, URLError):
            continue

    return None


def convert_1d_to_npy(path_1d: Path, output_dir: Path, sub_id: str, derivative: str) -> Path | None:
    """Convert .1D timeseries file to .npy array.

    .1D format: space-separated text, one row per timepoint, columns = ROIs.
    First row may be a comment (#) — skip it.

    Returns:
        Path to .npy file, or None on failure.
    """
    npy_dir = output_dir / "timeseries_cache" / derivative.replace("rois_", "")
    npy_dir.mkdir(parents=True, exist_ok=True)

    out_path = npy_dir / f"{sub_id}_{derivative.replace('rois_', '')}.npy"

    try:
        text = path_1d.read_text().strip()
        lines = [l for l in text.split("\n") if l.strip() and not l.strip().startswith("#")]
        ts = np.array([[float(v) for v in line.split()] for line in lines], dtype=np.float64)

        if ts.ndim != 2 or ts.shape[0] < 20 or ts.shape[1] < 10:
            logger.warning(f"Invalid timeseries shape for {sub_id}: {ts.shape}")
            return None

        # Remove zero-variance ROIs
        valid = np.std(ts, axis=0) > 1e-8
        n_invalid = np.sum(~valid)
        if n_invalid > ts.shape[1] * 0.2:
            logger.warning(f"Too many invalid ROIs for {sub_id}: {n_invalid}/{ts.shape[1]}")
            return None
        if n_invalid > 0:
            ts = ts[:, valid]

        np.save(out_path, ts)
        return out_path

    except Exception as e:
        logger.warning(f"Failed to convert {path_1d}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def download_all(
    pipeline: str = "athena",
    strategy: str = "filt_noglobal",
    derivative: str = "rois_cc200",
    output_dir: Path = OUTPUT_DIR,
    max_subjects: int = 0,
    resume: bool = True,
) -> dict:
    """Download all ADHD-200 PCP timeseries and build subject JSON.

    Args:
        pipeline: Preprocessing pipeline.
        strategy: Preprocessing strategy.
        derivative: ROI derivative (rois_cc200 or rois_cc400).
        output_dir: Base output directory.
        max_subjects: Limit downloads (0 = all).
        resume: Skip existing files.

    Returns:
        Summary dict with counts and paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get phenotypic data
    logger.info("=" * 60)
    logger.info("Step 1: Downloading phenotypic data")
    logger.info("=" * 60)
    pheno_path = download_phenotypic(output_dir)
    subjects = parse_phenotypic(pheno_path)

    if max_subjects > 0:
        subjects = subjects[:max_subjects]
        logger.info(f"Limited to first {max_subjects} subjects")

    # Step 2: Download timeseries
    logger.info("=" * 60)
    logger.info(f"Step 2: Downloading {derivative} timeseries ({pipeline}/{strategy})")
    logger.info(f"  Subjects to process: {len(subjects)}")
    logger.info("=" * 60)

    try:
        from tqdm import tqdm
        pbar = tqdm(subjects, desc="Downloading", unit="sub")
    except ImportError:
        pbar = subjects
        logger.info("(Install tqdm for progress bar: pip install tqdm)")

    success = []
    failed = []
    skipped = 0
    t0 = time.time()

    atlas_key = derivative.replace("rois_", "")  # cc200, cc400

    for sub in pbar:
        # Download .1D
        path_1d = download_timeseries_1d(
            sub, pipeline, strategy, derivative, output_dir, resume=resume
        )

        if path_1d is None:
            failed.append(sub["sub_id"])
            continue

        # Convert to .npy
        npy_dir = output_dir / "timeseries_cache" / atlas_key
        npy_path = npy_dir / f"{sub['sub_id']}_{atlas_key}.npy"

        if resume and npy_path.exists():
            ts = np.load(npy_path)
            skipped += 1
        else:
            npy_result = convert_1d_to_npy(path_1d, output_dir, sub["sub_id"], derivative)
            if npy_result is None:
                failed.append(sub["sub_id"])
                continue
            ts = np.load(npy_result)
            npy_path = npy_result

        # Get TR for this site
        tr = SITE_TR.get(sub["site"], 2.0)
        for prefix, val in SITE_TR.items():
            if prefix in sub["site"]:
                tr = val
                break

        sub["tr"] = tr
        sub["ts_path"] = str(npy_path)
        sub["n_trs"] = int(ts.shape[0])
        sub["n_rois"] = int(ts.shape[1])
        success.append(sub)

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(ok=len(success), fail=len(failed), skip=skipped)

    elapsed = time.time() - t0

    # Step 3: Save subject JSON
    logger.info("=" * 60)
    logger.info("Step 3: Saving subject metadata")
    logger.info("=" * 60)

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / f"adhd200_subjects_{atlas_key}.json"
    with open(json_path, "w") as f:
        json.dump(success, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved subject JSON: {json_path}")

    # Summary
    sites = sorted(set(s["site"] for s in success))
    groups = {g: sum(1 for s in success if s["group"] == g) for g in set(s["group"] for s in success)}

    summary = {
        "pipeline": pipeline,
        "strategy": strategy,
        "derivative": derivative,
        "n_total_phenotypic": len(subjects),
        "n_downloaded": len(success),
        "n_failed": len(failed),
        "n_sites": len(sites),
        "sites": sites,
        "groups": groups,
        "elapsed_sec": round(elapsed, 1),
        "json_path": str(json_path),
    }

    summary_path = results_dir / f"adhd200_download_summary_{atlas_key}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Download complete!")
    logger.info(f"  Success: {len(success)} / {len(subjects)}")
    logger.info(f"  Failed:  {len(failed)}")
    logger.info(f"  Sites:   {len(sites)} — {', '.join(sites)}")
    logger.info(f"  Groups:  {groups}")
    logger.info(f"  Time:    {elapsed:.0f}s")
    logger.info(f"  Output:  {json_path}")
    logger.info(f"{'=' * 60}")

    if failed:
        fail_path = results_dir / f"adhd200_failed_{atlas_key}.txt"
        fail_path.write_text("\n".join(failed))
        logger.info(f"Failed subjects saved to: {fail_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════
# Probe: test S3 path patterns before bulk download
# ═══════════════════════════════════════════════════════════════════════

def s3_list_prefix(prefix: str, delimiter: str = "/", max_keys: int = 100) -> tuple[list[str], list[str]]:
    """List S3 objects under a prefix using the public REST API.

    Args:
        prefix: S3 key prefix (e.g., "data/Projects/ADHD200/").
        delimiter: Delimiter for grouping (use "/" for directory listing).
        max_keys: Max number of keys to return.

    Returns:
        (common_prefixes, object_keys) — subdirectories and files.
    """
    import xml.etree.ElementTree as ET

    bucket = "fcp-indi"
    url = (
        f"https://{bucket}.s3.amazonaws.com/"
        f"?list-type=2&prefix={prefix}&delimiter={delimiter}&max-keys={max_keys}"
    )

    try:
        resp = urlopen(url, timeout=30)
        xml_data = resp.read().decode("utf-8")
    except (HTTPError, URLError) as e:
        logger.error(f"S3 list failed for prefix={prefix}: {e}")
        return [], []

    root = ET.fromstring(xml_data)
    ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""

    prefixes = [
        cp.find(f"{ns}Prefix").text
        for cp in root.findall(f"{ns}CommonPrefixes")
        if cp.find(f"{ns}Prefix") is not None
    ]
    keys = [
        c.find(f"{ns}Key").text
        for c in root.findall(f"{ns}Contents")
        if c.find(f"{ns}Key") is not None
    ]

    return prefixes, keys


def probe_s3_patterns(
    pipeline: str = "athena",
    strategy: str = "filt_noglobal",
    derivative: str = "rois_cc200",
) -> None:
    """Probe S3 bucket to discover actual directory structure for ADHD200.

    Uses the S3 List Objects REST API to explore the bucket, then tests
    file download with known subject IDs.
    """
    logger.info("=" * 60)
    logger.info("Probing S3 bucket structure for ADHD200")
    logger.info("=" * 60)

    # Level 1: List top-level ADHD200 directories
    base_prefix = "data/Projects/ADHD200/"
    logger.info(f"\n[1] Listing: s3://fcp-indi/{base_prefix}")
    dirs, files = s3_list_prefix(base_prefix)
    for d in dirs:
        logger.info(f"  DIR:  {d}")
    for f in files[:10]:
        logger.info(f"  FILE: {f}")

    if not dirs and not files:
        logger.warning("  (empty — project path may be different)")
        # Try alternative project names
        for alt in ["ADHD200/", "ADHD200_Preprocessed/", "adhd200/"]:
            alt_prefix = f"data/Projects/{alt}"
            logger.info(f"\n  Trying: s3://fcp-indi/{alt_prefix}")
            dirs, files = s3_list_prefix(alt_prefix)
            if dirs or files:
                base_prefix = alt_prefix
                for d in dirs:
                    logger.info(f"    DIR:  {d}")
                for f in files[:10]:
                    logger.info(f"    FILE: {f}")
                break

    # Level 2: Explore Outputs/ if it exists
    outputs_prefix = base_prefix + "Outputs/"
    logger.info(f"\n[2] Listing: s3://fcp-indi/{outputs_prefix}")
    dirs2, files2 = s3_list_prefix(outputs_prefix)
    for d in dirs2:
        logger.info(f"  DIR:  {d}")
    for f in files2[:5]:
        logger.info(f"  FILE: {f}")

    if not dirs2 and not files2:
        logger.info("  (no Outputs/ directory found)")
        # Try listing all subdirs more deeply
        for d in dirs:
            logger.info(f"\n  Exploring: s3://fcp-indi/{d}")
            sub_dirs, sub_files = s3_list_prefix(d)
            for sd in sub_dirs[:10]:
                logger.info(f"    DIR:  {sd}")
            for sf in sub_files[:5]:
                logger.info(f"    FILE: {sf}")

    # Level 3: Explore pipeline directory
    if dirs2:
        pipeline_prefix = None
        for d in dirs2:
            if pipeline in d.lower():
                pipeline_prefix = d
                break
        if not pipeline_prefix:
            pipeline_prefix = dirs2[0]

        logger.info(f"\n[3] Listing pipeline: s3://fcp-indi/{pipeline_prefix}")
        dirs3, files3 = s3_list_prefix(pipeline_prefix)
        for d in dirs3:
            logger.info(f"  DIR:  {d}")
        for f in files3[:5]:
            logger.info(f"  FILE: {f}")

        # Level 4: Explore strategy
        if dirs3:
            strat_prefix = None
            for d in dirs3:
                if strategy in d:
                    strat_prefix = d
                    break
            if not strat_prefix:
                strat_prefix = dirs3[0]

            logger.info(f"\n[4] Listing strategy: s3://fcp-indi/{strat_prefix}")
            dirs4, files4 = s3_list_prefix(strat_prefix)
            for d in dirs4:
                logger.info(f"  DIR:  {d}")
            for f in files4[:5]:
                logger.info(f"  FILE: {f}")

            # Level 5: Explore derivative
            if dirs4:
                deriv_prefix = None
                for d in dirs4:
                    if derivative in d:
                        deriv_prefix = d
                        break
                if not deriv_prefix:
                    deriv_prefix = dirs4[0]

                logger.info(f"\n[5] Listing derivative: s3://fcp-indi/{deriv_prefix}")
                dirs5, files5 = s3_list_prefix(deriv_prefix, max_keys=20)
                for d in dirs5:
                    logger.info(f"  DIR:  {d}")
                for f in files5[:20]:
                    logger.info(f"  FILE: {f}")
                logger.info(f"  ... (showing first 20 of potentially many files)")

    # Also try direct file probe with known subjects
    logger.info(f"\n[6] Testing direct file access with known subject IDs")
    test_ids = ["2014113", "3902469", "0010042"]
    test_sites = ["KKI", "KKI", "NYU"]

    # Build candidate paths from what we discovered
    candidate_bases = [
        f"{base_prefix}Outputs/{pipeline}/{strategy}/{derivative}",
        f"{base_prefix}{pipeline}/{strategy}/{derivative}",
    ]
    # Add any discovered paths
    if dirs2:
        for d in dirs2:
            candidate_bases.append(f"{d}{strategy}/{derivative}")

    for sid, site in zip(test_ids, test_sites):
        logger.info(f"\n  Subject: {sid} (site: {site})")
        for cb in candidate_bases:
            for pattern in [
                f"{sid}_{derivative}.1D",
                f"{site}_{sid}_{derivative}.1D",
                f"{sid}.1D",
            ]:
                url = f"https://s3.amazonaws.com/fcp-indi/{cb}/{pattern}"
                try:
                    resp = urlopen(url, timeout=10)
                    size = len(resp.read())
                    logger.info(f"    ✓ FOUND: {url}  ({size} bytes)")
                except HTTPError as e:
                    if e.code != 404:
                        logger.info(f"    ? {pattern}  (HTTP {e.code})")
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download ADHD-200 PCP timeseries from S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pipeline", default="athena", choices=PIPELINES,
        help="Preprocessing pipeline (default: athena)",
    )
    parser.add_argument(
        "--strategy", default="filt_noglobal", choices=STRATEGIES,
        help="Preprocessing strategy (default: filt_noglobal)",
    )
    parser.add_argument(
        "--derivative", default="rois_cc200", choices=DERIVATIVES,
        help="ROI derivative (default: rois_cc200)",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR), type=str,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-subjects", default=0, type=int,
        help="Limit number of subjects (default: 0 = all)",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Skip existing files (default: True)",
    )
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume",
        help="Re-download all files",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Probe S3 URL patterns without bulk download",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.probe:
        probe_s3_patterns(args.pipeline, args.strategy, args.derivative)
        return

    download_all(
        pipeline=args.pipeline,
        strategy=args.strategy,
        derivative=args.derivative,
        output_dir=Path(args.output_dir),
        max_subjects=args.max_subjects,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
