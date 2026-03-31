#!/usr/bin/env python3
"""Preprocess ds000243 (WashU resting-state) for BS-NET duration sweep.

Pipeline:
  1. Load fMRIPrep preprocessed BOLD (MNI space, 2mm)
  2. Load confounds (36P: 6 motion params + derivatives + quadratics + WM/CSF)
  3. Bandpass filter: 0.01–0.1 Hz  (NO task regression — pure resting-state)
  4. Parcellation: configurable atlas → (n_timepoints × n_rois) timeseries
  5. Save as .npy per subject

Input:
  ds000243/ (fMRIPrep output, BIDS derivative)
    sub-XX/func/sub-XX_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
    sub-XX/func/sub-XX_task-rest_desc-confounds_timeseries.tsv
    sub-XX/func/sub-XX_task-rest_bold.json   (optional — TR fallback = 2.0s)

Output:
  data/ds000243/timeseries_cache/{atlas}/sub-XX_{atlas}.npy

Notes:
  - ds000243 is the WashU resting-state dataset (N=50, ≥15 min scan).
  - No task regressors needed (pure resting-state — cf. ds007535 task-residual).
  - Same 36P confound schema and atlas support as preprocess_ds007535.py.
  - fMRIPrep filenames may vary; discover_subjects() uses flexible glob.

Usage:
    # Process all subjects
    python src/scripts/preprocess_ds000243.py --input-dir data/ds000243/raw

    # Limit to first 5
    python src/scripts/preprocess_ds000243.py --input-dir data/ds000243/raw --max-subjects 5

    # Custom atlas
    python src/scripts/preprocess_ds000243.py --input-dir data/ds000243/raw --atlas schaefer400
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── SSL: patch requests to use macOS system cert bundle ──────────────────
# nilearn creates its requests.Session at import time, so env vars alone
# are not enough. We monkey-patch requests.adapters before any import
# to fix 'certificate verify failed' on Homebrew/pyenv macOS Python.
_MACOS_CERT = Path("/etc/ssl/cert.pem")
if _MACOS_CERT.exists():
    os.environ["SSL_CERT_FILE"] = str(_MACOS_CERT)
    os.environ["REQUESTS_CA_BUNDLE"] = str(_MACOS_CERT)
    try:
        import requests.adapters as _ra
        _orig_send = _ra.HTTPAdapter.send

        def _patched_send(self, request, **kwargs):  # type: ignore[override]
            if kwargs.get("verify") is not False:
                kwargs["verify"] = str(_MACOS_CERT)
            return _orig_send(self, request, **kwargs)

        _ra.HTTPAdapter.send = _patched_send  # type: ignore[method-assign]
    except Exception:
        pass  # non-fatal: fall back to env vars

# ── Constants ────────────────────────────────────────────────────────────

# Default TR fallback for ds000243 (WashU HCP-style; sidecar JSON is preferred)
TR_FALLBACK = 2.0

# 36P confound columns (Ciric et al., 2017; Satterthwaite et al., 2013)
CONFOUND_COLS_BASE = [
    "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
    "csf", "white_matter",
]

# Atlas options — identical to preprocess_ds007535.py
ATLAS_MAP = {
    # ── Schaefer 2018 (7 Yeo networks, 2mm) ────────────────────────────
    "schaefer100":  {"family": "schaefer", "n_rois": 100,
                     "kwargs": {"n_rois": 100,  "yeo_networks": 7, "resolution_mm": 2}},
    "schaefer200":  {"family": "schaefer", "n_rois": 200,
                     "kwargs": {"n_rois": 200,  "yeo_networks": 7, "resolution_mm": 2}},
    "schaefer300":  {"family": "schaefer", "n_rois": 300,
                     "kwargs": {"n_rois": 300,  "yeo_networks": 7, "resolution_mm": 2}},
    "schaefer400":  {"family": "schaefer", "n_rois": 400,
                     "kwargs": {"n_rois": 400,  "yeo_networks": 7, "resolution_mm": 2}},
    "schaefer500":  {"family": "schaefer", "n_rois": 500,
                     "kwargs": {"n_rois": 500,  "yeo_networks": 7, "resolution_mm": 2}},
    "schaefer600":  {"family": "schaefer", "n_rois": 600,
                     "kwargs": {"n_rois": 600,  "yeo_networks": 7, "resolution_mm": 2}},
    "schaefer800":  {"family": "schaefer", "n_rois": 800,
                     "kwargs": {"n_rois": 800,  "yeo_networks": 7, "resolution_mm": 2}},
    "schaefer1000": {"family": "schaefer", "n_rois": 1000,
                     "kwargs": {"n_rois": 1000, "yeo_networks": 7, "resolution_mm": 2}},
    # ── AAL (Automated Anatomical Labeling, 116 ROIs) ───────────────────
    "aal": {"family": "aal", "n_rois": 116,
            "kwargs": {"version": "SPM12"}},
    # ── Harvard-Oxford cortical (48 ROIs, thr=25%) ──────────────────────
    "harvard_oxford": {"family": "harvard_oxford", "n_rois": 48,
                       "kwargs": {"atlas_name": "cort-maxprob-thr25-2mm"}},
    # ── Craddock 2012 (scorr_mean, same as ABIDE PCP CC200/CC400) ───────
    "cc200": {"family": "craddock", "n_rois": 200, "kwargs": {}},
    "cc400": {"family": "craddock", "n_rois": 400, "kwargs": {}},
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
        (n_timepoints, n_confounds) array with NaN-filled first row zeroed.
    """
    import pandas as pd

    df = pd.read_csv(confounds_path, sep="\t")

    cols = []
    for base_col in CONFOUND_COLS_BASE:
        if base_col in df.columns:
            cols.append(base_col)
            deriv_col = f"{base_col}_derivative1"
            if deriv_col in df.columns:
                cols.append(deriv_col)
            power_col = f"{base_col}_power2"
            if power_col in df.columns:
                cols.append(power_col)
            deriv_power_col = f"{base_col}_derivative1_power2"
            if deriv_power_col in df.columns:
                cols.append(deriv_power_col)

    if not cols:
        logger.warning(f"No confound columns found in {confounds_path}")
        return np.zeros((len(df), 1))

    confounds = df[cols].values.astype(np.float64)
    confounds = np.nan_to_num(confounds, nan=0.0)

    logger.debug(f"Loaded {len(cols)} confound columns from {confounds_path.name}")
    return confounds


# ── Atlas fetcher ────────────────────────────────────────────────────────

_ATLAS_CURL_INFO: dict[str, dict] = {
    "aal": {
        "url": "https://www.gin.cnrs.fr/AAL_files/aal_for_SPM12.tar.gz",
        "subdir": "aal_SPM12",
    },
    "harvard_oxford": {
        "url": (
            "https://www.nitrc.org/frs/download.php/14756/"
            "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"
        ),
        "subdir": "harvard_oxford",
    },
}


def _curl_download_atlas(family: str) -> None:
    """Download atlas file via system curl when Python SSL verification fails.

    Args:
        family: Atlas family key in ``_ATLAS_CURL_INFO``.

    Raises:
        KeyError: If the family has no curl entry.
        subprocess.CalledProcessError: If curl or tar fails.
    """
    import subprocess
    import tempfile

    info = _ATLAS_CURL_INFO[family]
    url = info["url"]
    nilearn_data_dir = Path.home() / "nilearn_data"
    target_dir = nilearn_data_dir / info["subdir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"SSL fallback: downloading {family} atlas via curl ...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            ["curl", "-fsSL", "-o", str(tmp_path), url],
            check=True,
        )
        if tmp_path.stat().st_size > 10_000:
            subprocess.run(
                ["tar", "-xzf", str(tmp_path), "-C", str(target_dir)],
                check=True,
            )
        logger.info(f"  Extracted to {target_dir}")
    finally:
        tmp_path.unlink(missing_ok=True)


def _fetch_atlas_craddock(n_rois: int) -> object:
    """Extract a single-volume (3D) Craddock 2012 atlas for NiftiLabelsMasker.

    Args:
        n_rois: Target ROI count (200 or 400).

    Returns:
        SimpleNamespace with ``maps`` set to a 3D Nifti1Image.

    Raises:
        ValueError: If no volume with max_label == n_rois is found.
    """
    import types

    import nibabel as nib
    import numpy as np
    from nilearn import datasets as nlds

    raw = nlds.fetch_atlas_craddock_2012()
    nii_path = getattr(raw, "maps", None) or raw.scorr_mean
    img4d = nib.load(nii_path)
    data4d = img4d.get_fdata()

    for vol_idx in range(data4d.shape[-1]):
        vol = data4d[..., vol_idx]
        labels = np.unique(vol[vol > 0]).astype(int)
        if len(labels) > 0 and int(labels.max()) == n_rois:
            img3d = nib.Nifti1Image(vol, img4d.affine, img4d.header)
            result = types.SimpleNamespace()
            result.maps = img3d
            logger.debug(
                f"Craddock CC{n_rois}: volume {vol_idx}, {len(labels)} labels"
            )
            return result

    raise ValueError(
        f"Craddock atlas: no volume found with max_label={n_rois}. "
        "Available options: 200 (CC200) or 400 (CC400)."
    )


def _fetch_atlas(atlas: str) -> object:
    """Return nilearn atlas object for the given atlas name.

    Args:
        atlas: Key in ATLAS_MAP.

    Returns:
        nilearn dataset object with a ``maps`` attribute.

    Raises:
        ValueError: If atlas family is unknown.
        ImportError: If nilearn is not installed.
    """
    from nilearn import datasets as nlds

    info = ATLAS_MAP[atlas]
    family = info["family"]
    kwargs = info["kwargs"]

    def _do_fetch() -> object:
        if family == "schaefer":
            return nlds.fetch_atlas_schaefer_2018(**kwargs)
        if family == "aal":
            return nlds.fetch_atlas_aal(**kwargs)
        if family == "harvard_oxford":
            return nlds.fetch_atlas_harvard_oxford(**kwargs)
        if family == "craddock":
            return _fetch_atlas_craddock(info["n_rois"])
        raise ValueError(f"Unknown atlas family: {family!r} (atlas={atlas!r})")

    try:
        return _do_fetch()
    except Exception as e:
        is_ssl = "SSL" in str(e) or "certificate" in str(e).lower()
        if is_ssl and family in _ATLAS_CURL_INFO:
            logger.warning(
                f"SSL error fetching {atlas} atlas — retrying via curl fallback"
            )
            _curl_download_atlas(family)
            return _do_fetch()
        raise


# ── Main processing ──────────────────────────────────────────────────────


def process_single_subject(
    bold_path: Path,
    confounds_path: Path,
    json_path: Path | None,
    atlas: str,
    output_dir: Path,
) -> dict | None:
    """Process one subject: 36P confound regression + bandpass + parcellation.

    No task regression is applied (pure resting-state).

    Args:
        bold_path: Path to preprocessed BOLD NIfTI.
        confounds_path: Path to confounds TSV.
        json_path: Path to BOLD sidecar JSON (optional; TR fallback = 2.0s).
        atlas: Atlas name (key in ATLAS_MAP).
        output_dir: Output directory for .npy files.

    Returns:
        Dict with subject info, or None on failure.
    """
    try:
        import nibabel as nib
        from nilearn.maskers import NiftiLabelsMasker
    except ImportError as e:
        logger.error(f"Required package missing: {e}")
        return None

    sub_id = bold_path.name.split("_")[0]  # sub-XX

    # Load TR from sidecar JSON (fallback to TR_FALLBACK)
    tr = TR_FALLBACK
    if json_path and json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        tr = meta.get("RepetitionTime", TR_FALLBACK)

    # Load BOLD
    img = nib.load(bold_path)
    n_timepoints = img.shape[-1]
    logger.info(
        f"{sub_id}: {n_timepoints} volumes, TR={tr}s, "
        f"total={n_timepoints * tr:.0f}s"
    )

    # Load confounds (36P)
    confounds_36p = load_confounds_36p(confounds_path)

    # Trim confounds if fMRIPrep included dummy scans not present in BOLD
    if confounds_36p.shape[0] > n_timepoints:
        n_trim = confounds_36p.shape[0] - n_timepoints
        logger.debug(f"  Trimming {n_trim} dummy volumes from confounds head")
        confounds_36p = confounds_36p[n_trim:]

    logger.debug(f"  Confound matrix: {confounds_36p.shape}")

    # Fetch atlas
    atlas_data = _fetch_atlas(atlas)

    # Extract parcellated timeseries with 36P denoising + bandpass
    # No task regressors — resting-state only
    masker = NiftiLabelsMasker(
        labels_img=atlas_data.maps,
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=tr,
    )

    try:
        timeseries = masker.fit_transform(img, confounds=confounds_36p)
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
    """Discover ds000243 subjects with fMRIPrep resting-state outputs.

    Searches for any ``*_bold.nii.gz`` under sub-XX/func/ (flexible glob to
    handle varying fMRIPrep space/res/desc suffix combinations). Skips
    subjects whose BOLD file is an unretrieved DataLad annex stub (<1 MB).

    Args:
        input_dir: Root directory containing sub-XX/ subdirectories.

    Returns:
        List of dicts with bold_path, confounds_path, json_path.
    """
    subjects = []
    for sub_dir in sorted(input_dir.glob("sub-*")):
        func_dir = sub_dir / "func"
        if not func_dir.exists():
            continue

        # Prefer MNI152NLin2009cAsym preprocessed BOLD; fall back to any bold
        bold_candidates = sorted(func_dir.glob("*desc-preproc_bold.nii.gz"))
        if not bold_candidates:
            bold_candidates = sorted(func_dir.glob("*_bold.nii.gz"))
        if not bold_candidates:
            continue

        bold_path = bold_candidates[0]

        # Skip DataLad broken symlinks / unretrieved annex objects
        try:
            actual_size = bold_path.stat().st_size
        except OSError:
            logger.debug(f"  Skipping {bold_path.name}: broken symlink")
            continue
        if actual_size < 1_000_000:
            logger.debug(
                f"  Skipping {bold_path.name}: too small ({actual_size} bytes)"
                " — likely unretrieved DataLad annex object"
            )
            continue

        stem = bold_path.name.replace("_bold.nii.gz", "")
        confounds_path = func_dir / f"{stem}_desc-confounds_timeseries.tsv"
        json_path      = func_dir / f"{stem}_bold.json"

        if not confounds_path.exists():
            # Try without desc-preproc in stem (some fMRIPrep versions differ)
            alt_stem = stem.replace("_desc-preproc", "")
            alt_confounds = func_dir / f"{alt_stem}_desc-confounds_timeseries.tsv"
            if alt_confounds.exists():
                confounds_path = alt_confounds
                alt_json = func_dir / f"{alt_stem}_bold.json"
                if alt_json.exists():
                    json_path = alt_json
            else:
                logger.debug(
                    f"  Skipping {sub_dir.name}: confounds not found "
                    f"({confounds_path.name})"
                )
                continue

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
        description="Preprocess ds000243 (WashU resting-state) for BS-NET duration sweep"
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Path to ds000243 fMRIPrep derivative root (containing sub-XX/func/)",
    )
    parser.add_argument(
        "--atlas", default="schaefer200",
        choices=list(ATLAS_MAP.keys()),
    )
    parser.add_argument(
        "--output-dir", default=None, type=Path,
        help="Output directory (default: data/ds000243/timeseries_cache/{atlas})",
    )
    parser.add_argument(
        "--max-subjects", type=int, default=0,
        help="Limit subjects (0 = all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reprocess subjects even if .npy already exists",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (
        Path("data/ds000243/timeseries_cache") / args.atlas
    )

    subjects = discover_subjects(args.input_dir)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]

    results = []
    for i, sub in enumerate(subjects):
        sub_id = sub["bold_path"].parent.parent.name
        out_path = output_dir / f"{sub_id}_{args.atlas}.npy"
        if not args.force and out_path.exists():
            logger.info(f"[{i + 1}/{len(subjects)}] Skipping {sub_id} (already exists)")
            continue
        logger.info(f"[{i + 1}/{len(subjects)}] Processing {sub_id}")
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
