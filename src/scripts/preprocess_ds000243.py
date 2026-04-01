#!/usr/bin/env python3
"""Preprocess ds000243 (WashU resting-state) for BS-NET duration sweep.

Pipeline:
  1. Load fMRIPrep preprocessed BOLD (MNI space, 2mm)
  2. Load confounds (36P: 6 motion params + derivatives + quadratics + WM/CSF)
  3. Bandpass filter: 0.01–0.1 Hz  (NO task regression — pure resting-state)
  4. Parcellation: configurable atlas → (n_timepoints × n_rois) timeseries
  5. Concatenate across runs (if multiple runs exist) and save as .npy per subject

Input:
  ds000243/ (fMRIPrep output, BIDS derivative)
    sub-XX/func/sub-XX_task-rest_run-N_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz
    sub-XX/func/sub-XX_task-rest_run-N_desc-confounds_timeseries.tsv
    sub-XX/func/sub-XX_task-rest_run-N_bold.json   (optional — TR fallback = 2.0s)

Output:
  data/ds000243/timeseries_cache/{atlas}/sub-XX_{atlas}.npy

Notes:
  - ds000243 is the WashU resting-state dataset (N=52, ≥15 min scan).
  - sub-001~014 have a single long run; sub-015~052 have run-1 + run-2 (concatenated).
  - No task regressors needed (pure resting-state — cf. ds007535 task-residual).
  - Same 36P confound schema and atlas support as preprocess_ds007535.py.
  - fMRIPrep filenames may vary; discover_subjects() uses flexible glob and
    strips space/res entities to locate confounds TSV (which lacks these entities).

Usage:
    # Process all subjects (--input-dir = fMRIPrep derivative root)
    python src/scripts/preprocess_ds000243.py --input-dir data/ds000243/results/fmrirep

    # Limit to first 5
    python src/scripts/preprocess_ds000243.py --input-dir data/ds000243/results/fmrirep --max-subjects 5

    # Custom atlas
    python src/scripts/preprocess_ds000243.py --input-dir data/ds000243/results/fmrirep --atlas schaefer400
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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


def _process_single_run(
    bold_path: Path,
    confounds_path: Path,
    json_path: Path | None,
    masker: object,
    tr: float,
) -> np.ndarray | None:
    """Process one run: confound regression + bandpass + parcellation.

    Args:
        bold_path: Path to preprocessed BOLD NIfTI.
        confounds_path: Path to confounds TSV.
        json_path: Unused (TR already resolved by caller); kept for API symmetry.
        masker: Fitted or unfitted NiftiLabelsMasker instance.
        tr: Repetition time in seconds.

    Returns:
        (n_timepoints, n_rois) float32 array, or None on failure.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        logger.error(f"Required package missing: {e}")
        return None

    img = nib.load(bold_path)
    n_timepoints = img.shape[-1]
    logger.info(
        f"  Run {bold_path.name}: {n_timepoints} vols, TR={tr}s, "
        f"total={n_timepoints * tr:.0f}s"
    )

    confounds_36p = load_confounds_36p(confounds_path)
    if confounds_36p.shape[0] > n_timepoints:
        n_trim = confounds_36p.shape[0] - n_timepoints
        logger.debug(f"  Trimming {n_trim} dummy volumes from confounds head")
        confounds_36p = confounds_36p[n_trim:]

    try:
        ts = masker.fit_transform(img, confounds=confounds_36p)
    except Exception as e:
        logger.error(f"  Masker failed for {bold_path.name}: {e}")
        return None

    return ts.astype(np.float32)


def process_subject(
    sub_id: str,
    runs: list[dict],
    atlas: str,
    output_dir: Path,
) -> dict | None:
    """Process one subject: all runs concatenated → .npy timeseries.

    Each run is denoised (36P) and bandpass-filtered independently, then
    concatenated along the time axis before saving. This matches standard
    multi-run FC preprocessing practice.

    Args:
        sub_id: Subject identifier (e.g. ``"sub-001"``).
        runs: List of dicts, each with ``bold_path``, ``confounds_path``,
              ``json_path`` (may be None).
        atlas: Atlas name (key in ATLAS_MAP).
        output_dir: Output directory for .npy files.

    Returns:
        Dict with subject summary, or None if all runs failed.
    """
    try:
        from nilearn.maskers import NiftiLabelsMasker
    except ImportError as e:
        logger.error(f"Required package missing: {e}")
        return None

    # Resolve TR from first available sidecar JSON
    tr = TR_FALLBACK
    for run in runs:
        jp = run.get("json_path")
        if jp and jp.exists():
            with open(jp) as f:
                meta = json.load(f)
            tr = meta.get("RepetitionTime", TR_FALLBACK)
            break

    atlas_data = _fetch_atlas(atlas)
    masker = NiftiLabelsMasker(
        labels_img=atlas_data.maps,
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=tr,
    )

    all_ts: list[np.ndarray] = []
    for run in runs:
        ts = _process_single_run(
            bold_path=run["bold_path"],
            confounds_path=run["confounds_path"],
            json_path=run.get("json_path"),
            masker=masker,
            tr=tr,
        )
        if ts is not None:
            all_ts.append(ts)

    if not all_ts:
        logger.error(f"{sub_id}: all runs failed")
        return None

    timeseries = np.concatenate(all_ts, axis=0)
    n_vols_total = timeseries.shape[0]
    n_rois_actual = timeseries.shape[1]

    if len(all_ts) > 1:
        logger.info(
            f"{sub_id}: concatenated {len(all_ts)} runs → "
            f"({n_vols_total}, {n_rois_actual}), "
            f"total={n_vols_total * tr:.0f}s"
        )
    else:
        logger.info(f"{sub_id}: ({n_vols_total}, {n_rois_actual})")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sub_id}_{atlas}.npy"
    np.save(out_path, timeseries)

    return {
        "sub_id": sub_id,
        "n_runs": len(all_ts),
        "n_vols": n_vols_total,
        "tr": tr,
        "total_sec": n_vols_total * tr,
        "n_rois": n_rois_actual,
        "ts_path": str(out_path),
    }


def _bold_base_entity(bold_name: str) -> str:
    """Strip space/res/desc BIDS entities from a BOLD filename stem.

    fMRIPrep embeds space/res/desc in BOLD filenames but NOT in confounds TSV.
    This function extracts the minimal BIDS entity string (sub, task, run)
    needed to locate the corresponding confounds file.

    Example::

        sub-001_task-rest_run-1_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz
        → sub-001_task-rest_run-1

    Args:
        bold_name: BOLD filename (basename only).

    Returns:
        Base entity string without space/res/desc suffixes.
    """
    # Strip .nii.gz / _bold.nii.gz suffix
    stem = bold_name.replace("_bold.nii.gz", "").replace(".nii.gz", "")
    # Remove _space-XXX, _res-XXX, _desc-XXX entities (and everything after)
    base = re.sub(r"_(space|res|desc)-[^_]+", "", stem)
    return base


def discover_subjects(input_dir: Path) -> list[dict]:
    """Discover ds000243 subjects with fMRIPrep resting-state outputs.

    Finds all runs per subject under sub-XX/func/. Multi-run subjects
    (run-1 + run-2) are returned with both runs so they can be concatenated.
    Skips BOLD files that are unretrieved DataLad annex stubs (<1 MB).

    The confounds TSV uses a shorter filename than the BOLD (no space/res
    entities). ``_bold_base_entity()`` strips those entities to resolve the
    correct confounds path.

    Args:
        input_dir: fMRIPrep derivative root containing sub-XX/ subdirectories.

    Returns:
        List of subject dicts, each with ``sub_id`` and ``runs`` (list of
        dicts with ``bold_path``, ``confounds_path``, ``json_path``).
    """
    subjects = []
    for sub_dir in sorted(input_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue  # skip sub-XX.html report files
        func_dir = sub_dir / "func"
        if not func_dir.exists():
            continue

        # Collect all preproc BOLD candidates (sorted → run-1 before run-2)
        bold_candidates = sorted(func_dir.glob("*desc-preproc_bold.nii.gz"))
        if not bold_candidates:
            bold_candidates = sorted(func_dir.glob("*_bold.nii.gz"))
        if not bold_candidates:
            continue

        runs: list[dict] = []
        for bold_path in bold_candidates:
            # Skip DataLad broken symlinks / unretrieved annex objects
            try:
                actual_size = bold_path.stat().st_size
            except OSError:
                logger.debug(f"  Skipping {bold_path.name}: broken symlink")
                continue
            if actual_size < 1_000_000:
                logger.debug(
                    f"  Skipping {bold_path.name}: too small ({actual_size} B)"
                    " — likely unretrieved DataLad annex object"
                )
                continue

            # Resolve confounds: strip space/res/desc to get base BIDS entity
            base = _bold_base_entity(bold_path.name)
            confounds_path = func_dir / f"{base}_desc-confounds_timeseries.tsv"
            json_path = func_dir / f"{base}_bold.json"

            if not confounds_path.exists():
                logger.debug(
                    f"  Skipping {bold_path.name}: confounds not found "
                    f"({confounds_path.name})"
                )
                continue

            runs.append({
                "bold_path": bold_path,
                "confounds_path": confounds_path,
                "json_path": json_path if json_path.exists() else None,
            })

        if not runs:
            logger.debug(f"  Skipping {sub_dir.name}: no valid runs found")
            continue

        subjects.append({
            "sub_id": sub_dir.name,
            "runs": runs,
        })

    logger.info(
        f"Discovered {len(subjects)} subjects in {input_dir} "
        f"(runs/subject: "
        f"{sum(len(s['runs']) for s in subjects) / max(len(subjects), 1):.1f} avg)"
    )
    return subjects


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess ds000243 (WashU resting-state) for BS-NET duration sweep"
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help=(
            "Path to ds000243 fMRIPrep derivative root (containing sub-XX/func/). "
            "Example: data/ds000243/results/fmrirep"
        ),
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
    parser.add_argument(
        "--n-jobs", type=int, default=8,
        help="Parallel workers (joblib loky). -1 = all CPUs, 1 = sequential (default: 8)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (
        Path("data/ds000243/timeseries_cache") / args.atlas
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = discover_subjects(args.input_dir)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]

    # Filter already-processed subjects before dispatching workers
    pending = []
    for sub in subjects:
        out_path = output_dir / f"{sub['sub_id']}_{args.atlas}.npy"
        if not args.force and out_path.exists():
            logger.info(f"Skipping {sub['sub_id']} (already exists)")
        else:
            pending.append(sub)

    logger.info(
        f"{len(pending)} subjects to process, {len(subjects) - len(pending)} already done "
        f"(n_jobs={args.n_jobs})"
    )

    def _run(sub: dict) -> dict | None:
        return process_subject(
            sub_id=sub["sub_id"],
            runs=sub["runs"],
            atlas=args.atlas,
            output_dir=output_dir,
        )

    if args.n_jobs == 1:
        results = [r for sub in pending if (r := _run(sub)) is not None]
    else:
        from joblib import Parallel, delayed
        raw = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
            delayed(_run)(sub) for sub in pending
        )
        results = [r for r in raw if r is not None]

    print(f"\nProcessed {len(results)}/{len(subjects)} subjects "
          f"({len(subjects) - len(pending)} skipped)")
    print(f"Output: {output_dir}")
    if results:
        durations = [r["total_sec"] for r in results]
        n_runs_list = [r["n_runs"] for r in results]
        print(f"Duration: {min(durations):.0f}–{max(durations):.0f}s "
              f"(mean={np.mean(durations):.0f}s)")
        print(f"Runs/subject: {min(n_runs_list)}–{max(n_runs_list)}")
        print(f"ROIs: {results[0]['n_rois']}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
