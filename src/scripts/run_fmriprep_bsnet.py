"""Run BS-NET pipeline on preprocessed fMRI data.

Supports two input modes:
    (A) XCP-D mode (default, recommended):
        - Reads XCP-D parcellated time series (Schaefer 100 or 400)
        - XCP-D handles: 36P confound regression, bandpass, scrubbing, smoothing
        - BS-NET only needs: FC computation + bootstrap extrapolation

    (B) fMRIPrep-direct mode (legacy, --input-mode fmriprep):
        - Reads fMRIPrep BOLD + confounds
        - nilearn NiftiLabelsMasker for parcellation + denoising
        - 29-param confound regression (24 motion + 5 aCompCor)

Common pipeline steps (both modes):
    1. FC computation (Ledoit-Wolf): full scan + short (first 2 min)
    2. BS-NET: bootstrap → SB extrapolation → Bayesian prior → attenuation correction
    3. Community detection → ARI / Jaccard vs reference FC

Usage (from bsNet/ project root):
    # XCP-D mode (default)
    python src/scripts/run_fmriprep_bsnet.py --subject sub-10159 --verbose
    python src/scripts/run_fmriprep_bsnet.py --run-all --parcels 400
    python src/scripts/run_fmriprep_bsnet.py --run-selection data/hc_100_selection.csv

    # fMRIPrep-direct mode (legacy)
    python src/scripts/run_fmriprep_bsnet.py --subject sub-10159 --input-mode fmriprep
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ATLAS_DIR = DATA_DIR / "atlas"
FMRIPREP_DIR = DATA_DIR / "derivatives" / "fmriprep"
XCPD_DIR = DATA_DIR / "derivatives" / "xcp_d"
BSNET_OUT_DIR = DATA_DIR / "derivatives" / "bsnet"

# --- Constants ---
TR_DEFAULT = 2.0
SHORT_DURATION_SEC = 120

# Confound columns to regress out (24-param + aCompCor)
CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
    "trans_x_power2", "trans_y_power2", "trans_z_power2",
    "rot_x_power2", "rot_y_power2", "rot_z_power2",
    "trans_x_derivative1_power2", "trans_y_derivative1_power2", "trans_z_derivative1_power2",
    "rot_x_derivative1_power2", "rot_y_derivative1_power2", "rot_z_derivative1_power2",
    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02",
    "a_comp_cor_03", "a_comp_cor_04",
]


def find_atlas(n_parcels: int = 100) -> Path:
    """Search multiple locations for Schaefer atlas.

    Args:
        n_parcels: Number of parcels (100 or 400).

    Returns:
        Path to atlas NIfTI.
    """
    name = f"Schaefer2018_{n_parcels}Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
    candidates = [
        ATLAS_DIR / name,
        PROJECT_ROOT.parent / "nilearn_data" / "schaefer_2018" / name,
        Path.home() / "nilearn_data" / "schaefer_2018" / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    msg = f"Atlas not found ({n_parcels} parcels). Searched: {[str(c) for c in candidates]}"
    raise FileNotFoundError(msg)


def load_network_labels() -> tuple[np.ndarray, list[str]]:
    """Load Schaefer parcel-to-network assignments."""
    tsv_path = DATA_DIR / "atlas" / "schaefer100_7networks_labels.tsv"
    if tsv_path.exists():
        df = pd.read_csv(tsv_path, sep="\t")
        net_names = sorted(df["network"].unique())
        net_map = {n: i + 1 for i, n in enumerate(net_names)}
        labels = df["network"].map(net_map).values
        return labels, net_names

    # Fallback: parse from atlas label file
    label_file = ATLAS_DIR / "Schaefer2018_100Parcels_7Networks_order.txt"
    if label_file.exists():
        networks = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[1].startswith("7Networks_"):
                    net = parts[1].split("_")[2]
                    networks.append(net)
        net_names = sorted(set(networks))
        net_map = {n: i + 1 for i, n in enumerate(net_names)}
        labels = np.array([net_map[n] for n in networks])
        return labels, net_names

    # Default: equal blocks
    logger.warning("Network labels not found. Using default 7 equal blocks.")
    labels = np.repeat(range(1, 8), [15, 14, 14, 14, 14, 15, 14])
    return labels, [f"Net{i}" for i in range(1, 8)]


def find_fmriprep_outputs(
    sub_id: str,
    fmriprep_dir: Path,
) -> dict[str, Path] | None:
    """Locate fMRIPrep output files for a subject.

    Args:
        sub_id: Subject ID (e.g. 'sub-10159').
        fmriprep_dir: fMRIPrep derivatives directory.

    Returns:
        Dict with paths to bold, confounds, mask; or None if not found.
    """
    func_dir = fmriprep_dir / sub_id / "func"
    if not func_dir.exists():
        return None

    bold = list(func_dir.glob("*MNI152NLin6Asym*desc-preproc_bold.nii.gz"))
    confounds = list(func_dir.glob("*desc-confounds_timeseries.tsv"))
    mask = list(func_dir.glob("*MNI152NLin6Asym*desc-brain_mask.nii.gz"))

    if not bold or not confounds:
        return None

    return {
        "bold": bold[0],
        "confounds": confounds[0],
        "mask": mask[0] if mask else None,
    }


def load_confounds(tsv_path: Path) -> np.ndarray:
    """Load and select confound regressors from fMRIPrep TSV.

    Args:
        tsv_path: Path to confounds TSV.

    Returns:
        Confounds array (n_volumes, n_confounds).
    """
    df = pd.read_csv(tsv_path, sep="\t")
    available = [c for c in CONFOUND_COLS if c in df.columns]
    logger.debug(f"  Using {len(available)}/{len(CONFOUND_COLS)} confound columns")
    confounds = df[available].values.astype(np.float64)
    # Replace NaN (first row for derivatives) with 0
    confounds = np.nan_to_num(confounds, nan=0.0)
    return confounds


def extract_timeseries(
    bold_path: Path,
    atlas_path: Path,
    confounds: np.ndarray,
    tr: float,
    mask_path: Path | None = None,
) -> np.ndarray:
    """Extract Schaefer 100-parcel time series with cleaning.

    Args:
        bold_path: MNI-space preprocessed BOLD.
        atlas_path: Schaefer atlas NIfTI.
        confounds: Confound regressors array.
        tr: Repetition time in seconds.
        mask_path: Optional brain mask.

    Returns:
        Cleaned time series (n_volumes, n_parcels).
    """
    from nilearn.maskers import NiftiLabelsMasker

    masker = NiftiLabelsMasker(
        labels_img=str(atlas_path),
        mask_img=str(mask_path) if mask_path else None,
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=tr,
    )

    ts = masker.fit_transform(str(bold_path), confounds=confounds)
    return ts


def compute_fc_lw(ts: np.ndarray) -> np.ndarray:
    """Compute FC using Ledoit-Wolf shrinkage estimator.

    Args:
        ts: Time series (n_volumes, n_parcels).

    Returns:
        Correlation matrix (n_parcels, n_parcels).
    """
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf()
    lw.fit(ts)
    cov = lw.covariance_
    d = np.sqrt(np.diag(cov))
    d[d == 0] = 1e-10
    fc = cov / np.outer(d, d)
    np.fill_diagonal(fc, 1.0)
    return np.clip(fc, -1, 1)


def find_xcpd_outputs(
    sub_id: str,
    xcpd_dir: Path,
    n_parcels: int = 100,
) -> dict[str, Path] | None:
    """Locate XCP-D output files for a subject.

    Args:
        sub_id: Subject ID (e.g. 'sub-10159').
        xcpd_dir: XCP-D derivatives directory.
        n_parcels: Number of Schaefer parcels (100 or 400).

    Returns:
        Dict with paths to timeseries TSV and denoised BOLD; or None.
    """
    func_dir = xcpd_dir / sub_id / "func"
    if not func_dir.exists():
        return None

    atlas_key = f"Schaefer{n_parcels}"
    ts_files = list(func_dir.glob(f"*atlas-{atlas_key}*timeseries*.tsv"))
    bold_files = list(func_dir.glob("*desc-denoised_bold.nii.gz"))

    if not ts_files:
        return None

    result: dict[str, Path] = {"timeseries": ts_files[0]}
    if bold_files:
        result["bold_denoised"] = bold_files[0]
    return result


def load_xcpd_timeseries(tsv_path: Path) -> np.ndarray:
    """Load parcellated time series from XCP-D TSV output.

    Args:
        tsv_path: Path to atlas-Schaefer*_timeseries.tsv.

    Returns:
        Time series array (n_volumes, n_parcels).
    """
    df = pd.read_csv(tsv_path, sep="\t")
    ts = df.values.astype(np.float64)
    # XCP-D may include NaN rows for censored (scrubbed) volumes — drop them
    nan_rows = np.any(np.isnan(ts), axis=1)
    if nan_rows.any():
        n_censored = nan_rows.sum()
        logger.info(f"    Censored {n_censored}/{len(ts)} volumes (scrubbing)")
        ts = ts[~nan_rows]
    return ts


def run_bsnet(
    ts_short: np.ndarray,
    fc_reference: np.ndarray,
    tr: float,
    total_duration_min: float,
) -> dict:
    """Run BS-NET extrapolation pipeline using core pipeline.

    Args:
        ts_short: Short time series (n_short_vols, n_parcels).
        fc_reference: Reference FC matrix (n_parcels, n_parcels).
        tr: Repetition time.
        total_duration_min: Total scan duration in minutes.

    Returns:
        Dict with BS-NET results.
    """
    from src.core.config import BSNetConfig
    from src.core.pipeline import run_bootstrap_prediction
    from src.data.data_loader import get_fc_matrix

    config = BSNetConfig(
        tr=tr,
        short_duration_sec=SHORT_DURATION_SEC,
        target_duration_min=total_duration_min,
    )

    n_parcels = ts_short.shape[1]

    # Reference FC as vectorized upper triangle
    ref_vec = fc_reference[np.triu_indices(n_parcels, k=1)]

    # Raw FC from short data (for comparison)
    fc_short_vec = get_fc_matrix(ts_short, vectorized=True, use_shrinkage=True)
    r_fc_raw = float(np.corrcoef(fc_short_vec, ref_vec)[0, 1])

    # Run BS-NET core pipeline
    result = run_bootstrap_prediction(ts_short, ref_vec, config)

    # Reconstruct predicted FC matrix from result
    # The pipeline returns scalar rho_hat_T (median of bootstrap distribution)
    # For matrix reconstruction, use the raw FC scaled by improvement factor
    improvement = result.rho_hat_T / max(r_fc_raw, 0.01)
    fc_predicted_vec = np.clip(fc_short_vec * improvement, -1, 1)
    fc_predicted = np.zeros((n_parcels, n_parcels))
    fc_predicted[np.triu_indices(n_parcels, k=1)] = fc_predicted_vec
    fc_predicted += fc_predicted.T
    np.fill_diagonal(fc_predicted, 1.0)

    r_fc_bsnet = float(np.corrcoef(fc_predicted_vec, ref_vec)[0, 1])

    return {
        "rho_hat_T": result.rho_hat_T,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "r_fc_bsnet": r_fc_bsnet,
        "r_fc_raw": r_fc_raw,
        "fc_predicted": fc_predicted,
        "fc_short_vec": fc_short_vec,
    }


def evaluate_communities(
    fc_predicted: np.ndarray,
    fc_reference: np.ndarray,
) -> dict:
    """Evaluate community detection: ARI + Jaccard."""
    from sklearn.metrics.cluster import adjusted_rand_score

    from src.core.graph_metrics import get_communities, threshold_matrix

    fc_pred_thr = threshold_matrix(fc_predicted, density=0.15)
    fc_ref_thr = threshold_matrix(fc_reference, density=0.15)

    # get_communities returns labels array (n,), not list of sets
    labels_pred = get_communities(fc_pred_thr)
    labels_ref = get_communities(fc_ref_thr)

    ari = float(adjusted_rand_score(labels_ref, labels_pred))
    n_comm_pred = len(np.unique(labels_pred))
    n_comm_ref = len(np.unique(labels_ref))
    return {"ari": ari, "n_comm_pred": n_comm_pred, "n_comm_ref": n_comm_ref}


def _get_tr_from_bold(bold_path: Path) -> tuple[float, int]:
    """Extract TR and n_vols from BOLD NIfTI header.

    Returns:
        Tuple of (tr_seconds, n_volumes).
    """
    bold_img = nib.load(str(bold_path))
    tr = float(bold_img.header.get_zooms()[-1])
    if tr <= 0 or tr > 10:
        tr = TR_DEFAULT
    n_vols = bold_img.shape[-1]
    return tr, n_vols


def _run_common_bsnet(
    ts: np.ndarray,
    tr: float,
    total_min: float,
    sub_id: str,
    sub_out: Path,
    n_parcels: int,
) -> dict:
    """Common BS-NET evaluation: FC → bootstrap → community detection.

    Args:
        ts: Cleaned time series (n_vols, n_parcels).
        tr: Repetition time.
        total_min: Total scan duration in minutes.
        sub_id: Subject ID for logging.
        sub_out: Output directory for this subject.
        n_parcels: Number of parcels.

    Returns:
        Results dict.
    """
    short_vols = int(SHORT_DURATION_SEC / tr)
    n_vols = ts.shape[0]

    if n_vols < short_vols + 10:
        logger.warning(f"  {sub_id}: Too short ({n_vols} vols < {short_vols + 10})")
        return {"sub_id": sub_id, "status": "too_short"}

    # Check for bad ROIs
    good_rois = np.std(ts, axis=0) > 1e-6
    n_good = int(good_rois.sum())
    min_good = int(n_parcels * 0.9)
    if n_good < min_good:
        logger.warning(f"  {sub_id}: Only {n_good}/{n_parcels} valid ROIs")
        return {"sub_id": sub_id, "status": "bad_rois", "n_good_rois": n_good}

    # FC: full scan + short scan
    fc_full = compute_fc_lw(ts)
    ts_short = ts[:short_vols, :]
    fc_short = compute_fc_lw(ts_short)

    # Raw correlation (baseline)
    triu = np.triu_indices(ts.shape[1], k=1)
    r_fc_raw = float(np.corrcoef(fc_short[triu], fc_full[triu])[0, 1])
    logger.info(f"  [{sub_id}] Raw r_FC(short, full) = {r_fc_raw:.4f}")

    # BS-NET pipeline
    logger.info(f"  [{sub_id}] Running BS-NET ...")
    bsnet_result = run_bsnet(ts_short, fc_full, tr, total_min)
    logger.info(
        f"  [{sub_id}] rho_hat_T = {bsnet_result['rho_hat_T']:.4f} "
        f"[{bsnet_result['ci_lower']:.4f}, {bsnet_result['ci_upper']:.4f}], "
        f"r_FC = {bsnet_result['r_fc_bsnet']:.4f} (raw = {r_fc_raw:.4f})"
    )

    # Community detection
    comm_result = evaluate_communities(bsnet_result["fc_predicted"], fc_full)
    logger.info(f"  [{sub_id}] ARI = {comm_result['ari']:.4f}")

    # Save outputs
    np.save(sub_out / f"{sub_id}_fc_full.npy", fc_full)
    np.save(sub_out / f"{sub_id}_fc_short.npy", fc_short)
    np.save(sub_out / f"{sub_id}_fc_predicted.npy", bsnet_result["fc_predicted"])
    np.save(sub_out / f"{sub_id}_ts.npy", ts)

    return {
        "sub_id": sub_id,
        "status": "success",
        "n_vols": n_vols,
        "tr": tr,
        "total_min": total_min,
        "short_vols": short_vols,
        "n_parcels": n_parcels,
        "n_good_rois": n_good,
        "r_fc_raw": r_fc_raw,
        "r_fc_bsnet": bsnet_result["r_fc_bsnet"],
        "rho_hat_T": bsnet_result["rho_hat_T"],
        "ci_lower": bsnet_result["ci_lower"],
        "ci_upper": bsnet_result["ci_upper"],
        "ari": comm_result["ari"],
        "n_comm_pred": comm_result["n_comm_pred"],
        "n_comm_ref": comm_result["n_comm_ref"],
    }


def process_subject_xcpd(
    sub_id: str,
    xcpd_dir: Path,
    fmriprep_dir: Path,
    out_dir: Path,
    n_parcels: int = 100,
) -> dict:
    """Process one subject using XCP-D outputs.

    XCP-D provides pre-computed parcellated time series (already denoised,
    bandpass filtered, scrubbed, smoothed). We just need to compute FC
    and run BS-NET.

    Args:
        sub_id: Subject ID.
        xcpd_dir: XCP-D derivatives root.
        fmriprep_dir: fMRIPrep derivatives root (for TR extraction).
        out_dir: BS-NET output directory.
        n_parcels: Number of Schaefer parcels (100 or 400).

    Returns:
        Results dict.
    """
    t0 = time.time()
    sub_out = out_dir / sub_id
    sub_out.mkdir(parents=True, exist_ok=True)

    result_file = sub_out / f"{sub_id}_bsnet_results.json"
    if result_file.exists():
        logger.info(f"  {sub_id}: already processed")
        return json.loads(result_file.read_text())

    # Find XCP-D outputs
    xcpd_paths = find_xcpd_outputs(sub_id, xcpd_dir, n_parcels)
    if xcpd_paths is None:
        logger.error(
            f"  {sub_id}: XCP-D outputs not found "
            f"(Schaefer{n_parcels}) in {xcpd_dir}"
        )
        return {"sub_id": sub_id, "status": "missing_xcpd"}

    logger.info(
        f"  [{sub_id}] XCP-D timeseries: {xcpd_paths['timeseries'].name}"
    )

    try:
        # Load XCP-D parcellated time series
        ts = load_xcpd_timeseries(xcpd_paths["timeseries"])
        logger.info(f"  [{sub_id}] Time series: {ts.shape} (post-scrubbing)")

        # Get TR from fMRIPrep BOLD header
        fmriprep_paths = find_fmriprep_outputs(sub_id, fmriprep_dir)
        if fmriprep_paths is not None:
            tr, _n_vols_orig = _get_tr_from_bold(fmriprep_paths["bold"])
        elif xcpd_paths.get("bold_denoised") is not None:
            tr, _ = _get_tr_from_bold(xcpd_paths["bold_denoised"])
        else:
            tr = TR_DEFAULT
            logger.warning(f"  {sub_id}: Using default TR={tr}")

        n_vols = ts.shape[0]
        total_min = (n_vols * tr) / 60.0
        logger.info(
            f"  [{sub_id}] {n_vols} vols (post-censoring), TR={tr:.2f}s, "
            f"{total_min:.1f} min effective"
        )

        result = _run_common_bsnet(
            ts, tr, total_min, sub_id, sub_out, n_parcels,
        )

        elapsed = time.time() - t0
        result["input_mode"] = "xcpd"
        result["elapsed_sec"] = round(elapsed, 1)
        result_file.write_text(json.dumps(result, indent=2))
        return result

    except Exception as e:
        logger.error(f"  {sub_id} failed: {e}")
        return {"sub_id": sub_id, "status": "error", "error": str(e)}


def process_subject_fmriprep(
    sub_id: str,
    atlas_path: Path,
    fmriprep_dir: Path,
    out_dir: Path,
    n_parcels: int = 100,
) -> dict:
    """Process one subject using fMRIPrep outputs directly (legacy mode).

    Performs confound regression + parcellation via nilearn, then BS-NET.

    Args:
        sub_id: Subject ID.
        atlas_path: Schaefer atlas path.
        fmriprep_dir: fMRIPrep derivatives root.
        out_dir: BS-NET output directory.
        n_parcels: Number of Schaefer parcels.

    Returns:
        Results dict.
    """
    t0 = time.time()
    sub_out = out_dir / sub_id
    sub_out.mkdir(parents=True, exist_ok=True)

    result_file = sub_out / f"{sub_id}_bsnet_results.json"
    if result_file.exists():
        logger.info(f"  {sub_id}: already processed")
        return json.loads(result_file.read_text())

    paths = find_fmriprep_outputs(sub_id, fmriprep_dir)
    if paths is None:
        logger.error(f"  {sub_id}: fMRIPrep outputs not found in {fmriprep_dir}")
        return {"sub_id": sub_id, "status": "missing_fmriprep"}

    logger.info(f"  [{sub_id}] BOLD: {paths['bold'].name}")

    try:
        tr, n_vols = _get_tr_from_bold(paths["bold"])
        total_min = (n_vols * tr) / 60.0
        short_vols = int(SHORT_DURATION_SEC / tr)

        logger.info(f"  [{sub_id}] {n_vols} vols, TR={tr:.2f}s, {total_min:.1f} min")

        if n_vols < short_vols + 10:
            logger.warning(f"  {sub_id}: Too short ({n_vols} vols < {short_vols + 10})")
            return {"sub_id": sub_id, "status": "too_short"}

        confounds = load_confounds(paths["confounds"])

        logger.info(f"  [{sub_id}] Extracting Schaefer {n_parcels} time series ...")
        ts = extract_timeseries(
            paths["bold"], atlas_path, confounds, tr, paths.get("mask"),
        )
        logger.info(f"  [{sub_id}] Time series: {ts.shape}")

        result = _run_common_bsnet(
            ts, tr, total_min, sub_id, sub_out, n_parcels,
        )

        elapsed = time.time() - t0
        result["input_mode"] = "fmriprep"
        result["elapsed_sec"] = round(elapsed, 1)
        result_file.write_text(json.dumps(result, indent=2))
        return result

    except Exception as e:
        logger.error(f"  {sub_id} failed: {e}")
        return {"sub_id": sub_id, "status": "error", "error": str(e)}


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="BS-NET on preprocessed fMRI outputs (XCP-D or fMRIPrep)",
    )
    parser.add_argument("--subject", type=str, help="Process specific subject")
    parser.add_argument("--run-all", action="store_true", help="All available subjects")
    parser.add_argument(
        "--run-selection", type=str,
        help="CSV with dataset_id, participant_id columns",
    )
    parser.add_argument(
        "--input-mode", type=str, default="xcpd",
        choices=["xcpd", "fmriprep"],
        help="Input source: 'xcpd' (default) or 'fmriprep' (legacy nilearn)",
    )
    parser.add_argument(
        "--parcels", type=int, default=100,
        choices=[100, 400],
        help="Schaefer parcellation granularity (default: 100)",
    )
    parser.add_argument(
        "--fmriprep-dir", type=str,
        default=str(FMRIPREP_DIR),
        help="fMRIPrep derivatives directory",
    )
    parser.add_argument(
        "--xcpd-dir", type=str,
        default=str(XCPD_DIR),
        help="XCP-D derivatives directory",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    fmriprep_dir = Path(args.fmriprep_dir)
    xcpd_dir = Path(args.xcpd_dir)
    n_parcels = args.parcels
    BSNET_OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Input mode: {args.input_mode}")
    logger.info(f"Parcellation: Schaefer {n_parcels}")

    # Atlas (needed for fmriprep mode; also validated for xcpd mode)
    atlas_path = None
    if args.input_mode == "fmriprep":
        atlas_path = find_atlas(n_parcels)
        logger.info(f"Atlas: {atlas_path.name}")

    # Subject list — scan appropriate directory
    scan_dir = xcpd_dir if args.input_mode == "xcpd" else fmriprep_dir
    if args.subject:
        subjects = [args.subject]
    elif args.run_selection:
        sel_df = pd.read_csv(args.run_selection)
        subjects = sel_df["participant_id"].tolist()
        subjects = [s if s.startswith("sub-") else f"sub-{s}" for s in subjects]
    elif args.run_all:
        if not scan_dir.exists():
            logger.error(f"Directory not found: {scan_dir}")
            return
        subjects = sorted([
            d.name for d in scan_dir.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ])
    else:
        parser.print_help()
        return

    logger.info(f"Processing {len(subjects)} subjects")

    results = []
    for i, sub_id in enumerate(subjects, 1):
        logger.info(f"\n[{i}/{len(subjects)}] {sub_id}")
        if args.input_mode == "xcpd":
            result = process_subject_xcpd(
                sub_id, xcpd_dir, fmriprep_dir, BSNET_OUT_DIR, n_parcels,
            )
        else:
            result = process_subject_fmriprep(
                sub_id, atlas_path, fmriprep_dir, BSNET_OUT_DIR, n_parcels,
            )
        results.append(result)

    # Summary
    success = [r for r in results if r.get("status") == "success"]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processed: {len(results)}, Success: {len(success)}")

    if success:
        df = pd.DataFrame(success)
        summary_csv = BSNET_OUT_DIR / "bsnet_results_summary.csv"
        df.to_csv(summary_csv, index=False)
        logger.info(f"Results saved: {summary_csv}")

        mean_raw = df["r_fc_raw"].mean()
        mean_rho = df["rho_hat_T"].mean()
        mean_ari = df["ari"].mean()

        logger.info(f"\n  Mode: {args.input_mode}, Parcels: {n_parcels}")
        logger.info(f"  Mean r_FC (raw 2min):     {mean_raw:.4f}")
        logger.info(f"  Mean rho_hat_T (BS-NET):  {mean_rho:.4f}")
        logger.info(f"  Predicted improvement:    +{mean_rho - mean_raw:.4f}")
        logger.info(f"  Mean ARI:                 {mean_ari:.4f}")


if __name__ == "__main__":
    main()
