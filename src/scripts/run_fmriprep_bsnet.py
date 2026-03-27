"""Run BS-NET pipeline on fMRIPrep-preprocessed data.

Takes fMRIPrep outputs (MNI-space BOLD + confounds) and runs:
    1. Confound regression (select confounds from fMRIPrep TSV)
    2. Schaefer 100-parcel time series extraction (nilearn NiftiLabelsMasker)
    3. Bandpass filter (0.01–0.1 Hz), detrend, z-score
    4. FC computation (Ledoit-Wolf): full scan + short (first 2 min)
    5. BS-NET: bootstrap → SB extrapolation → Bayesian prior → attenuation correction
    6. Community detection → ARI / Jaccard vs reference FC

Usage (from bsNet/ project root):
    python src/scripts/run_fmriprep_bsnet.py --subject sub-10159 --verbose
    python src/scripts/run_fmriprep_bsnet.py --run-all
    python src/scripts/run_fmriprep_bsnet.py --run-selection data/hc_100_selection.csv
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


def find_atlas() -> Path:
    """Search multiple locations for Schaefer 100 atlas."""
    name = "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
    candidates = [
        ATLAS_DIR / name,
        PROJECT_ROOT.parent / "nilearn_data" / "schaefer_2018" / name,
        Path.home() / "nilearn_data" / "schaefer_2018" / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    msg = f"Atlas not found. Searched: {[str(c) for c in candidates]}"
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


def process_subject(
    sub_id: str,
    atlas_path: Path,
    fmriprep_dir: Path,
    out_dir: Path,
    verbose: bool = False,
) -> dict:
    """Process one subject: fMRIPrep outputs → BS-NET evaluation.

    Args:
        sub_id: Subject ID.
        atlas_path: Schaefer atlas path.
        fmriprep_dir: fMRIPrep derivatives root.
        out_dir: BS-NET output directory.
        verbose: Print progress.

    Returns:
        Results dict.
    """
    t0 = time.time()
    sub_out = out_dir / sub_id
    sub_out.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    result_file = sub_out / f"{sub_id}_bsnet_results.json"
    if result_file.exists():
        logger.info(f"  {sub_id}: already processed")
        return json.loads(result_file.read_text())

    # Find fMRIPrep outputs
    paths = find_fmriprep_outputs(sub_id, fmriprep_dir)
    if paths is None:
        logger.error(f"  {sub_id}: fMRIPrep outputs not found in {fmriprep_dir}")
        return {"sub_id": sub_id, "status": "missing_fmriprep"}

    logger.info(f"  [{sub_id}] BOLD: {paths['bold'].name}")

    try:
        # Get TR from BOLD header
        bold_img = nib.load(str(paths["bold"]))
        tr = float(bold_img.header.get_zooms()[-1])
        if tr <= 0 or tr > 10:
            tr = TR_DEFAULT
        n_vols = bold_img.shape[-1]
        total_min = (n_vols * tr) / 60.0
        short_vols = int(SHORT_DURATION_SEC / tr)

        logger.info(f"  [{sub_id}] {n_vols} vols, TR={tr:.2f}s, {total_min:.1f} min")

        if n_vols < short_vols + 10:
            logger.warning(f"  {sub_id}: Too short ({n_vols} vols < {short_vols + 10})")
            return {"sub_id": sub_id, "status": "too_short"}

        # Load confounds
        confounds = load_confounds(paths["confounds"])

        # Extract time series
        logger.info(f"  [{sub_id}] Extracting Schaefer 100 time series ...")
        ts = extract_timeseries(
            paths["bold"], atlas_path, confounds, tr, paths.get("mask"),
        )
        logger.info(f"  [{sub_id}] Time series: {ts.shape}")

        # Check for bad ROIs
        good_rois = np.std(ts, axis=0) > 1e-6
        n_good = good_rois.sum()
        if n_good < 90:
            logger.warning(f"  {sub_id}: Only {n_good}/100 valid ROIs")
            return {"sub_id": sub_id, "status": "bad_rois", "n_good_rois": int(n_good)}

        # FC: full scan + short scan
        fc_full = compute_fc_lw(ts)
        ts_short = ts[:short_vols, :]
        fc_short = compute_fc_lw(ts_short)

        # Raw correlation (baseline)
        triu = np.triu_indices(ts.shape[1], k=1)
        r_fc_raw = float(np.corrcoef(
            fc_short[triu], fc_full[triu],
        )[0, 1])
        logger.info(f"  [{sub_id}] Raw r_FC(short, full) = {r_fc_raw:.4f}")

        # BS-NET pipeline
        logger.info(f"  [{sub_id}] Running BS-NET ...")
        bsnet_result = run_bsnet(ts_short, fc_full, tr, total_min)
        logger.info(
            f"  [{sub_id}] BS-NET rho_hat_T = {bsnet_result['rho_hat_T']:.4f} "
            f"[{bsnet_result['ci_lower']:.4f}, {bsnet_result['ci_upper']:.4f}], "
            f"r_FC = {bsnet_result['r_fc_bsnet']:.4f} (raw = {bsnet_result['r_fc_raw']:.4f})"
        )

        # Community detection
        comm_result = evaluate_communities(bsnet_result["fc_predicted"], fc_full)
        logger.info(f"  [{sub_id}] ARI = {comm_result['ari']:.4f}")

        elapsed = time.time() - t0

        # Save outputs
        np.save(sub_out / f"{sub_id}_fc_full.npy", fc_full)
        np.save(sub_out / f"{sub_id}_fc_short.npy", fc_short)
        np.save(sub_out / f"{sub_id}_fc_predicted.npy", bsnet_result["fc_predicted"])
        np.save(sub_out / f"{sub_id}_ts.npy", ts)

        result = {
            "sub_id": sub_id,
            "status": "success",
            "n_vols": n_vols,
            "tr": tr,
            "total_min": total_min,
            "short_vols": short_vols,
            "n_good_rois": int(n_good),
            "r_fc_raw": r_fc_raw,
            "r_fc_bsnet": bsnet_result["r_fc_bsnet"],
            "rho_hat_T": bsnet_result["rho_hat_T"],
            "ci_lower": bsnet_result["ci_lower"],
            "ci_upper": bsnet_result["ci_upper"],
            "ari": comm_result["ari"],
            "n_comm_pred": comm_result["n_comm_pred"],
            "n_comm_ref": comm_result["n_comm_ref"],
            "elapsed_sec": round(elapsed, 1),
        }

        result_file.write_text(json.dumps(result, indent=2))
        return result

    except Exception as e:
        logger.error(f"  {sub_id} failed: {e}")
        return {"sub_id": sub_id, "status": "error", "error": str(e)}


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="BS-NET on fMRIPrep outputs",
    )
    parser.add_argument("--subject", type=str, help="Process specific subject")
    parser.add_argument("--run-all", action="store_true", help="All fMRIPrep subjects")
    parser.add_argument(
        "--run-selection", type=str,
        help="CSV with dataset_id, participant_id columns",
    )
    parser.add_argument(
        "--fmriprep-dir", type=str,
        default=str(FMRIPREP_DIR),
        help="fMRIPrep derivatives directory",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    fmriprep_dir = Path(args.fmriprep_dir)
    BSNET_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Atlas
    atlas_path = find_atlas()
    atlas_img = nib.load(str(atlas_path))
    n_parcels = int(atlas_img.get_fdata().max())
    logger.info(f"Atlas: {atlas_path.name}, {n_parcels} parcels")

    # Subject list
    if args.subject:
        subjects = [args.subject]
    elif args.run_selection:
        sel_df = pd.read_csv(args.run_selection)
        subjects = sel_df["participant_id"].tolist()
        subjects = [s if s.startswith("sub-") else f"sub-{s}" for s in subjects]
    elif args.run_all:
        subjects = sorted([
            d.name for d in fmriprep_dir.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ])
    else:
        parser.print_help()
        return

    logger.info(f"Processing {len(subjects)} subjects")

    results = []
    for i, sub_id in enumerate(subjects, 1):
        logger.info(f"\n[{i}/{len(subjects)}] {sub_id}")
        result = process_subject(
            sub_id, atlas_path, fmriprep_dir, BSNET_OUT_DIR, args.verbose,
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

        logger.info(f"\n  Mean r_FC (raw 2min):     {mean_raw:.4f}")
        logger.info(f"  Mean rho_hat_T (BS-NET):  {mean_rho:.4f}")
        logger.info(f"  Predicted improvement:    +{mean_rho - mean_raw:.4f}")
        logger.info(f"  Mean ARI:                 {mean_ari:.4f}")


if __name__ == "__main__":
    main()
