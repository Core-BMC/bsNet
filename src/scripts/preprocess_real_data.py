"""Preprocess raw BOLD + T1w for BS-NET real-data validation.

Pipeline (per subject):
    1. T1w → MNI affine registration (dipy)
    2. Mean EPI → T1w coregistration (dipy)
    3. Compose: EPI → T1w → MNI
    4. Apply transform to 4D BOLD (volume-by-volume)
    5. Extract Schaefer 100-parcel time series (nilearn)
    6. Bandpass filter (0.01–0.1 Hz), detrend, z-score
    7. Compute FC (Ledoit-Wolf shrinkage): full + short (2 min)
    8. BS-NET pipeline: bootstrap → SB extrapolation → prior → attenuation correction
    9. Community detection → ARI / Jaccard vs reference FC

Usage (from bsNet/ project root):
    python -m src.scripts.preprocess_real_data --subject sub-10159 --verbose
    python src/scripts/preprocess_real_data.py --subject sub-10159 --verbose
    python src/scripts/preprocess_real_data.py --run-all
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

# Ensure project root is on sys.path so `from src.core...` works
# regardless of invocation method (python -m / python script.py)
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
DERIV_DIR = DATA_DIR / "derivatives"
# openneuro-py may nest under 1.0.0/uncompressed/ or place directly.
# Both paths are searched per-file because downloads can be mixed.
_RAW_DIRS = [
    DATA_DIR / "openneuro" / "ds000030",
    DATA_DIR / "openneuro" / "ds000030" / "1.0.0" / "uncompressed",
]

# --- Constants ---
TR = 2.0
SHORT_DURATION_SEC = 120
SHORT_VOLUMES = int(SHORT_DURATION_SEC / TR)  # 60


def _find_raw_file(rel_path: str) -> Path | None:
    """Search _RAW_DIRS for a file by relative path (e.g. 'sub-10159/anat/sub-10159_T1w.nii.gz')."""
    for root in _RAW_DIRS:
        candidate = root / rel_path
        if candidate.exists():
            return candidate
    return None


# ============================================================================
# Registration
# ============================================================================
def _dipy_affine_register(
    static: np.ndarray,
    static_affine: np.ndarray,
    moving: np.ndarray,
    moving_affine: np.ndarray,
    level_iters: list[int] | None = None,
) -> np.ndarray:
    """Run dipy COM → Rigid → Affine registration pipeline.

    Returns:
        4x4 affine matrix.
    """
    from dipy.align.imaffine import (
        AffineRegistration,
        MutualInformationMetric,
        transform_centers_of_mass,
    )
    from dipy.align.transforms import AffineTransform3D, RigidTransform3D

    if level_iters is None:
        level_iters = [1000, 100, 10]

    c_of_mass = transform_centers_of_mass(
        static, static_affine, moving, moving_affine
    )
    metric = MutualInformationMetric(nbins=32, sampling_proportion=0.5)
    affreg = AffineRegistration(metric=metric, level_iters=level_iters)

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
    return affine_result.affine


def register_t1w_to_mni(
    t1w_path: Path,
    template_img: nib.Nifti1Image,
) -> np.ndarray:
    """Register T1w to MNI152 template.

    Args:
        t1w_path: Path to T1w NIfTI.
        template_img: MNI152 template.

    Returns:
        4x4 affine transform (T1w → MNI).
    """
    t1w_img = nib.load(str(t1w_path))
    return _dipy_affine_register(
        template_img.get_fdata(), template_img.affine,
        t1w_img.get_fdata(), t1w_img.affine,
    )


def register_epi_to_t1w(
    bold_path: Path,
    t1w_path: Path,
) -> np.ndarray:
    """Coregister mean EPI to T1w.

    Args:
        bold_path: Path to 4D BOLD.
        t1w_path: Path to T1w.

    Returns:
        4x4 affine transform (EPI → T1w).
    """
    from nilearn import image as nimg

    mean_epi = nimg.mean_img(str(bold_path))
    t1w_img = nib.load(str(t1w_path))
    return _dipy_affine_register(
        t1w_img.get_fdata(), t1w_img.affine,
        mean_epi.get_fdata(), mean_epi.affine,
    )


def compose_transforms(
    epi_to_t1w: np.ndarray,
    t1w_to_mni: np.ndarray,
) -> np.ndarray:
    """Compose EPI→T1w and T1w→MNI into EPI→MNI.

    Args:
        epi_to_t1w: 4x4 affine (EPI → T1w).
        t1w_to_mni: 4x4 affine (T1w → MNI).

    Returns:
        4x4 affine (EPI → MNI).
    """
    return t1w_to_mni @ epi_to_t1w


# ============================================================================
# Time-series extraction (memory-efficient)
# ============================================================================
def extract_timeseries_volumewise(
    bold_path: Path,
    epi_to_mni_affine: np.ndarray,
    template_img: nib.Nifti1Image,
    atlas_path: Path,
) -> np.ndarray:
    """Extract parcel time series volume-by-volume (memory-efficient).

    Avoids loading full 4D MNI-registered BOLD into memory.

    Args:
        bold_path: Path to 4D BOLD (native space).
        epi_to_mni_affine: Composed EPI→MNI affine.
        template_img: MNI template (defines output grid).
        atlas_path: Path to parcellation NIfTI in MNI space.

    Returns:
        Time series array (n_volumes, n_parcels).
    """
    from dipy.align.imaffine import AffineMap

    epi_img = nib.load(str(bold_path))
    epi_data = epi_img.get_fdata()
    n_vols = epi_data.shape[-1]

    atlas_data = nib.load(str(atlas_path)).get_fdata()
    n_parcels = int(atlas_data.max())

    aff_map = AffineMap(
        epi_to_mni_affine,
        template_img.shape, template_img.affine,
        epi_img.shape[:3], epi_img.affine,
    )

    # Pre-compute parcel masks
    parcel_masks = {}
    for p in range(1, n_parcels + 1):
        mask = atlas_data == p
        if mask.sum() > 0:
            parcel_masks[p] = mask

    ts = np.zeros((n_vols, n_parcels), dtype=np.float32)
    for t in range(n_vols):
        vol_mni = aff_map.transform(epi_data[..., t]).astype(np.float32)
        for p, mask in parcel_masks.items():
            ts[t, p - 1] = vol_mni[mask].mean()

    return ts


def clean_timeseries(
    ts: np.ndarray,
    tr: float = TR,
    low_pass: float = 0.1,
    high_pass: float = 0.01,
) -> np.ndarray:
    """Apply bandpass filter, detrend, and z-score to time series.

    Args:
        ts: Raw time series (n_volumes, n_parcels).
        tr: Repetition time in seconds.
        low_pass: Low-pass cutoff frequency (Hz).
        high_pass: High-pass cutoff frequency (Hz).

    Returns:
        Cleaned time series.
    """
    from nilearn.signal import clean

    return clean(
        ts,
        detrend=True,
        standardize="zscore_sample",
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=tr,
    )


# ============================================================================
# FC computation
# ============================================================================
def compute_fc_lw(ts: np.ndarray) -> np.ndarray:
    """Compute FC matrix using Ledoit-Wolf shrinkage.

    Args:
        ts: Time series (n_volumes, n_parcels).

    Returns:
        FC matrix (n_parcels, n_parcels), diagonal zeroed.
    """
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf()
    cov = lw.fit(ts).covariance_
    d = np.sqrt(np.diag(cov))
    d[d == 0] = 1e-10
    fc = cov / np.outer(d, d)
    fc = np.clip(fc, -1.0, 1.0)
    np.fill_diagonal(fc, 0)
    return fc


# ============================================================================
# BS-NET pipeline
# ============================================================================
def run_bsnet_pipeline(
    ts_short: np.ndarray,
    fc_reference: np.ndarray,
) -> dict:
    """Run BS-NET extrapolation on short time series.

    Args:
        ts_short: Short time series (n_short_vols, n_parcels).
        fc_reference: Reference FC from full scan.

    Returns:
        Dict with rho_hat_T, fc_predicted, r_fc (correlation with reference).
    """
    from src.core.bootstrap import (
        block_bootstrap_indices,
        correct_attenuation,
        fisher_z,
        fisher_z_inv,
    )
    from src.core.config import BSNetConfig

    config = BSNetConfig()
    n_parcels = ts_short.shape[1]

    # Step 1: FC from short data (LW shrinkage)
    fc_short = compute_fc_lw(ts_short)

    # Step 2: Block bootstrap
    n_samples = ts_short.shape[0]
    block_size = max(10, n_samples // 6)
    n_bootstraps = config.n_bootstraps

    boot_fcs = []
    for _ in range(n_bootstraps):
        indices = block_bootstrap_indices(n_samples, block_size)
        ts_boot = ts_short[indices, :]
        fc_boot = compute_fc_lw(ts_boot)
        boot_fcs.append(fc_boot[np.triu_indices(n_parcels, k=1)])

    boot_fcs = np.array(boot_fcs)

    # Step 3: Spearman-Brown extrapolation (k = target / short)
    k = config.target_duration_min * 60 / (n_samples * TR)
    fc_short_vec = fc_short[np.triu_indices(n_parcels, k=1)]
    z_short = fisher_z(fc_short_vec)
    z_extrapolated = z_short * k / (1 + (k - 1) * z_short / fisher_z(np.ones_like(z_short) * 0.99))
    # Simplified SB in z-space
    rho_sb = fisher_z_inv(z_extrapolated * np.sqrt(k) / np.sqrt(1 + (k - 1) * np.abs(z_extrapolated)))

    # Step 4: Bayesian empirical prior (SB-extrapolated estimate + bootstrap variance)
    prior_mean, prior_var = config.empirical_prior
    boot_var = boot_fcs.var(axis=0)
    # Use SB-extrapolated values as the data estimate, not raw bootstrap mean
    shrunk = (boot_var * prior_mean + prior_var * rho_sb) / (boot_var + prior_var)

    # Step 5: Attenuation correction
    rho_hat_T = correct_attenuation(shrunk, config.reliability_coeff)
    rho_hat_T = np.clip(rho_hat_T, -1, 1)

    # Reconstruct matrix
    fc_predicted = np.zeros((n_parcels, n_parcels))
    fc_predicted[np.triu_indices(n_parcels, k=1)] = rho_hat_T
    fc_predicted += fc_predicted.T

    # Evaluate against reference
    ref_vec = fc_reference[np.triu_indices(n_parcels, k=1)]
    r_fc = np.corrcoef(rho_hat_T, ref_vec)[0, 1]
    r_fc_raw = np.corrcoef(fc_short_vec, ref_vec)[0, 1]

    return {
        "rho_hat_T_mean": float(rho_hat_T.mean()),
        "r_fc_bsnet": float(r_fc),
        "r_fc_raw": float(r_fc_raw),
        "fc_predicted": fc_predicted,
    }


# ============================================================================
# Community detection + ARI/Jaccard
# ============================================================================
def evaluate_communities(
    fc_predicted: np.ndarray,
    fc_reference: np.ndarray,
    network_labels: np.ndarray,
) -> dict:
    """Evaluate community detection accuracy.

    Args:
        fc_predicted: Predicted FC matrix.
        fc_reference: Reference FC matrix.
        network_labels: Parcel-to-network assignment (1-indexed).

    Returns:
        Dict with ARI and per-network Jaccard overlap.
    """
    from sklearn.metrics.cluster import adjusted_rand_score

    from src.core.graph_metrics import get_communities, threshold_matrix

    # Threshold and detect communities
    fc_pred_thr = threshold_matrix(fc_predicted, density=0.15)
    fc_ref_thr = threshold_matrix(fc_reference, density=0.15)

    comm_pred = get_communities(fc_pred_thr)
    comm_ref = get_communities(fc_ref_thr)

    # ARI
    n = fc_predicted.shape[0]
    labels_pred = np.zeros(n, dtype=int)
    labels_ref = np.zeros(n, dtype=int)
    for i, nodes in enumerate(comm_pred):
        for node in nodes:
            labels_pred[node] = i
    for i, nodes in enumerate(comm_ref):
        for node in nodes:
            labels_ref[node] = i

    ari = adjusted_rand_score(labels_ref, labels_pred)

    # Per-network Jaccard
    unique_nets = sorted(set(network_labels))
    jaccard_per_net = {}
    for net in unique_nets:
        net_nodes = set(np.where(network_labels == net)[0])
        # Find best matching community in predicted
        best_jaccard = 0
        for comm in comm_pred:
            intersection = len(net_nodes & comm)
            union = len(net_nodes | comm)
            if union > 0:
                j = intersection / union
                best_jaccard = max(best_jaccard, j)
        jaccard_per_net[int(net)] = float(best_jaccard)

    return {"ari": float(ari), "jaccard_per_network": jaccard_per_net}


# ============================================================================
# Load network labels from TSV
# ============================================================================
def load_network_labels() -> tuple[np.ndarray, dict[int, str]]:
    """Load Schaefer parcel-to-network mapping.

    Returns:
        (network_ids, id_to_name): array of network IDs per parcel,
            and mapping from ID to network name.
    """
    import pandas as pd

    tsv_path = ATLAS_DIR / "schaefer100_7networks_labels.tsv"
    df = pd.read_csv(tsv_path, sep="\t")

    networks = df["network"].unique()
    net_to_id = {n: i for i, n in enumerate(networks)}
    id_to_name = {i: n for n, i in net_to_id.items()}

    network_ids = np.array([net_to_id[n] for n in df["network"]])
    return network_ids, id_to_name


# ============================================================================
# Main subject processor
# ============================================================================
def process_subject(
    sub_id: str,
    atlas_path: Path,
    template_img: nib.Nifti1Image,
    network_labels: np.ndarray,
    verbose: bool = False,
) -> dict:
    """Full pipeline for one subject.

    Args:
        sub_id: Subject ID.
        atlas_path: Path to atlas NIfTI.
        template_img: MNI152 template.
        network_labels: Network assignment per parcel.
        verbose: Print progress.

    Returns:
        Results dict.
    """
    t0 = time.time()
    out_dir = DERIV_DIR / sub_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check required files (search across all known raw directories)
    t1w_path = _find_raw_file(f"{sub_id}/anat/{sub_id}_T1w.nii.gz")
    bold_path = _find_raw_file(f"{sub_id}/func/{sub_id}_task-rest_bold.nii.gz")

    if t1w_path is None:
        logger.error(f"  T1w not found for {sub_id} in {[str(d) for d in _RAW_DIRS]}")
        return {"sub_id": sub_id, "status": "missing_t1w"}
    if bold_path is None:
        logger.error(f"  BOLD not found for {sub_id} in {[str(d) for d in _RAW_DIRS]}")
        return {"sub_id": sub_id, "status": "missing_bold"}
    logger.info(f"  T1w:  {t1w_path}")
    logger.info(f"  BOLD: {bold_path}")

    # Check if already processed
    result_file = out_dir / f"{sub_id}_results.json"
    if result_file.exists():
        logger.info(f"  {sub_id}: already processed, loading results")
        return json.loads(result_file.read_text())

    try:
        # Step 1: T1w → MNI
        logger.info(f"  [{sub_id}] T1w → MNI registration...")
        t1w_to_mni = register_t1w_to_mni(t1w_path, template_img)

        # Step 2: EPI → T1w
        logger.info(f"  [{sub_id}] EPI → T1w coregistration...")
        epi_to_t1w = register_epi_to_t1w(bold_path, t1w_path)

        # Step 3: Compose EPI → MNI
        epi_to_mni = compose_transforms(epi_to_t1w, t1w_to_mni)
        np.save(out_dir / f"{sub_id}_epi2mni_affine.npy", epi_to_mni)

        # Step 4: Extract time series
        logger.info(f"  [{sub_id}] Extracting time series (volume-by-volume)...")
        ts_raw = extract_timeseries_volumewise(
            bold_path, epi_to_mni, template_img, atlas_path
        )

        # Step 5: Clean (bandpass + detrend + zscore)
        logger.info(f"  [{sub_id}] Cleaning time series...")
        ts = clean_timeseries(ts_raw)
        np.save(out_dir / f"{sub_id}_ts_mni.npy", ts)

        # Step 6: FC matrices
        logger.info(f"  [{sub_id}] Computing FC...")
        fc_full = compute_fc_lw(ts)
        np.save(out_dir / f"{sub_id}_fc_full.npy", fc_full)

        n_short = min(SHORT_VOLUMES, ts.shape[0])
        ts_short = ts[:n_short, :]
        fc_short = compute_fc_lw(ts_short)
        np.save(out_dir / f"{sub_id}_fc_short.npy", fc_short)

        # Step 7: BS-NET
        logger.info(f"  [{sub_id}] Running BS-NET pipeline...")
        bsnet = run_bsnet_pipeline(ts_short, fc_full)

        # Step 8: Community evaluation
        logger.info(f"  [{sub_id}] Evaluating communities...")
        comm_eval = evaluate_communities(
            bsnet["fc_predicted"], fc_full, network_labels
        )

        elapsed = time.time() - t0
        triu = np.triu_indices(100, k=1)
        result = {
            "sub_id": sub_id,
            "status": "success",
            "elapsed_sec": round(elapsed, 1),
            "n_volumes": int(ts.shape[0]),
            "n_volumes_short": n_short,
            "fc_full_mean": float(fc_full[triu].mean()),
            "fc_short_mean": float(fc_short[triu].mean()),
            "r_fc_raw_vs_full": float(np.corrcoef(
                fc_short[triu], fc_full[triu]
            )[0, 1]),
            "r_fc_bsnet": bsnet["r_fc_bsnet"],
            "r_fc_raw": bsnet["r_fc_raw"],
            "rho_hat_T_mean": bsnet["rho_hat_T_mean"],
            "ari": comm_eval["ari"],
            "jaccard_per_network": comm_eval["jaccard_per_network"],
        }

        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(
            f"  [{sub_id}] DONE in {elapsed:.0f}s — "
            f"r_FC(BS-NET)={bsnet['r_fc_bsnet']:.3f}, "
            f"r_FC(raw)={bsnet['r_fc_raw']:.3f}, "
            f"ARI={comm_eval['ari']:.3f}"
        )
        return result

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"  [{sub_id}] FAILED in {elapsed:.0f}s — {e}")
        result = {"sub_id": sub_id, "status": "failed", "error": str(e)}
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        return result


# ============================================================================
# CLI
# ============================================================================
def get_hc_subjects() -> list[str]:
    """Get healthy control subjects with rest + T1w."""
    import pandas as pd

    tsv = _find_raw_file("participants.tsv")
    if tsv is None:
        # Fall back to listing directories across all raw dirs
        subs = set()
        for raw_dir in _RAW_DIRS:
            if raw_dir.exists():
                subs.update(
                    d.name for d in raw_dir.iterdir()
                    if d.is_dir() and d.name.startswith("sub-")
                )
        return sorted(subs)

    df = pd.read_csv(tsv, sep="\t")
    hc = df[(df["diagnosis"] == "CONTROL") & (df["rest"] == 1) & (df["T1w"] == 1)]
    return sorted(hc["participant_id"].tolist())


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="BS-NET real-data preprocessing")
    parser.add_argument("--subject", type=str, help="Process specific subject")
    parser.add_argument("--test-one", action="store_true", help="Process first HC")
    parser.add_argument("--run-all", action="store_true", help="Process all HCs")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Atlas — search multiple candidate locations
    _atlas_name = "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
    _atlas_candidates = [
        ATLAS_DIR / _atlas_name,
        # nilearn default cache (fetch_atlas_schaefer_2018)
        PROJECT_ROOT.parent / "nilearn_data" / "schaefer_2018" / _atlas_name,
        Path.home() / "nilearn_data" / "schaefer_2018" / _atlas_name,
    ]
    atlas_path = None
    for _cand in _atlas_candidates:
        if _cand.exists():
            atlas_path = _cand
            break
    if atlas_path is None:
        logger.error(f"Atlas not found. Searched: {[str(c) for c in _atlas_candidates]}")
        logger.error("Run setup_local_env.sh or: nilearn.datasets.fetch_atlas_schaefer_2018()")
        sys.exit(1)

    atlas_img = nib.load(str(atlas_path))
    n_parcels = int(atlas_img.get_fdata().max())
    logger.info(f"Atlas: {atlas_img.shape}, {n_parcels} parcels")

    # Template
    from nilearn.datasets import load_mni152_template
    template = load_mni152_template(resolution=2)

    # Network labels
    network_labels, net_names = load_network_labels()
    logger.info(f"Networks: {net_names}")

    # Subject list
    hc_subjects = get_hc_subjects()
    if args.subject:
        subjects = [args.subject]
    elif args.test_one:
        subjects = hc_subjects[:1]
    elif args.run_all:
        subjects = hc_subjects
    else:
        logger.info("Use --subject, --test-one, or --run-all")
        return

    logger.info(f"Processing {len(subjects)} subjects...")

    results = []
    for i, sub in enumerate(subjects):
        logger.info(f"\n[{i+1}/{len(subjects)}] {sub}")
        r = process_subject(sub, atlas_path, template, network_labels, args.verbose)
        results.append(r)

    # Summary
    ok = [r for r in results if r.get("status") == "success"]
    fail = [r for r in results if r.get("status") == "failed"]
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: {len(ok)} success, {len(fail)} failed out of {len(subjects)}")

    if ok:
        r_bsnet = [r["r_fc_bsnet"] for r in ok]
        r_raw = [r["r_fc_raw"] for r in ok]
        aris = [r["ari"] for r in ok]
        logger.info(f"  r_FC (BS-NET): {np.mean(r_bsnet):.3f} ± {np.std(r_bsnet):.3f}")
        logger.info(f"  r_FC (raw):    {np.mean(r_raw):.3f} ± {np.std(r_raw):.3f}")
        logger.info(f"  ARI:           {np.mean(aris):.3f} ± {np.std(aris):.3f}")

    # Save summary
    DERIV_DIR.mkdir(parents=True, exist_ok=True)
    with open(DERIV_DIR / "real_data_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {DERIV_DIR / 'real_data_results.json'}")


if __name__ == "__main__":
    main()
