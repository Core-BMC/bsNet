import logging
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.data.data_loader import create_masker, fetch_schaefer_atlas, get_fc_matrix

logger = logging.getLogger(__name__)

# Connect to local MoBSE repo
_mobse_path = os.environ.get("MOBSE_PATH", str(Path.home() / "GitHub" / "MoBSE"))
sys.path.append(_mobse_path)
try:
    from mobse.data.nuisance import build_paper_nuisance_confounds
    from mobse.data.prepare import _download_openneuro_rest_bold

    logger.info("Successfully connected to local MoBSE repository.")
except ImportError as e:
    logger.error(f"Error importing from MoBSE: {e}")
    sys.exit(1)


def run_empirical_pipeline() -> None:
    """Run BS-NET on empirical resting-state fMRI data from OpenNeuro.

    Integrates MoBSE data loading and preprocessing with BS-NET core
    prediction pipeline to validate on real neuroimaging data.
    """
    print("--- Phase 2: OpenNeuro Real Data Integration ---")

    cache_dir = Path("data/openneuro")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch empirical data using MoBSE optimized logic
    logger.info("Fetching ds000030 adults (HC)...")
    print("Fetching ds000030 adults (HC)...")
    fetch_res = _download_openneuro_rest_bold(
        dataset_id="ds000030",
        snapshot_tag=None,
        task="rest",
        n_subjects=1,  # fetch 1 subject to test pipeline
        min_age=18,
        diagnosis="CONTROL",
        strict_hc=True,
        api_url="https://openneuro.org/crn/graphql",
        cache_root=cache_dir,
        progress=None,
    )

    bold_files = fetch_res["bold_files"]
    if not bold_files:
        logger.warning("No BOLD files downloaded.")
        print("No BOLD files downloaded.")
        return

    target_bold = bold_files[0]
    logger.info(f"Process BOLD target: {target_bold}")
    print(f"Process BOLD target: {target_bold}")

    # Extract TR
    img = nib.load(target_bold)
    tr = img.header.get_zooms()[3]
    if tr <= 0.0 or tr > 5.0:
        tr = 2.0  # fallback

    logger.info(f"Extracted TR: {tr}s")
    print(f"Extracted TR: {tr}s")

    # 2. Nuisance Regression
    logger.info("Applying MoBSE CompCor/GSR Denoising...")
    print("Applying MoBSE CompCor/GSR Denoising...")
    nuisance = build_paper_nuisance_confounds(
        bold_path=target_bold,
        tr=tr,
        external_confounds=None,  # Pure data-driven
        include_compcor=True,
        compcor_components=5,
        include_gsr=True,
        add_derivatives=True,
        add_quadratic=True,
    )

    # 3. Atlas Masking
    logger.info("Extracting Schaefer 100 networks...")
    print("Extracting Schaefer 100 networks...")
    atlas = fetch_schaefer_atlas(n_rois=100)
    masker = create_masker(
        atlas.maps,
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.008,
        t_r=tr,
    )

    time_series = masker.fit_transform(target_bold, confounds=nuisance)
    T_samples = time_series.shape[0]
    total_min = (T_samples * tr) / 60

    logger.info(f"Extracted Clean Time-series: {time_series.shape}")
    print(f"Extracted Clean Time-series: {time_series.shape} ({total_min:.1f} "
          f"minutes total)")

    # 4. BS-NET Execution
    short_len_min = 2.0
    t_samples = int(short_len_min * 60 / tr)

    if t_samples >= T_samples:
        logger.warning(
            f"Full scan is only {total_min:.1f}m, which is too short for validation."
        )
        print(
            f"Warning: Full scan is only {total_min:.1f}m, "
            f"which is too short for validation. Exiting."
        )
        return

    # Oracle Truth
    fc_true_T = get_fc_matrix(time_series, vectorized=True)

    # Short Slice
    short_obs = time_series[:t_samples, :]

    logger.info("Running BS-NET Core Prediction")
    print("\n--- Running BS-NET Core Prediction ---")

    # Create config for empirical data
    config = BSNetConfig(
        n_rois=100,
        tr=tr,
        short_duration_sec=short_len_min * 60,
        target_duration_min=total_min,
        n_bootstraps=100,
    )

    # Empirical Fake-Oracle Model for test (mock prediction from neural network)
    fc_pred_t = fc_true_T + 0.1 * np.random.randn(*fc_true_T.shape)

    # Run bootstrap prediction
    result = run_bootstrap_prediction(short_obs, fc_pred_t, config)

    actual_rho_T = np.corrcoef(fc_pred_t, fc_true_T)[0, 1]

    print("========== [Final Results (Empirical Data)] ==========")
    print(f"Subject Scan Length: {total_min:.1f}m (TR: {tr}s)")
    print(f"True Oracle ρ*(T):      {actual_rho_T:.4f}")
    if result.rho_hat_T > 0.8:
        print(
            f"Predicted ρ̂T:         {result.rho_hat_T:.4f}  "
            "✅ (>80% Target Goal Reached)"
        )
    else:
        print(
            f"Predicted ρ̂T:         {result.rho_hat_T:.4f}  "
            "⚠️ (Below 80% target)"
        )
    print(f"95% Confidence Interval: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print("======================================================")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    np.random.seed(42)
    run_empirical_pipeline()
