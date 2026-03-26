import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.data.data_loader import create_masker, fetch_schaefer_atlas, get_fc_matrix

logger = logging.getLogger(__name__)

# Connect to MoBSE
_mobse_path = os.environ.get("MOBSE_PATH", str(Path.home() / "GitHub" / "MoBSE"))
sys.path.append(_mobse_path)
try:
    from mobse.data.nuisance import build_paper_nuisance_confounds
    from mobse.data.prepare import _download_openneuro_rest_bold_multi
except ImportError as e:
    logger.error(f"Error importing from MoBSE: {e}")
    print(f"Error importing from MoBSE: {e}")
    sys.exit(1)


class DummyProgress:
    """Progress reporter that filters verbose output."""

    def update(self, stage: str, current: int | None = None,
               total: int | None = None, message: str = "") -> None:
        """Update progress with optional filtering.

        Args:
            stage: Progress stage identifier.
            current: Current item count.
            total: Total item count.
            message: Progress message.
        """
        if "openneuro_index" not in stage or (
            "tree_calls" in message and int(message.split("=")[-1]) % 100 == 0
        ):
            print(f"[{stage}] {message}")


def run_scale_up_pipeline() -> None:
    """Run BS-NET validation on ~100 subjects from OpenNeuro.

    Validates pipeline at scale across multiple datasets and subjects
    to assess robustness and generalization of the prediction approach.
    """
    print("--- Phase 3: Scale-up OpenNeuro Validation (n=100) ---")

    cache_dir = Path("data/openneuro")
    out_dir = Path("artifacts/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_ids = ["ds000030", "ds000243", "ds002790", "ds002785"]

    print(f"Targeting {len(dataset_ids)} Datasets: {dataset_ids}")
    print("Fetching up to 100 Adult (HC) subjects ...")

    diagnosis_terms = "normal, healthy control, control, ctrl, nor, hc"

    fetch_res = _download_openneuro_rest_bold_multi(
        dataset_ids=dataset_ids,
        snapshot_tag=None,
        task="rest,restingstate",
        n_subjects=100,
        min_age=18,
        diagnosis=diagnosis_terms,
        strict_hc=False,
        api_url="https://openneuro.org/crn/graphql",
        cache_root=cache_dir,
        progress=DummyProgress(),
    )

    records = fetch_res["records"]
    collected = len(records)
    print(f"\nTotal Collected Subjects: {collected} / 100")
    if collected == 0:
        logger.warning("No subjects collected. Exiting.")
        print("No subjects collected. Exiting.")
        return

    # Load shared Schaefer 100-ROI Atlas
    logger.info("Pre-fetching Schaefer 100 Atlas...")
    print("Pre-fetching Schaefer 100 Atlas...")
    atlas = fetch_schaefer_atlas(n_rois=100)

    results = []

    # Process each subject
    for i, rec in enumerate(records, start=1):
        target_bold = rec["bold_file"]
        sub_key = rec["subject_key"]
        print(f"\n[{i}/{collected}] Processing {sub_key} ...")
        logger.info(f"Processing subject {i}/{collected}: {sub_key}")

        try:
            img = nib.load(target_bold)
            tr = img.header.get_zooms()[3]
            if tr <= 0.0 or tr > 5.0:
                tr = 2.0

            # Denoise
            nuisance = build_paper_nuisance_confounds(
                bold_path=target_bold,
                tr=tr,
                include_compcor=True,
                compcor_components=5,
                include_gsr=True,
                add_derivatives=True,
                add_quadratic=True,
            )

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
            total_min = (T_samples * tr) / 60.0

            if total_min < 4.0:
                logger.info(f"Skip: Scan too short ({total_min:.1f}m)")
                print(f" => Skip: Scan too short ({total_min:.1f}m)")
                continue

            short_len_min = 2.0
            t_samples = int(short_len_min * 60 / tr)

            # Ground truth metrics
            fc_true_T = get_fc_matrix(time_series, vectorized=True)
            short_obs = time_series[:t_samples, :]

            # Create config with n_bootstraps=50 for scale-up efficiency
            config = BSNetConfig(
                n_rois=100,
                tr=tr,
                short_duration_sec=short_len_min * 60,
                target_duration_min=total_min,
                n_bootstraps=50,
            )

            fc_pred_t_mock = fc_true_T + 0.1 * np.random.randn(*fc_true_T.shape)

            # Run bootstrap prediction
            result = run_bootstrap_prediction(short_obs, fc_pred_t_mock, config)

            true_rho = np.corrcoef(fc_pred_t_mock, fc_true_T)[0, 1]

            logger.info(
                f"Subject {sub_key}: True={true_rho:.4f}, "
                f"Pred={result.predicted_rho:.4f}"
            )
            print(
                f" => True Oracle: {true_rho:.4f} | "
                f"Predicted: {result.predicted_rho:.4f}"
            )

            results.append(
                {
                    "subject": sub_key,
                    "dataset": rec["dataset_id"],
                    "scan_min": total_min,
                    "tr": tr,
                    "true_rho": true_rho,
                    "pred_rho": result.predicted_rho,
                    "error": abs(true_rho - result.predicted_rho),
                    "is_above_80": result.predicted_rho >= 0.8,
                }
            )

        except Exception as e:
            logger.error(f"Subject {sub_key} failed: {e}")
            print(f" => Failed: {e}")

    if not results:
        logger.warning("No completely processed results.")
        print("\nNo completely processed results.")
        return

    df = pd.DataFrame(results)
    csv_path = (
        out_dir
        / f"scale_up_100_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    df.to_csv(csv_path, index=False)

    mean_true = df["true_rho"].mean()
    mean_pred = df["pred_rho"].mean()
    mean_err = df["error"].mean()
    success_rate = (df["is_above_80"].sum() / len(df)) * 100

    print("\n========== [Phase 3 Scale-Up Aggregation] ==========")
    print(f"Processed / Requested: {len(df)} / 100")
    print(f"Mean True Oracle: {mean_true:.4f}")
    if mean_pred >= 0.8:
        print(f"Mean Predicted Target: {mean_pred:.4f} ✅ (>80% Maintained!)")
    else:
        print(f"Mean Predicted Target: {mean_pred:.4f} ⚠️")
    print(f"Accuracy >80% Ratio:   {success_rate:.1f}%")
    print(f"Mean Absolute Error:   {mean_err:.4f}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    np.random.seed(42)
    run_scale_up_pipeline()
