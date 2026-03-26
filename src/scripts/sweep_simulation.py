import logging

import numpy as np
import pandas as pd

from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.core.simulate import generate_synthetic_timeseries
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)


def run_duration_sweep() -> None:
    """Run BS-NET duration sweep across multiple seeds for validation.

    Evaluates prediction performance across different observation durations
    and random seeds to assess reliability of the bootstrapping procedure.
    """
    # Canonical theoretical simulation parameters
    tr = 1.0
    t_minutes = 15
    T_samples = int(t_minutes * 60 / tr)
    n_rois = 50

    sweep_seconds = [30, 60, 90, 120, 150, 180, 210, 240]
    seeds = [42, 123, 777, 2026, 9999]

    print("--- BS-NET Time Duration Sweep (Multi-Seed Validation) ---")
    print(f"Full Target: {t_minutes} minutes ({T_samples} TRs, TR={tr}s)")
    print("Sweeping Durations:", sweep_seconds, "seconds")
    print("Evaluating Across Seeds:", seeds, "\n")

    all_results = []

    for seed in seeds:
        np.random.seed(seed)
        long_obs, long_signal = generate_synthetic_timeseries(
            n_samples=T_samples, n_rois=n_rois, noise_level=0.25, ar1=0.6
        )
        time_series = long_obs.T
        pure_signal = long_signal.T

        fc_true_T = get_fc_matrix(pure_signal, vectorized=True)
        fc_pred_t = fc_true_T + 0.1 * np.random.randn(*fc_true_T.shape)

        for t_sec in sweep_seconds:
            t_samples = int(t_sec / tr)
            short_obs = time_series[:t_samples, :]

            logger.info(
                f"Processing seed={seed}, duration={t_sec}s, samples={t_samples}"
            )

            # Create config for this duration
            config = BSNetConfig(
                n_rois=n_rois,
                tr=tr,
                short_duration_sec=t_sec,
                target_duration_min=t_minutes,
                n_bootstraps=100,
            )

            # Run bootstrap prediction
            result = run_bootstrap_prediction(short_obs, fc_pred_t, config)

            actual_rho_T = np.corrcoef(fc_pred_t, fc_true_T)[0, 1]
            error = abs(actual_rho_T - result.predicted_rho)

            all_results.append(
                {
                    "Seed": seed,
                    "Duration (s)": t_sec,
                    "Predicted": result.predicted_rho,
                    "Error": error,
                    "CI Lower": result.ci_lower,
                    "CI Upper": result.ci_upper,
                }
            )

    # Aggregate results over seeds
    df = pd.DataFrame(all_results)
    agg_df = df.groupby("Duration (s)").mean().reset_index()

    print(
        f"{'Dur(s)':<8} | {'Pred_Mean':<10} | {'Err_Mean':<10} | "
        f"{'CI Lower':<10} | {'CI Upper':<10}"
    )
    print("-" * 65)
    for _, row in agg_df.iterrows():
        print(
            f"{int(row['Duration (s)']):<8} | {row['Predicted']:.4f}     | "
            f"{row['Error']:.4f}     | {row['CI Lower']:.4f}     | "
            f"{row['CI Upper']:.4f}"
        )

    df.to_csv("artifacts/reports/duration_sweep_seeds_results.csv", index=False)
    agg_df.to_csv(
        "artifacts/reports/duration_sweep_seeds_aggregated.csv", index=False
    )
    print("\nSweep Complete! Saved detailed and aggregated CSVs to artifacts/reports/")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_duration_sweep()
