"""
Failure case characterization for BS-NET pipeline.

Simulates N subjects with varying noise/SNR conditions through the real
pipeline path, identifies subjects with rho_hat_T < 0.80, and characterizes
what distinguishes failing subjects from passing ones.

Metrics analyzed:
  - Split-half reliability of short scan
  - Mean absolute FC strength
  - AR(1) autocorrelation structure
  - Number of near-zero-variance ROIs
  - SNR proxy (signal variance / noise variance)

Output: artifacts/reports/failure_analysis.csv
        artifacts/reports/failure_summary.txt
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.core.simulate import generate_synthetic_timeseries
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.80


def compute_subject_characteristics(
    short_obs: np.ndarray,
    long_obs: np.ndarray,
    long_signal: np.ndarray,
) -> dict[str, float]:
    """Compute diagnostic characteristics of a simulated subject.

    Args:
        short_obs: Short noisy observation (n_samples, n_rois).
        long_obs: Full noisy observation (n_samples, n_rois).
        long_signal: Full noise-free signal (n_samples, n_rois).

    Returns:
        Dict with diagnostic metrics.
    """
    n_samples, n_rois = short_obs.shape

    # Split-half reliability of short scan
    n_split = n_samples // 2
    fc_h1 = get_fc_matrix(short_obs[:n_split, :], vectorized=True, use_shrinkage=True)
    fc_h2 = get_fc_matrix(short_obs[n_split:, :], vectorized=True, use_shrinkage=True)
    split_half_r = float(np.corrcoef(fc_h1, fc_h2)[0, 1])

    # Mean absolute FC strength (short scan)
    fc_short = get_fc_matrix(short_obs, vectorized=True, use_shrinkage=True)
    mean_abs_fc = float(np.mean(np.abs(fc_short)))

    # Temporal SNR proxy: mean signal variance / mean noise variance
    noise = long_obs - long_signal
    signal_var = float(np.mean(np.var(long_signal[:n_samples, :], axis=0)))
    noise_var = float(np.mean(np.var(noise[:n_samples, :], axis=0)))
    snr = signal_var / noise_var if noise_var > 1e-10 else float("inf")

    # AR(1) coefficient (mean across ROIs)
    ar1_vals = []
    for i in range(min(n_rois, 50)):
        ts = short_obs[:, i]
        std_val = np.std(ts)
        if std_val > 1e-10:
            c = np.corrcoef(ts[:-1], ts[1:])[0, 1]
            if not np.isnan(c):
                ar1_vals.append(c)
    mean_ar1 = float(np.mean(ar1_vals)) if ar1_vals else 0.0

    # Near-zero-variance ROIs
    roi_vars = np.var(short_obs, axis=0)
    n_low_var_rois = int(np.sum(roi_vars < 1e-6))

    # Temporal variance (how "active" is the short scan)
    temporal_var = float(np.mean(np.var(short_obs, axis=1)))

    return {
        "split_half_r": split_half_r,
        "mean_abs_fc": mean_abs_fc,
        "snr": snr,
        "mean_ar1": mean_ar1,
        "n_low_var_rois": n_low_var_rois,
        "temporal_var": temporal_var,
        "signal_var": signal_var,
        "noise_var": noise_var,
    }


def run_failure_analysis(
    n_subjects: int = 100,
    n_rois: int = 50,
    short_samples: int = 120,
    target_samples: int = 900,
    noise_level: float = 0.25,
    ar1: float = 0.6,
    n_bootstraps: int = 50,
) -> list[dict]:
    """Simulate N subjects and characterize pass/fail outcomes.

    Args:
        n_subjects: Number of simulated subjects.
        n_rois: Number of ROIs.
        short_samples: Short scan duration in samples.
        target_samples: Target scan duration in samples.
        noise_level: Base noise level.
        ar1: AR(1) coefficient.
        n_bootstraps: Bootstrap iterations per subject.

    Returns:
        List of dicts with per-subject results and characteristics.
    """
    config = BSNetConfig(
        n_rois=n_rois,
        short_duration_sec=short_samples,
        target_duration_min=int(target_samples / 60),
        n_bootstraps=n_bootstraps,
        tr=1.0,
    )

    results = []

    print(f"\nSimulating {n_subjects} subjects...")
    print(f"Config: {n_rois} ROIs, {short_samples}s short, "
          f"{target_samples}s target, noise={noise_level}")
    print()

    for subj_id in range(n_subjects):
        seed = 1000 + subj_id
        np.random.seed(seed)

        # Generate subject-specific data
        long_obs, long_signal = generate_synthetic_timeseries(
            target_samples, n_rois, noise_level=noise_level, ar1=ar1
        )
        long_obs = long_obs.T
        long_signal = long_signal.T

        # Reference FC from full noisy observation
        fc_reference = get_fc_matrix(long_obs, vectorized=True, use_shrinkage=False)

        # Short observation
        short_obs = long_obs[:short_samples, :]

        # Run pipeline
        result = run_bootstrap_prediction(short_obs, fc_reference, config)

        # Subject characteristics
        chars = compute_subject_characteristics(short_obs, long_obs, long_signal)

        # rFC: correlation between short-scan FC and reference FC
        fc_short = get_fc_matrix(short_obs, vectorized=True, use_shrinkage=True)
        r_fc = float(np.corrcoef(fc_reference, fc_short)[0, 1])

        passed = result.rho_hat_T >= PASS_THRESHOLD
        status = "PASS" if passed else "FAIL"

        if subj_id % 20 == 0 or not passed:
            print(
                f"  [{subj_id + 1:3d}/{n_subjects}] "
                f"rho_hat_T={result.rho_hat_T:.4f}, "
                f"r_fc={r_fc:.4f}, "
                f"split_half={chars['split_half_r']:.3f}, "
                f"snr={chars['snr']:.2f}  {status}"
            )

        row = {
            "subject_id": subj_id,
            "seed": seed,
            "rho_hat_T": result.rho_hat_T,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "r_fc": r_fc,
            "pass": passed,
            **chars,
        }
        results.append(row)

    return results


def save_and_summarize(results: list[dict], artifacts_dir: Path) -> None:
    """Save results and print failure characterization summary.

    Args:
        results: List of per-subject result dicts.
        artifacts_dir: Directory for output files.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = artifacts_dir / "failure_analysis.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {csv_path}")

    # Summary analysis
    pass_subjects = [r for r in results if r["pass"]]
    fail_subjects = [r for r in results if not r["pass"]]
    n_total = len(results)
    n_fail = len(fail_subjects)
    fail_rate = 100 * n_fail / n_total

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("FAILURE ANALYSIS SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Total subjects: {n_total}")
    summary_lines.append(f"Pass (rho_hat_T >= {PASS_THRESHOLD}): {len(pass_subjects)}")
    summary_lines.append(f"Fail (rho_hat_T <  {PASS_THRESHOLD}): {n_fail} ({fail_rate:.1f}%)")
    summary_lines.append("")

    # Compare characteristics between pass and fail
    metrics = [
        ("rho_hat_T", "rho_hat_T"),
        ("r_fc", "r_fc (FC agreement)"),
        ("split_half_r", "Split-half reliability"),
        ("snr", "SNR (signal/noise var)"),
        ("mean_ar1", "Mean AR(1)"),
        ("mean_abs_fc", "Mean |FC|"),
        ("temporal_var", "Temporal variance"),
    ]

    summary_lines.append(f"{'Metric':<25} {'Pass (mean+-SD)':<20} {'Fail (mean+-SD)':<20} {'Effect'}")
    summary_lines.append("-" * 70)

    for key, label in metrics:
        pass_vals = [r[key] for r in pass_subjects]
        fail_vals = [r[key] for r in fail_subjects] if fail_subjects else [0]

        p_mean = np.mean(pass_vals)
        p_std = np.std(pass_vals)
        f_mean = np.mean(fail_vals)
        f_std = np.std(fail_vals)
        diff = f_mean - p_mean

        summary_lines.append(
            f"{label:<25} {p_mean:>7.4f} +- {p_std:<7.4f}  "
            f"{f_mean:>7.4f} +- {f_std:<7.4f}  {diff:+.4f}"
        )

    summary_lines.append("=" * 70)

    if fail_subjects:
        summary_lines.append("")
        summary_lines.append("FAILURE CHARACTERIZATION:")
        summary_lines.append(
            "Failing subjects tend to have:"
        )

        # Determine dominant failure characteristics
        pass_split = np.mean([r["split_half_r"] for r in pass_subjects])
        fail_split = np.mean([r["split_half_r"] for r in fail_subjects])
        pass_snr = np.mean([r["snr"] for r in pass_subjects])
        fail_snr = np.mean([r["snr"] for r in fail_subjects])

        if fail_split < pass_split:
            summary_lines.append(
                f"  - Lower split-half reliability ({fail_split:.3f} vs {pass_split:.3f})"
            )
        if fail_snr < pass_snr:
            summary_lines.append(
                f"  - Lower temporal SNR ({fail_snr:.2f} vs {pass_snr:.2f})"
            )
        summary_lines.append("")
        summary_lines.append(
            "Clinical implication: These subjects likely represent cases with "
            "high motion, low BOLD signal, or poor data quality — consistent "
            "with standard QC exclusion criteria in rsfMRI studies."
        )
    else:
        summary_lines.append("")
        summary_lines.append(
            "NO FAILURES DETECTED at current noise/ROI settings. "
            "All subjects pass the rho_hat_T >= 0.80 threshold."
        )
        summary_lines.append(
            "This suggests the 9% failure rate observed in prior experiments "
            "may be specific to higher noise conditions or different ROI counts."
        )

    summary_lines.append("=" * 70)

    # Print and save summary
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    txt_path = artifacts_dir / "failure_summary.txt"
    with open(txt_path, "w") as f:
        f.write(summary_text)
    print(f"\nSummary saved: {txt_path}")


def main() -> None:
    """Run failure characterization analysis."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    artifacts_dir = Path("artifacts/reports")
    results = run_failure_analysis(
        n_subjects=100,
        n_rois=50,
        short_samples=120,
        target_samples=900,
        noise_level=0.25,
        ar1=0.6,
        n_bootstraps=50,
    )
    save_and_summarize(results, artifacts_dir)


if __name__ == "__main__":
    main()
