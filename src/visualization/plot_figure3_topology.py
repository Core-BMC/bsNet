"""Generate Figure 3: Network topology preservation analysis (10-seed design).

Validates that BS-NET predictions preserve small-worldness and degree
variance compared to reference FC connectivity graphs.

Design: 10 seeds × 20 subjects × 100 ROIs
  - Cross-seed variability captured via Seed column
  - Boxplots pool all (seed × subject) observations
  - Summary statistics (mean ± SD across seed-level means) reported
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.core.bootstrap import (
    block_bootstrap_indices,
    correct_attenuation,
    estimate_optimal_block_length,
    fisher_z,
    fisher_z_inv,
)
from src.core.config import BSNetConfig
from src.core.graph_metrics import (
    compute_degree_variance,
    compute_small_worldness,
    threshold_matrix,
)
from src.core.simulate import generate_synthetic_timeseries
from src.data.data_loader import get_fc_matrix
from src.visualization.style import PALETTE, apply_bsnet_theme, save_figure

warnings.filterwarnings("ignore")

# Reproducible 10-seed set (shared across all figures)
SEEDS: list[int] = [42, 123, 777, 2026, 9999, 314, 628, 1414, 2718, 3141]


def run_single_seed(
    seed: int,
    n_subjects: int = 20,
    n_rois: int = 100,
    t_samples_long: int = 450,
    t_samples_short: int = 60,
) -> list[dict]:
    """Run topology analysis for a single seed.

    Args:
        seed: Random seed for reproducibility.
        n_subjects: Number of subjects per seed.
        n_rois: Number of ROIs per subject.
        t_samples_long: Number of samples in the full (15-min) scan.
        t_samples_short: Number of samples in the short (2-min) scan.

    Returns:
        List of result dicts with keys: Seed, Subject, Model,
        Degree Variance, Small-worldness (Sigma).
    """
    np.random.seed(seed)
    results: list[dict] = []

    for sub in range(n_subjects):
        ts, _ = generate_synthetic_timeseries(t_samples_long, n_rois)
        ts = ts.T  # (samples, rois)

        # 1. Reference 15-min FC
        fc_true = get_fc_matrix(ts, vectorized=False, use_shrinkage=True)

        # 2. Raw 2-min FC
        ts_short = ts[:t_samples_short, :]
        fc_raw = get_fc_matrix(ts_short, vectorized=False, use_shrinkage=True)

        # 3. BS-NET Predicted FC
        true_overlap = np.corrcoef(fc_raw.flatten(), fc_true.flatten())[0, 1]
        block_size = estimate_optimal_block_length(ts_short)
        rho_hat_b = []
        n_split = ts_short.shape[0] // 2

        for _b in range(5):
            idx = block_bootstrap_indices(
                t_samples_short,
                block_size,
                n_blocks=t_samples_short // block_size,
            )
            ts_b = ts_short[idx, :]
            fc_b = get_fc_matrix(ts_b, vectorized=False, use_shrinkage=True)
            r_obs = np.corrcoef(fc_raw.flatten(), fc_b.flatten())[0, 1]
            r_split = np.corrcoef(
                get_fc_matrix(
                    ts_b[:n_split, :], vectorized=True, use_shrinkage=True
                ),
                get_fc_matrix(
                    ts_b[n_split:, :], vectorized=True, use_shrinkage=True
                ),
            )[0, 1]
            rho_est = correct_attenuation(
                r_obs, 0.98, r_split, k=t_samples_long / t_samples_short
            )
            rho_hat_b.append(rho_est)

        median_rho = np.nanmedian(rho_hat_b)
        inflation_ratio = median_rho / true_overlap
        fc_pred_z = fisher_z(fc_raw) * min(inflation_ratio, 1.5)
        fc_pred = fisher_z_inv(fc_pred_z)

        # Compute topology metrics
        for name, fc_mat in [
            ("Reference FC (15m)", fc_true),
            ("Raw FC (2m)", fc_raw),
            ("BS-NET (2m)", fc_pred),
        ]:
            adj = threshold_matrix(fc_mat, density=0.25)
            deg_var = compute_degree_variance(adj)
            sigma = compute_small_worldness(adj)
            results.append(
                {
                    "Seed": seed,
                    "Subject": sub,
                    "Model": name,
                    "Degree Variance": deg_var,
                    "Small-worldness (Sigma)": sigma,
                }
            )

    return results


def main() -> None:
    """Generate and save Figure 3: Network topology preservation.

    Runs 10 seeds x 20 subjects x 100 ROIs, then creates a two-panel
    visualization showing degree variance and small-worldness preservation
    with cross-seed variability.
    """
    print("--- Phase 4: Topology Tracking & Validation (10-seed design) ---")
    config = BSNetConfig()
    out_dir = Path(config.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(config.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for i, seed in enumerate(SEEDS):
        print(f"  Seed {i + 1}/{len(SEEDS)} (seed={seed})...")
        seed_results = run_single_seed(seed, n_subjects=20, n_rois=100)
        all_results.extend(seed_results)

    df = pd.DataFrame(all_results).dropna()
    df.to_csv(out_dir / "topology_results.csv", index=False)

    # Print cross-seed summary
    print("\n--- [Topology Preservation — Cross-seed Summary] ---")
    for model in ["Reference FC (15m)", "Raw FC (2m)", "BS-NET (2m)"]:
        seed_means = (
            df[df["Model"] == model]
            .groupby("Seed")["Small-worldness (Sigma)"]
            .mean()
        )
        print(
            f"  {model}: sigma = {seed_means.mean():.3f} +/- {seed_means.std():.3f} "
            f"(across {len(SEEDS)} seeds)"
        )

    # Plotting — violin + swarm + mean±SD style
    apply_bsnet_theme()
    models = ["Reference FC (15m)", "Raw FC (2m)", "BS-NET (2m)"]
    topo_palette = [PALETTE["true"], PALETTE["raw"], PALETTE["bsnet"]]
    swarm_palette = ["#1a5276", "#9C1C3D", "#2154a8"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Degree Variance
    sns.violinplot(
        data=df, x="Model", y="Degree Variance", order=models,
        palette=topo_palette, inner="box", linewidth=1, width=0.8, ax=ax1,
    )
    sns.swarmplot(
        data=df, x="Model", y="Degree Variance", order=models,
        palette=swarm_palette, alpha=0.6, size=2.5,
        edgecolor="white", linewidth=0.3, ax=ax1,
    )
    ax1.set_title(
        "A. Conservation of Network Hub Variance", fontweight="bold", fontsize=13,
    )
    ax1.set_ylabel("Variance of Node Degrees (Var(k))")
    ax1.set_xlabel("")

    # Panel B: Small-worldness
    sns.violinplot(
        data=df, x="Model", y="Small-worldness (Sigma)", order=models,
        palette=topo_palette, inner="box", linewidth=1, width=0.8, ax=ax2,
    )
    sns.swarmplot(
        data=df, x="Model", y="Small-worldness (Sigma)", order=models,
        palette=swarm_palette, alpha=0.6, size=2.5,
        edgecolor="white", linewidth=0.3, ax=ax2,
    )
    ax2.axhline(
        y=1.0, color=PALETTE["highlight"], linestyle="--", alpha=0.7,
        label="Random Graph Limit (sigma=1)",
    )
    ax2.set_title(
        "B. Preservation of Small-worldness (sigma)", fontweight="bold", fontsize=13,
    )
    ax2.set_ylabel("Small-worldness Index (sigma)")
    ax2.set_xlabel("")

    # Add cross-seed mean±SD diamond markers
    for ax, metric in [(ax1, "Degree Variance"), (ax2, "Small-worldness (Sigma)")]:
        for i, model in enumerate(models):
            seed_means = (
                df[df["Model"] == model].groupby("Seed")[metric].mean()
            )
            grand_mean = seed_means.mean()
            grand_sd = seed_means.std()
            ax.errorbar(
                i, grand_mean, yerr=grand_sd,
                fmt="D", color="red", markersize=7, capsize=6, capthick=1.5,
                zorder=10, label="Seed mean +/- SD" if i == 0 else None,
            )
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_figure(fig, "Figure3_Topology.png")
    print("\nSuccessfully exported topology profiles.")


if __name__ == "__main__":
    main()
