"""Generate Figure 3: Network topology preservation analysis.

Validates that BS-NET predictions preserve small-worldness and degree
variance compared to ground truth connectivity graphs.
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


def main() -> None:
    """Generate and save Figure 3: Network topology preservation.

    Creates a two-panel visualization showing:
    - Panel A: Degree variance conservation across models
    - Panel B: Small-worldness preservation

    Returns:
        None
    """
    print("--- Phase 4: Topology Tracking & Validation ---")
    config = BSNetConfig()
    out_dir = Path(config.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(config.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = 100
    n_rois = 400
    t_samples_long = 450
    t_samples_short = 60

    results = []

    print(f"Processing Graphical Networks for {n_subjects} Subjects...")
    for sub in range(n_subjects):
        if sub % 5 == 0:
            print(f" > Tracking Topology: Subject {sub}/{n_subjects}...")
        ts, _ = generate_synthetic_timeseries(t_samples_long, n_rois)
        ts = ts.T  # (samples, rois)

        # 1. True 15-min FC
        fc_true = get_fc_matrix(ts, vectorized=False, use_shrinkage=True)

        # 2. Raw 2-min FC
        ts_short = ts[:t_samples_short, :]
        fc_raw = get_fc_matrix(ts_short, vectorized=False, use_shrinkage=True)

        # 3. BS-NET Predicted 2-min FC
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

        # Compute topology
        for name, fc_mat in [
            ("True FC (15m)", fc_true),
            ("Raw FC (2m)", fc_raw),
            ("BS-NET (2m)", fc_pred),
        ]:
            adj = threshold_matrix(fc_mat, density=0.25)
            deg_var = compute_degree_variance(adj)
            sigma = compute_small_worldness(adj)
            results.append(
                {
                    "Subject": sub,
                    "Model": name,
                    "Degree Variance": deg_var,
                    "Small-worldness (Sigma)": sigma,
                }
            )

    df = pd.DataFrame(results).dropna()
    df.to_csv(out_dir / "topology_results.csv", index=False)

    print("\n--- [Topology Preservation Check] ---")
    true_sig = df[df["Model"] == "True FC (15m)"][
        "Small-worldness (Sigma)"
    ].mean()
    raw_sig = df[df["Model"] == "Raw FC (2m)"]["Small-worldness (Sigma)"].mean()
    pred_sig = df[df["Model"] == "BS-NET (2m)"][
        "Small-worldness (Sigma)"
    ].mean()
    print(f"Average Sigma σ -> True: {true_sig:.3f} | Raw: {raw_sig:.3f} "
          f"| Pred: {pred_sig:.3f}")

    # Plotting
    apply_bsnet_theme()
    fig = plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(
        data=df, x="Model", y="Degree Variance", showfliers=False, width=0.5
    )
    sns.stripplot(
        data=df,
        x="Model",
        y="Degree Variance",
        color="black",
        alpha=0.4,
        jitter=True,
    )
    plt.title("Conservation of Network Hub Variance", fontweight="bold")
    plt.ylabel("Variance of Node Degrees (Var(k))")

    plt.subplot(1, 2, 2)
    sns.boxplot(
        data=df,
        x="Model",
        y="Small-worldness (Sigma)",
        showfliers=False,
        width=0.5,
    )
    sns.stripplot(
        data=df,
        x="Model",
        y="Small-worldness (Sigma)",
        color="black",
        alpha=0.4,
        jitter=True,
    )
    plt.axhline(
        y=1.0,
        color=PALETTE["highlight"],
        linestyle="--",
        alpha=0.7,
        label="Random Graph Limit (σ=1)",
    )
    plt.title("Preservation of Small-worldness (σ)", fontweight="bold")
    plt.ylabel("Small-worldness Index (σ)")
    plt.legend(loc="lower right")

    plt.tight_layout()
    save_figure(fig, "Figure3_Topology.png")
    print("\nSuccessfully exported topology profiles.")


if __name__ == "__main__":
    np.random.seed(42)
    main()
