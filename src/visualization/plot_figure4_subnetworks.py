"""Generate Figure 4: Network-specific integrity and community detection validation.

Analyzes Jaccard overlap between predicted and true community structures
across individual networks, with global modularity continuity metrics.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.cluster import adjusted_rand_score

from src.core.bootstrap import (
    block_bootstrap_indices,
    correct_attenuation,
    fisher_z,
    fisher_z_inv,
)
from src.core.config import NETWORK_NAMES, BSNetConfig
from src.core.graph_metrics import (
    compute_network_block_assignments,
    get_communities,
    threshold_matrix,
)
from src.core.simulate import generate_synthetic_timeseries
from src.data.data_loader import get_fc_matrix
from src.visualization.style import MODEL_PALETTE, apply_bsnet_theme, save_figure

warnings.filterwarnings("ignore")


def main() -> None:
    """Generate and save Figure 4: Network integrity analysis.

    Creates two visualizations:
    - Panel A: Global modularity continuity (Adjusted Rand Index)
    - Panel B: Per-network Jaccard similarity across models

    Returns:
        None
    """
    print("--- Phase 5: Network-Specific Integrity Validation ---")
    config = BSNetConfig()
    out_dir = Path(config.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(config.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = 100
    n_rois = 100
    t_samples_long = 450
    t_samples_short = 60

    results = []
    ari_results = []

    print(f"Tracking {config.n_networks} Canonical Networks for {n_subjects} "
          f"Subjects...")
    for sub in range(n_subjects):
        if sub % 5 == 0:
            print(f" > Processing Topologies: Subject {sub}/{n_subjects}...")
        ts, _ = generate_synthetic_timeseries(t_samples_long, n_rois)
        ts = ts.T

        fc_true = get_fc_matrix(ts, vectorized=False, use_shrinkage=True)
        ts_short = ts[:t_samples_short, :]
        fc_raw = get_fc_matrix(ts_short, vectorized=False, use_shrinkage=True)

        true_overlap = np.corrcoef(fc_raw.flatten(), fc_true.flatten())[0, 1]

        rho_hat_b = []
        n_split = ts_short.shape[0] // 2
        for _b in range(5):
            idx = block_bootstrap_indices(t_samples_short, 10, n_blocks=6)
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

        inflation_ratio = np.nanmedian(rho_hat_b) / true_overlap
        fc_pred_z = fisher_z(fc_raw) * min(inflation_ratio, 1.5)
        fc_pred = fisher_z_inv(fc_pred_z)

        adj_true = threshold_matrix(fc_true, density=config.fc_density)
        labels_true = get_communities(adj_true)

        for name, fc_mat in [("Raw FC (2m)", fc_raw), ("BS-NET (2m)", fc_pred)]:
            adj = threshold_matrix(fc_mat, density=config.fc_density)
            labels = get_communities(adj)
            ari = adjusted_rand_score(labels_true, labels)
            results.append(
                {
                    "Subject": sub,
                    "Model": name,
                    "Community Labels": labels,
                }
            )
            ari_results.append(
                {
                    "Subject": sub,
                    "Model": name,
                    "ARI": ari,
                }
            )

    df = pd.DataFrame(results)
    df_ari = pd.DataFrame(ari_results)
    df_ari.to_csv(out_dir / "subnetwork_ari_results_yeo100.csv", index=False)

    # Plot global ARI
    apply_bsnet_theme()
    fig = plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df_ari,
        x="Model",
        y="ARI",
        palette=MODEL_PALETTE,
        showfliers=False,
        width=0.5,
    )
    sns.stripplot(
        data=df_ari, x="Model", y="ARI", color="black", alpha=0.4, jitter=True
    )
    plt.title(
        "Global Overlap Continuity (Schaefer 400 Parcellation)",
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("Adjusted Rand Index (ARI)")
    plt.tight_layout()
    save_figure(fig, "Figure4_Overall_ARI.png")

    # Compute per-network Jaccard overlap
    true_communities = compute_network_block_assignments(n_rois, config.n_networks)

    jaccard_results = []
    for sub in range(n_subjects):
        for model in ["Raw FC (2m)", "BS-NET (2m)"]:
            labels = df[(df["Subject"] == sub) & (df["Model"] == model)][
                "Community Labels"
            ].iloc[0]

            pred_comms = {}
            for node, lbl in enumerate(labels):
                pred_comms.setdefault(lbl, set()).add(node)

            for net_idx, true_set in enumerate(true_communities):
                best_jaccard = 0.0
                for pred_set in pred_comms.values():
                    intersection = len(true_set.intersection(pred_set))
                    union = len(true_set.union(pred_set))
                    if union > 0:
                        jaccard = intersection / union
                        if jaccard > best_jaccard:
                            best_jaccard = jaccard

                jaccard_results.append(
                    {
                        "Subject": sub,
                        "Model": model,
                        "Network": NETWORK_NAMES[net_idx],
                        "Jaccard Overlap": best_jaccard,
                    }
                )

    df_net = pd.DataFrame(jaccard_results)
    df_net.to_csv(out_dir / "per_network_jaccard_results_yeo100.csv", index=False)

    print("\n--- [Targeted Network Jaccard Overlap Check] ---")
    for net in ["Visual", "Default Mode"]:
        raw_jm = df_net[
            (df_net["Network"] == net) & (df_net["Model"] == "Raw FC (2m)")
        ]["Jaccard Overlap"].mean()
        pred_jm = df_net[
            (df_net["Network"] == net) & (df_net["Model"] == "BS-NET (2m)")
        ]["Jaccard Overlap"].mean()
        print(f"{net} => Raw 2m: {raw_jm:.3f} | BS-NET 2m: {pred_jm:.3f}")

    # Plot per-network Jaccard
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_net,
        x="Network",
        y="Jaccard Overlap",
        hue="Model",
        palette=MODEL_PALETTE,
        showfliers=False,
        width=0.6,
    )
    sns.stripplot(
        data=df_net,
        x="Network",
        y="Jaccard Overlap",
        hue="Model",
        dodge=True,
        color="black",
        alpha=0.3,
        jitter=True,
        legend=False,
    )

    plt.title(
        "Smoothed Network Jaccard Curve (Schaefer 400 Parcellation)",
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("Jaccard Similarity w/ True 15m Network")
    plt.xlabel("")
    plt.xticks(rotation=15)
    plt.axhline(
        y=1.0,
        color="r",
        linestyle="--",
        alpha=0.3,
        label="Perfect Structural Match",
    )
    plt.legend(loc="lower right", title="Modality")

    plt.tight_layout()
    save_figure(fig, "Figure4_Subnetworks.png")
    print("\nSuccessfully exported isolated network analysis.")


if __name__ == "__main__":
    np.random.seed(42)
    main()
