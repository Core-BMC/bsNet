"""Generate Figure 4: Network-specific integrity and community detection (10-seed).

Analyzes Jaccard overlap between predicted and true community structures
across individual networks, with global modularity continuity metrics.

Design: 10 seeds x 20 subjects x 100 ROIs
  - Cross-seed variability captured via Seed column
  - Boxplots pool all (seed x subject) observations
  - Summary statistics (mean +/- SD across seed-level means) reported
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

# Reproducible 10-seed set (shared across all figures)
SEEDS: list[int] = [42, 123, 777, 2026, 9999, 314, 628, 1414, 2718, 3141]


def run_single_seed(
    seed: int,
    config: BSNetConfig,
    n_subjects: int = 20,
    n_rois: int = 100,
    t_samples_long: int = 450,
    t_samples_short: int = 60,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run network integrity analysis for a single seed.

    Args:
        seed: Random seed for reproducibility.
        config: BS-NET configuration.
        n_subjects: Number of subjects per seed.
        n_rois: Number of ROIs per subject.
        t_samples_long: Full-scan sample count.
        t_samples_short: Short-scan sample count.

    Returns:
        Tuple of (community_results, ari_results, jaccard_results).
    """
    np.random.seed(seed)
    community_results: list[dict] = []
    ari_results: list[dict] = []
    jaccard_results: list[dict] = []

    true_communities = compute_network_block_assignments(n_rois, config.n_networks)

    for sub in range(n_subjects):
        ts, _ = generate_synthetic_timeseries(t_samples_long, n_rois)
        ts = ts.T  # (samples, rois)

        fc_true = get_fc_matrix(ts, vectorized=False, use_shrinkage=True)
        ts_short = ts[:t_samples_short, :]
        fc_raw = get_fc_matrix(ts_short, vectorized=False, use_shrinkage=True)

        true_overlap = np.corrcoef(fc_raw.flatten(), fc_true.flatten())[0, 1]

        # BS-NET prediction
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

        # Community detection
        adj_true = threshold_matrix(fc_true, density=config.fc_density)
        labels_true = get_communities(adj_true)

        for name, fc_mat in [("Raw FC (2m)", fc_raw), ("BS-NET (2m)", fc_pred)]:
            adj = threshold_matrix(fc_mat, density=config.fc_density)
            labels = get_communities(adj)
            ari = adjusted_rand_score(labels_true, labels)

            community_results.append(
                {
                    "Seed": seed,
                    "Subject": sub,
                    "Model": name,
                    "Community Labels": labels,
                }
            )
            ari_results.append(
                {
                    "Seed": seed,
                    "Subject": sub,
                    "Model": name,
                    "ARI": ari,
                }
            )

            # Per-network Jaccard overlap
            pred_comms: dict[int, set[int]] = {}
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
                        "Seed": seed,
                        "Subject": sub,
                        "Model": name,
                        "Network": NETWORK_NAMES[net_idx],
                        "Jaccard Overlap": best_jaccard,
                    }
                )

    return community_results, ari_results, jaccard_results


def main() -> None:
    """Generate and save Figure 4: Network integrity analysis.

    Runs 10 seeds x 20 subjects x 100 ROIs, creates:
    - ARI boxplot (global modularity continuity)
    - Per-network Jaccard similarity boxplot
    Both include cross-seed error markers.
    """
    print("--- Phase 5: Network-Specific Integrity (10-seed design) ---")
    config = BSNetConfig()
    out_dir = Path(config.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(config.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_ari: list[dict] = []
    all_jaccard: list[dict] = []

    for i, seed in enumerate(SEEDS):
        print(f"  Seed {i + 1}/{len(SEEDS)} (seed={seed})...")
        _, ari_results, jaccard_results = run_single_seed(
            seed, config, n_subjects=20, n_rois=100,
        )
        all_ari.extend(ari_results)
        all_jaccard.extend(jaccard_results)

    df_ari = pd.DataFrame(all_ari)
    df_net = pd.DataFrame(all_jaccard)

    df_ari.to_csv(out_dir / "subnetwork_ari_results.csv", index=False)
    df_net.to_csv(out_dir / "per_network_jaccard_results.csv", index=False)

    # Print cross-seed summary
    print("\n--- [ARI — Cross-seed Summary] ---")
    for model in ["Raw FC (2m)", "BS-NET (2m)"]:
        seed_means = (
            df_ari[df_ari["Model"] == model].groupby("Seed")["ARI"].mean()
        )
        print(
            f"  {model}: ARI = {seed_means.mean():.3f} +/- {seed_means.std():.3f}"
        )

    print("\n--- [Targeted Network Jaccard — Cross-seed Summary] ---")
    for net in ["Visual", "Default Mode"]:
        for model in ["Raw FC (2m)", "BS-NET (2m)"]:
            seed_means = (
                df_net[
                    (df_net["Network"] == net) & (df_net["Model"] == model)
                ]
                .groupby("Seed")["Jaccard Overlap"]
                .mean()
            )
            print(
                f"  {net} / {model}: "
                f"{seed_means.mean():.3f} +/- {seed_means.std():.3f}"
            )

    # --- Plot 1: Global ARI (violin + swarm + mean±SD) ---
    apply_bsnet_theme()
    ari_models = ["Raw FC (2m)", "BS-NET (2m)"]
    swarm_pal = ["#9C1C3D", "#0A4D8D"]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.violinplot(
        data=df_ari, x="Model", y="ARI", order=ari_models,
        palette=MODEL_PALETTE, inner="box", linewidth=1, width=0.7, ax=ax,
    )
    sns.swarmplot(
        data=df_ari, x="Model", y="ARI", order=ari_models,
        palette=swarm_pal, alpha=0.7, size=3,
        edgecolor="white", linewidth=0.4, ax=ax,
    )

    # Cross-seed mean±SD diamond markers
    for i, model in enumerate(ari_models):
        seed_means = (
            df_ari[df_ari["Model"] == model].groupby("Seed")["ARI"].mean()
        )
        ax.errorbar(
            i, seed_means.mean(), yerr=seed_means.std(),
            fmt="D", color="red", markersize=7, capsize=6, capthick=1.5,
            zorder=10, label="Seed mean +/- SD" if i == 0 else None,
        )

    ax.set_title(
        "Global Overlap Continuity (100-ROI Parcellation)",
        fontweight="bold", pad=15,
    )
    ax.set_ylabel("Adjusted Rand Index (ARI)", fontweight="bold")
    ax.set_xlabel("")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_figure(fig, "Figure4_Overall_ARI.png")

    # --- Plot 2: Per-network Jaccard (violin + swarm + mean±SD) ---
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111)
    sns.violinplot(
        data=df_net, x="Network", y="Jaccard Overlap", hue="Model",
        palette=MODEL_PALETTE, inner="box", linewidth=1, width=0.8, ax=ax,
    )
    sns.swarmplot(
        data=df_net, x="Network", y="Jaccard Overlap", hue="Model",
        palette=swarm_pal, dodge=True, alpha=0.7, size=2.5,
        edgecolor="white", linewidth=0.3, ax=ax, legend=False,
    )

    # Per-network mean±SD overlay
    networks = df_net["Network"].unique()
    for net_i, net in enumerate(networks):
        for m_i, model in enumerate(ari_models):
            seed_means = (
                df_net[(df_net["Network"] == net) & (df_net["Model"] == model)]
                .groupby("Seed")["Jaccard Overlap"]
                .mean()
            )
            offset = -0.2 + m_i * 0.4  # dodge offset
            ax.errorbar(
                net_i + offset, seed_means.mean(), yerr=seed_means.std(),
                fmt="D", color="red", markersize=5, capsize=4, capthick=1.2,
                zorder=10,
                label="Seed mean +/- SD" if (net_i == 0 and m_i == 0) else None,
            )

    ax.axhline(
        y=1.0, color="r", linestyle="--", alpha=0.3,
        label="Perfect Structural Match",
    )
    ax.set_title(
        "Network Jaccard Similarity (100-ROI Parcellation)",
        fontweight="bold", pad=15,
    )
    ax.set_ylabel("Jaccard Similarity w/ Reference FC 15m Network", fontweight="bold")
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], loc="lower right", title="Modality")

    plt.tight_layout()
    save_figure(fig, "Figure4_Subnetworks.png")
    print("\nSuccessfully exported isolated network analysis.")


if __name__ == "__main__":
    main()
