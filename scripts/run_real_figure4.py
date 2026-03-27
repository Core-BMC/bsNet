#!/usr/bin/env python3
"""Generate Figure 4 from real fMRI data (Schaefer 100 parcellation).

Reads preprocessed time series from data/preprocessed/ and runs:
  - BS-NET bootstrap prediction per subject
  - Community detection (Louvain) on thresholded FC
  - ARI and per-network Jaccard overlap
  - 10-seed bootstrap resampling for cross-seed variability

Prerequisites:
  python scripts/preprocess_bold.py --all

Usage:
  python scripts/run_real_figure4.py
"""

from __future__ import annotations

import logging
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
from src.core.graph_metrics import get_communities, threshold_matrix
from src.data.data_loader import get_fc_matrix
from src.visualization.style import MODEL_PALETTE, apply_bsnet_theme, save_figure

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
SEEDS: list[int] = [42, 123, 777, 2026, 9999, 314, 628, 1414, 2718, 3141]


def load_preprocessed_subjects() -> list[dict]:
    """Load all preprocessed time series files.

    Returns:
        List of dicts with 'sub_id', 'timeseries', 'n_volumes', 'tr'.
    """
    ts_files = sorted(PREPROCESSED_DIR.glob("sub-*_schaefer100_timeseries.npy"))
    if not ts_files:
        raise FileNotFoundError(
            f"No preprocessed files found in {PREPROCESSED_DIR}.\n"
            "Run: python scripts/preprocess_bold.py --all"
        )

    # Read log for TR info
    log_path = PREPROCESSED_DIR / "preprocessing_log.csv"
    tr_map: dict[str, float] = {}
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        for _, row in log_df.iterrows():
            if "tr" in row and pd.notna(row["tr"]):
                tr_map[row["subject"]] = float(row["tr"])

    subjects = []
    for f in ts_files:
        sub_id = f.stem.replace("_schaefer100_timeseries", "")
        ts = np.load(f)
        tr = tr_map.get(sub_id, 2.0)
        subjects.append({
            "sub_id": sub_id,
            "timeseries": ts,
            "n_volumes": ts.shape[0],
            "tr": tr,
        })

    logger.info(f"Loaded {len(subjects)} subjects")
    return subjects


def get_yeo7_assignments(n_rois: int = 100) -> list[set[int]]:
    """Get Yeo 7-network ROI assignments for Schaefer 100 parcellation.

    Schaefer 100 labels follow the pattern:
        7Networks_LH_Vis_1, 7Networks_LH_SomMot_1, ...
    Each label encodes which of the 7 Yeo networks it belongs to.

    This function maps ROI indices to their canonical network.

    Returns:
        List of 7 sets, each containing ROI indices for that network.
    """
    try:
        from nilearn.datasets import fetch_atlas_schaefer_2018
        atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
        labels = [l.decode() if isinstance(l, bytes) else str(l) for l in atlas.labels]
    except Exception:
        # Fallback: equal block partition (same as simulation)
        logger.warning("Cannot load Schaefer labels — using block partition fallback")
        block_size = n_rois // 7
        communities = []
        for i in range(7):
            start = i * block_size
            end = ((i + 1) * block_size) if i < 6 else n_rois
            communities.append(set(range(start, end)))
        return communities

    # Parse network names from Schaefer labels
    # Label format: "7Networks_LH_Vis_1" → network = "Vis"
    network_map = {
        "Vis": 0, "SomMot": 1, "DorsAttn": 2,
        "SalVentAttn": 3, "Limbic": 4, "Cont": 5, "Default": 6,
    }
    communities: list[set[int]] = [set() for _ in range(7)]

    for roi_idx, label in enumerate(labels):
        if label == "Background":
            continue
        # Parse: "7Networks_LH_Vis_1" → parts[2] = "Vis"
        parts = label.split("_")
        if len(parts) >= 3:
            net_name = parts[2]
            net_idx = network_map.get(net_name)
            if net_idx is not None:
                communities[net_idx].add(roi_idx)
            else:
                logger.warning(f"Unknown network '{net_name}' in label '{label}'")

    # Validate
    assigned = sum(len(c) for c in communities)
    logger.info(f"Yeo 7-network assignment: {assigned}/{n_rois} ROIs assigned")
    for i, name in enumerate(NETWORK_NAMES):
        logger.info(f"  {name}: {len(communities[i])} ROIs")

    return communities


def analyze_single_seed(
    subjects: list[dict],
    seed: int,
    config: BSNetConfig,
    true_communities: list[set[int]],
) -> tuple[list[dict], list[dict]]:
    """Run BS-NET + community detection for one bootstrap seed.

    For each subject:
      1. Split timeseries into reference (full) and short (2 min)
      2. Compute FC matrices
      3. Run BS-NET bootstrap prediction
      4. Detect communities via Louvain
      5. Compute ARI and per-network Jaccard

    Args:
        subjects: List of subject dicts from load_preprocessed_subjects().
        seed: Random seed for bootstrap resampling.
        config: BS-NET configuration.
        true_communities: Yeo 7-network ROI assignments.

    Returns:
        Tuple of (ari_results, jaccard_results).
    """
    np.random.seed(seed)
    ari_results: list[dict] = []
    jaccard_results: list[dict] = []

    for subj in subjects:
        ts = subj["timeseries"]  # (n_volumes, 100)
        tr = subj["tr"]
        n_vols = ts.shape[0]
        t_short = int(2.0 * 60 / tr)  # 2-minute equivalent

        if n_vols < t_short + 10:
            continue

        # Reference FC (full scan) and short FC (2 min)
        fc_ref = get_fc_matrix(ts, vectorized=False, use_shrinkage=True)
        ts_short = ts[:t_short, :]
        fc_raw = get_fc_matrix(ts_short, vectorized=False, use_shrinkage=True)

        # BS-NET prediction
        rho_hat_b = []
        n_split = t_short // 2
        n_bootstraps = 5
        for _ in range(n_bootstraps):
            idx = block_bootstrap_indices(t_short, 10, n_blocks=6)
            ts_b = ts_short[idx, :]
            fc_b = get_fc_matrix(ts_b, vectorized=False, use_shrinkage=True)
            r_obs = np.corrcoef(fc_raw.flatten(), fc_b.flatten())[0, 1]
            r_split = np.corrcoef(
                get_fc_matrix(ts_b[:n_split, :], vectorized=True, use_shrinkage=True),
                get_fc_matrix(ts_b[n_split:, :], vectorized=True, use_shrinkage=True),
            )[0, 1]
            rho_est = correct_attenuation(
                r_obs, 0.98, r_split, k=n_vols / t_short,
            )
            rho_hat_b.append(rho_est)

        true_overlap = np.corrcoef(fc_raw.flatten(), fc_ref.flatten())[0, 1]
        inflation_ratio = np.nanmedian(rho_hat_b) / max(true_overlap, 0.01)
        fc_pred_z = fisher_z(fc_raw) * min(inflation_ratio, 1.5)
        fc_pred = fisher_z_inv(fc_pred_z)

        # Community detection on reference
        adj_ref = threshold_matrix(fc_ref, density=config.fc_density)
        labels_ref = get_communities(adj_ref)

        for name, fc_mat in [("Raw FC (2m)", fc_raw), ("BS-NET (2m)", fc_pred)]:
            adj = threshold_matrix(fc_mat, density=config.fc_density)
            labels = get_communities(adj)
            ari = adjusted_rand_score(labels_ref, labels)

            ari_results.append({
                "Seed": seed,
                "Subject": subj["sub_id"],
                "Model": name,
                "ARI": ari,
            })

            # Per-network Jaccard overlap
            pred_comms: dict[int, set[int]] = {}
            for node, lbl in enumerate(labels):
                pred_comms.setdefault(lbl, set()).add(node)

            for net_idx, true_set in enumerate(true_communities):
                best_jaccard = 0.0
                for pred_set in pred_comms.values():
                    inter = len(true_set & pred_set)
                    union = len(true_set | pred_set)
                    if union > 0:
                        jaccard = inter / union
                        best_jaccard = max(best_jaccard, jaccard)

                jaccard_results.append({
                    "Seed": seed,
                    "Subject": subj["sub_id"],
                    "Model": name,
                    "Network": NETWORK_NAMES[net_idx],
                    "Jaccard Overlap": best_jaccard,
                })

    return ari_results, jaccard_results


def plot_figure4(df_ari: pd.DataFrame, df_net: pd.DataFrame) -> None:
    """Generate Figure 4 plots: ARI violin + Jaccard violin.

    Args:
        df_ari: DataFrame with columns [Seed, Subject, Model, ARI].
        df_net: DataFrame with columns [Seed, Subject, Model, Network, Jaccard Overlap].
    """
    apply_bsnet_theme()
    ari_models = ["Raw FC (2m)", "BS-NET (2m)"]
    swarm_pal = ["#9C1C3D", "#0A4D8D"]

    # --- Plot 1: Overall ARI ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.violinplot(
        data=df_ari, x="Model", y="ARI", hue="Model",
        palette=MODEL_PALETTE, inner="box", linewidth=1, width=0.7, ax=ax,
    )
    sns.swarmplot(
        data=df_ari, x="Model", y="ARI", hue="Model",
        palette=swarm_pal, dodge=False, alpha=0.7, size=3,
        edgecolor="white", linewidth=0.4, ax=ax, legend=False,
    )

    for i, model in enumerate(ari_models):
        seed_means = df_ari[df_ari["Model"] == model].groupby("Seed")["ARI"].mean()
        ax.errorbar(
            i, seed_means.mean(), yerr=seed_means.std(),
            fmt="D", color="red", markersize=7, capsize=6, capthick=1.5,
            zorder=10, label="Seed mean +/- SD" if i == 0 else None,
        )

    ax.set_title(
        "Global Overlap Continuity (Schaefer 100 Parcellation)",
        fontweight="bold", pad=15,
    )
    ax.set_ylabel("Adjusted Rand Index (ARI)", fontweight="bold")
    ax.set_xlabel("")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_figure(fig, "Figure4_Overall_ARI.png")

    # --- Plot 2: Per-network Jaccard ---
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

    networks = df_net["Network"].unique()
    for net_i, net in enumerate(networks):
        for m_i, model in enumerate(ari_models):
            seed_means = (
                df_net[(df_net["Network"] == net) & (df_net["Model"] == model)]
                .groupby("Seed")["Jaccard Overlap"]
                .mean()
            )
            offset = -0.2 + m_i * 0.4
            ax.errorbar(
                net_i + offset, seed_means.mean(), yerr=seed_means.std(),
                fmt="D", color="red", markersize=5, capsize=4, capthick=1.2,
                zorder=10,
                label="Seed mean +/- SD" if (net_i == 0 and m_i == 0) else None,
            )

    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.3, label="Perfect Structural Match")
    ax.set_title(
        "Network Jaccard Similarity (Schaefer 100 Parcellation)",
        fontweight="bold", pad=15,
    )
    ax.set_ylabel("Jaccard Similarity w/ Reference FC 15m Network", fontweight="bold")
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], loc="lower right", title="Modality")
    plt.tight_layout()
    save_figure(fig, "Figure4_Subnetworks.png")


def main() -> None:
    """Entry point: load data, run 10-seed analysis, generate Figure 4."""
    config = BSNetConfig(n_rois=100)
    out_dir = Path(config.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    subjects = load_preprocessed_subjects()
    print(f"\nLoaded {len(subjects)} preprocessed subjects")

    # Get Yeo 7-network assignments (real Schaefer labels)
    true_communities = get_yeo7_assignments(n_rois=100)

    # Run 10-seed analysis
    all_ari: list[dict] = []
    all_jaccard: list[dict] = []

    for i, seed in enumerate(SEEDS):
        print(f"  Seed {i + 1}/{len(SEEDS)} (seed={seed})...")
        ari, jaccard = analyze_single_seed(subjects, seed, config, true_communities)
        all_ari.extend(ari)
        all_jaccard.extend(jaccard)
        print(f"    → {len(ari)} ARI, {len(jaccard)} Jaccard records")

    df_ari = pd.DataFrame(all_ari)
    df_net = pd.DataFrame(all_jaccard)

    # Save CSVs
    df_ari.to_csv(out_dir / "subnetwork_ari_results.csv", index=False)
    df_net.to_csv(out_dir / "per_network_jaccard_results.csv", index=False)

    # Cross-seed summary
    print("\n--- [ARI — Cross-seed Summary] ---")
    for model in ["Raw FC (2m)", "BS-NET (2m)"]:
        seed_means = df_ari[df_ari["Model"] == model].groupby("Seed")["ARI"].mean()
        print(f"  {model}: ARI = {seed_means.mean():.3f} +/- {seed_means.std():.3f}")

    print("\n--- [Targeted Network Jaccard — Cross-seed Summary] ---")
    for net in ["Visual", "Default Mode"]:
        for model in ["Raw FC (2m)", "BS-NET (2m)"]:
            seed_means = (
                df_net[(df_net["Network"] == net) & (df_net["Model"] == model)]
                .groupby("Seed")["Jaccard Overlap"]
                .mean()
            )
            print(f"  {net} / {model}: {seed_means.mean():.3f} +/- {seed_means.std():.3f}")

    # Plot
    plot_figure4(df_ari, df_net)
    print("\nFigure 4 generated successfully.")


if __name__ == "__main__":
    main()
