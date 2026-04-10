"""Generate Figure 4: Network Structure Preservation (merged topology + community).

Validates that BS-NET predictions preserve network topology (hub variance,
small-worldness) and community structure (ARI, per-network Jaccard) compared
to reference 15-min FC.

Design: 10 seeds × 20 subjects × 100 ROIs
  - Single simulation pass computes both topology and community metrics
  - Panels A–C: global metrics (ARI, hub variance, small-worldness)
  - Panel D: network-level Jaccard similarity (7 Yeo networks)
  - Panel E: sliding-window temporal stability metrics
  - Cross-seed variability shown via red diamond mean±SD markers

Layout (GridSpec):
  ┌───────────┬───────────┬───────────┐
  │ A. ARI    │ B. Hub    │ C. Small- │
  │           │  Variance │  worldness│
  ├───────────┬───────────┴───────────┤
  │ D. Per-   │ E. Sliding-window     │
  │ network   │ temporal stability    │
  │ Jaccard   │                       │
  └───────────┴───────────────────────┘
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import sys

# Allow direct script execution:
#   python3 src/visualization/plot_figure4_structure.py
if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.cluster import adjusted_rand_score

from src.core.bootstrap import (
    block_bootstrap_indices,
    correct_attenuation,
    estimate_optimal_block_length,
    fisher_z,
    fisher_z_inv,
)
from src.core.config import NETWORK_NAMES, BSNetConfig
from src.core.graph_metrics import (
    compute_degree_variance,
    compute_network_block_assignments,
    compute_small_worldness,
    get_communities,
    threshold_matrix,
)
from src.core.simulate import generate_synthetic_timeseries
from src.data.data_loader import get_fc_matrix
from src.visualization.style import (
    FONT,
    PALETTE,
    apply_bsnet_theme,
    save_figure,
)

warnings.filterwarnings("ignore")

# Reproducible 10-seed set (shared across all figures)
SEEDS: list[int] = [42, 123, 777, 2026, 9999, 314, 628, 1414, 2718, 3141]

# Three-model order and palette (reference, raw, BS-NET)
# Fig 7 color schema: gray (Reference) + amber (Raw) + blue (BS-NET)
THREE_MODELS: list[str] = ["Reference FC (15m)", "Raw FC (2m)", "BS-NET (2m)"]
THREE_PALETTE: list[str] = ["#95a5a6", "#fdae61", "#4A90E2"]
#   Reference: silver-gray (neutral baseline)
#   Raw FC:    amber (matches Fig7 CC400 tone)
#   BS-NET:    blue  (matches Fig7 CC200 tone, hero)

# Two-model order (raw, BS-NET) — for ARI and Jaccard
TWO_MODELS: list[str] = ["Raw FC (2m)", "BS-NET (2m)"]
TWO_PALETTE: list[str] = ["#fdae61", "#4A90E2"]



def run_single_seed(
    seed: int,
    config: BSNetConfig,
    n_subjects: int = 20,
    n_rois: int = 100,
    t_samples_long: int = 450,
    t_samples_short: int = 60,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Run combined topology + community analysis for a single seed.

    Args:
        seed: Random seed for reproducibility.
        config: BS-NET configuration.
        n_subjects: Number of subjects per seed.
        n_rois: Number of ROIs per subject.
        t_samples_long: Number of samples in the full (15-min) scan.
        t_samples_short: Number of samples in the short (2-min) scan.

    Returns:
        Tuple of (topology_results, ari_results, jaccard_results, sliding_results).
    """
    np.random.seed(seed)
    topology_results: list[dict] = []
    ari_results: list[dict] = []
    jaccard_results: list[dict] = []
    sliding_results: list[dict] = []

    true_communities = compute_network_block_assignments(n_rois, config.n_networks)

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
        rho_hat_b: list[float] = []
        n_split = ts_short.shape[0] // 2

        for _b in range(5):
            idx = block_bootstrap_indices(
                t_samples_short,
                block_size,
                n_blocks=t_samples_short // max(block_size, 1),
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
                r_obs, 0.98, r_split, k=t_samples_long / t_samples_short,
            )
            rho_hat_b.append(rho_est)

        inflation_ratio = np.nanmedian(rho_hat_b) / true_overlap
        fc_pred_z = fisher_z(fc_raw) * min(inflation_ratio, 1.5)
        fc_pred = fisher_z_inv(fc_pred_z)

        # --- Topology metrics (3 models) ---
        for name, fc_mat in [
            ("Reference FC (15m)", fc_true),
            ("Raw FC (2m)", fc_raw),
            ("BS-NET (2m)", fc_pred),
        ]:
            adj = threshold_matrix(fc_mat, density=0.25)
            deg_var = compute_degree_variance(adj)
            sigma = compute_small_worldness(adj)
            topology_results.append({
                "Seed": seed, "Subject": sub, "Model": name,
                "Degree Variance": deg_var,
                "Small-worldness": sigma,
            })

        # --- Community metrics (2 models: Raw, BS-NET vs Reference) ---
        adj_true = threshold_matrix(fc_true, density=config.fc_density)
        labels_true = get_communities(adj_true)

        for name, fc_mat in [("Raw FC (2m)", fc_raw), ("BS-NET (2m)", fc_pred)]:
            adj = threshold_matrix(fc_mat, density=config.fc_density)
            labels = get_communities(adj)
            ari = adjusted_rand_score(labels_true, labels)
            ari_results.append({
                "Seed": seed, "Subject": sub, "Model": name, "ARI": ari,
            })

            # Per-network Jaccard overlap
            pred_comms: dict[int, set[int]] = {}
            for node, lbl in enumerate(labels):
                pred_comms.setdefault(lbl, set()).add(node)

            for net_idx, true_set in enumerate(true_communities):
                best_jaccard = 0.0
                for pred_set in pred_comms.values():
                    inter = len(true_set.intersection(pred_set))
                    union = len(true_set.union(pred_set))
                    if union > 0:
                        best_jaccard = max(best_jaccard, inter / union)
                jaccard_results.append({
                    "Seed": seed, "Subject": sub, "Model": name,
                    "Network": NETWORK_NAMES[net_idx],
                    "Jaccard": best_jaccard,
                })

        # --- Sliding-window temporal stability metrics ---
        win_size = t_samples_short
        step = max(1, win_size // 2)
        fc_window_vecs: list[np.ndarray] = []

        for start in range(0, t_samples_long - win_size + 1, step):
            ts_win = ts[start : start + win_size, :]
            fc_win = get_fc_matrix(ts_win, vectorized=False, use_shrinkage=True)
            fc_window_vecs.append(fc_win.flatten())
            sliding_results.append({
                "Seed": seed,
                "Subject": sub,
                "Metric": "Window vs Reference",
                "Value": float(np.corrcoef(fc_win.flatten(), fc_true.flatten())[0, 1]),
            })

        for i in range(len(fc_window_vecs) - 1):
            sliding_results.append({
                "Seed": seed,
                "Subject": sub,
                "Metric": "Adjacent Window Consistency",
                "Value": float(np.corrcoef(fc_window_vecs[i], fc_window_vecs[i + 1])[0, 1]),
            })

    return topology_results, ari_results, jaccard_results, sliding_results


DOT_COLOR = "#333333"
DOT_SIZE = 3.6
DOT_ALPHA = 0.50
JITTER_X_SIGMA = 0.04
JITTER_Y_SIGMA = 0.012


def _jitter_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    order: list[str],
    rng: np.random.RandomState,
    dodge_offset: float = 0.0,
    hue_col: str | None = None,
    hue_val: str | None = None,
) -> None:
    """Add dark scatter dots with gaussian x+y jitter (Figure 7 style).

    Args:
        ax: Target axes.
        df: Data frame.
        x_col: Column for x-axis categories.
        y_col: Column for y-axis metric.
        order: Category order.
        rng: RandomState for reproducible jitter.
        dodge_offset: Horizontal offset for hue groups.
        hue_col: Optional hue column for filtering.
        hue_val: Optional hue value.
    """
    for i, cat in enumerate(order):
        mask = df[x_col] == cat
        if hue_col and hue_val:
            mask = mask & (df[hue_col] == hue_val)
        vals = df.loc[mask, y_col].values
        n = len(vals)
        if n == 0:
            continue
        x_jit = rng.normal(0, JITTER_X_SIGMA, size=n)
        y_jit = rng.normal(0, JITTER_Y_SIGMA, size=n)
        ax.scatter(
            i + dodge_offset + x_jit, vals + y_jit,
            s=DOT_SIZE, c=DOT_COLOR, alpha=DOT_ALPHA,
            edgecolors="none", zorder=4,
        )


def _set_violin_alpha(ax: plt.Axes, alpha: float = 0.65) -> None:
    """Set uniform alpha on all violin body PolyCollections.

    Args:
        ax: Axes containing violin plot.
        alpha: Target alpha for all violin bodies.
    """
    from matplotlib.collections import PolyCollection

    for art in ax.collections:
        if isinstance(art, PolyCollection):
            art.set_alpha(alpha)


def _add_mean_sd_diamond(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    order: list[str],
    dodge_offset: float = 0.0,
    hue_col: str | None = None,
    hue_val: str | None = None,
) -> None:
    """Overlay red diamond mean±SD markers (cross-seed aggregation).

    Args:
        ax: Target axes.
        df: Data frame with Seed column.
        x_col: Column for x-axis categories.
        y_col: Column for y-axis metric.
        order: Category order.
        dodge_offset: Horizontal dodge for hue groups.
        hue_col: Optional hue column for filtering.
        hue_val: Optional hue value to filter.
    """
    for i, cat in enumerate(order):
        mask = df[x_col] == cat
        if hue_col and hue_val:
            mask = mask & (df[hue_col] == hue_val)
        seed_means = df[mask].groupby("Seed")[y_col].mean()
        ax.errorbar(
            i + dodge_offset, seed_means.mean(), yerr=seed_means.std(),
            fmt="D", color="red", markersize=6,
            markeredgecolor="darkred", markeredgewidth=1.0,
            ecolor="darkred", elinewidth=1.2, capsize=3, capthick=1.2,
            zorder=15,
            label="Seed mean ± SD" if i == 0 and dodge_offset <= 0 else None,
        )


def plot_merged_figure(
    df_topo: pd.DataFrame,
    df_ari: pd.DataFrame,
    df_jac: pd.DataFrame,
    df_slide: pd.DataFrame,
) -> plt.Figure:
    """Create the merged 5-panel Network Structure Preservation figure.

    Args:
        df_topo: Topology results (Degree Variance, Small-worldness).
        df_ari: ARI results (Global community overlap).
        df_jac: Per-network Jaccard results.
        df_slide: Sliding-window temporal stability results.

    Returns:
        Matplotlib Figure object.
    """
    apply_bsnet_theme()

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1, 1.1],
        hspace=0.35, wspace=0.30,
    )

    # === Panel A: Global ARI (2 models) ===
    ax_a = fig.add_subplot(gs[0, 0])
    sns.violinplot(
        data=df_ari, x="Model", y="ARI", order=TWO_MODELS,
        palette=TWO_PALETTE, inner="box", linewidth=0, width=0.7,
        cut=2, ax=ax_a,
    )
    _set_violin_alpha(ax_a)
    rng = np.random.RandomState(42)
    _jitter_scatter(ax_a, df_ari, "Model", "ARI", TWO_MODELS, rng)
    _add_mean_sd_diamond(ax_a, df_ari, "Model", "ARI", TWO_MODELS)
    ax_a.set_title("A. Global Community Overlap", fontweight="bold", fontsize=FONT["title"])
    ax_a.set_ylabel("Adjusted Rand Index (ARI)", fontsize=FONT["axis_label"])
    ax_a.set_xlabel("")
    ax_a.legend(loc="lower right", fontsize=FONT["legend_small"])

    # === Panel B: Hub Variance (3 models) ===
    ax_b = fig.add_subplot(gs[0, 1])
    sns.violinplot(
        data=df_topo, x="Model", y="Degree Variance", order=THREE_MODELS,
        palette=THREE_PALETTE, inner="box", linewidth=0, width=0.8,
        cut=2, ax=ax_b,
    )
    _set_violin_alpha(ax_b)
    _jitter_scatter(ax_b, df_topo, "Model", "Degree Variance", THREE_MODELS, rng)
    _add_mean_sd_diamond(ax_b, df_topo, "Model", "Degree Variance", THREE_MODELS)
    ax_b.set_title("B. Hub Variance Conservation", fontweight="bold", fontsize=FONT["title"])
    ax_b.set_ylabel("Variance of Node Degrees", fontsize=FONT["axis_label"])
    ax_b.set_xlabel("")
    ax_b.tick_params(axis="x", rotation=10)
    ax_b.legend(loc="upper right", fontsize=FONT["legend_small"])

    # === Panel C: Small-worldness (3 models) ===
    ax_c = fig.add_subplot(gs[0, 2])
    sns.violinplot(
        data=df_topo, x="Model", y="Small-worldness", order=THREE_MODELS,
        palette=THREE_PALETTE, inner="box", linewidth=0, width=0.8,
        cut=2, ax=ax_c,
    )
    _set_violin_alpha(ax_c)
    _jitter_scatter(ax_c, df_topo, "Model", "Small-worldness", THREE_MODELS, rng)
    ax_c.axhline(
        y=1.0, color=PALETTE["highlight"], linestyle="--", alpha=0.5,
        linewidth=1.5, label="Random graph (σ=1)",
    )
    _add_mean_sd_diamond(ax_c, df_topo, "Model", "Small-worldness", THREE_MODELS)
    ax_c.set_title("C. Small-worldness Preservation", fontweight="bold", fontsize=FONT["title"])
    ax_c.set_ylabel("Small-worldness Index (σ)", fontsize=FONT["axis_label"])
    ax_c.set_xlabel("")
    ax_c.tick_params(axis="x", rotation=10)
    ax_c.legend(loc="upper right", fontsize=FONT["legend_small"])

    # === Panel D: Per-Network Jaccard (bottom-left, 2 columns) ===
    ax_d = fig.add_subplot(gs[1, :2])
    sns.violinplot(
        data=df_jac, x="Network", y="Jaccard", hue="Model",
        palette=TWO_PALETTE, inner="box", linewidth=0, width=0.8,
        cut=2, ax=ax_d,
    )
    _set_violin_alpha(ax_d)
    # Jitter scatter with manual hue dodge
    for m_i, model in enumerate(TWO_MODELS):
        d_off = -0.2 + m_i * 0.4
        _jitter_scatter(
            ax_d, df_jac, "Network", "Jaccard",
            [n for n in NETWORK_NAMES if n in df_jac["Network"].unique()],
            rng, dodge_offset=d_off, hue_col="Model", hue_val=model,
        )

    # Per-network mean±SD overlay (dodge manually)
    networks = [n for n in NETWORK_NAMES if n in df_jac["Network"].unique()]
    for net_i, net in enumerate(networks):
        for m_i, model in enumerate(TWO_MODELS):
            offset = -0.2 + m_i * 0.4
            mask = (df_jac["Network"] == net) & (df_jac["Model"] == model)
            seed_means = df_jac[mask].groupby("Seed")["Jaccard"].mean()
            ax_d.errorbar(
                net_i + offset, seed_means.mean(), yerr=seed_means.std(),
                fmt="D", color="red", markersize=5,
                markeredgecolor="darkred", markeredgewidth=0.8,
                ecolor="darkred", elinewidth=1.0, capsize=2.5, capthick=1.0,
                zorder=15,
                label="Seed mean ± SD" if (net_i == 0 and m_i == 0) else None,
            )

    ax_d.axhline(
        y=1.0, color="red", linestyle="--", alpha=0.3, linewidth=1.5,
        label="Perfect match",
    )
    ax_d.set_title(
        "D. Per-Network Jaccard Similarity (100-ROI Parcellation)",
        fontweight="bold", fontsize=FONT["title"],
    )
    ax_d.set_ylabel("Jaccard Similarity w/ Reference FC", fontsize=FONT["axis_label"])
    ax_d.set_xlabel("")
    ax_d.tick_params(axis="x", rotation=10)

    # Clean legend: keep only first occurrence of each label
    handles, labels = ax_d.get_legend_handles_labels()
    seen: set[str] = set()
    unique_h, unique_l = [], []
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            seen.add(lbl)
            unique_h.append(h)
            unique_l.append(lbl)
    ax_d.legend(unique_h, unique_l, loc="lower right", fontsize=FONT["legend_small"])

    # === Panel E: Sliding-window temporal stability (bottom-right) ===
    ax_e = fig.add_subplot(gs[1, 2])
    slide_order = ["Window vs Reference", "Adjacent Window Consistency"]
    slide_palette = ["#95a5a6", "#fdae61"]
    sns.violinplot(
        data=df_slide, x="Metric", y="Value", order=slide_order,
        palette=slide_palette, inner="box", linewidth=0, width=0.8,
        cut=2, ax=ax_e,
    )
    _set_violin_alpha(ax_e)
    _jitter_scatter(ax_e, df_slide, "Metric", "Value", slide_order, rng)
    _add_mean_sd_diamond(ax_e, df_slide, "Metric", "Value", slide_order)
    ax_e.axhline(
        y=0.0, color=PALETTE["highlight"], linestyle="--", alpha=0.3, linewidth=1.2,
    )
    ax_e.set_title("E. Sliding-window Temporal Stability", fontweight="bold", fontsize=FONT["title"])
    ax_e.set_ylabel("Correlation (r)", fontsize=FONT["axis_label"])
    ax_e.set_xlabel("")
    ax_e.tick_params(axis="x", rotation=10)
    ax_e.legend(loc="lower right", fontsize=FONT["legend_small"])

    return fig


def main() -> None:
    """Generate Figure 4: Network Structure Preservation (merged).

    Runs 10 seeds × 20 subjects × 100 ROIs in a single pass, computing
    topology metrics and community structure metrics, then creates the
    merged 4-panel figure.
    """
    parser = argparse.ArgumentParser(
        description="Figure 4: Network Structure Preservation",
    )
    parser.add_argument(
        "--n-subjects", type=int, default=20,
        help="Number of subjects per seed (default: 20)",
    )
    parser.add_argument(
        "--n-rois", type=int, default=100,
        help="Number of ROIs (default: 100)",
    )
    args = parser.parse_args()

    print("--- Figure 4: Network Structure Preservation + Sliding-window (merged) ---")
    config = BSNetConfig()
    out_dir = Path(config.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_topo: list[dict] = []
    all_ari: list[dict] = []
    all_jac: list[dict] = []
    all_slide: list[dict] = []

    for i, seed in enumerate(SEEDS):
        print(f"  Seed {i + 1}/{len(SEEDS)} (seed={seed})...")
        topo, ari, jac, slide = run_single_seed(
            seed, config,
            n_subjects=args.n_subjects,
            n_rois=args.n_rois,
        )
        all_topo.extend(topo)
        all_ari.extend(ari)
        all_jac.extend(jac)
        all_slide.extend(slide)

    df_topo = pd.DataFrame(all_topo).dropna()
    df_ari = pd.DataFrame(all_ari)
    df_jac = pd.DataFrame(all_jac)
    df_slide = pd.DataFrame(all_slide)

    # Save intermediate CSVs
    df_topo.to_csv(out_dir / "figure4_topology.csv", index=False)
    df_ari.to_csv(out_dir / "figure4_ari.csv", index=False)
    df_jac.to_csv(out_dir / "figure4_jaccard.csv", index=False)
    df_slide.to_csv(out_dir / "figure4_sliding_window.csv", index=False)

    # Cross-seed summary
    print("\n--- Cross-seed Summary ---")
    for model in TWO_MODELS:
        seed_means = df_ari[df_ari["Model"] == model].groupby("Seed")["ARI"].mean()
        print(f"  ARI  {model}: {seed_means.mean():.3f} ± {seed_means.std():.3f}")
    for model in THREE_MODELS:
        for metric in ["Degree Variance", "Small-worldness"]:
            seed_means = (
                df_topo[df_topo["Model"] == model].groupby("Seed")[metric].mean()
            )
            print(
                f"  {metric}  {model}: "
                f"{seed_means.mean():.3f} ± {seed_means.std():.3f}"
            )
    for metric in ["Window vs Reference", "Adjacent Window Consistency"]:
        seed_means = (
            df_slide[df_slide["Metric"] == metric].groupby("Seed")["Value"].mean()
        )
        print(f"  Sliding  {metric}: {seed_means.mean():.3f} ± {seed_means.std():.3f}")

    # Plot and save
    fig = plot_merged_figure(df_topo, df_ari, df_jac, df_slide)
    save_figure(fig, "Figure4_Structure_Preservation.png")
    print("\nFigure 4 saved: Figure4_Structure_Preservation.png")


if __name__ == "__main__":
    main()
