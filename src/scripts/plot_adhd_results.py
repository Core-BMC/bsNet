#!/usr/bin/env python3
"""
ADHD-200 BS-NET 결과 시각화.

Single-seed, multi-seed, group comparison, atlas comparison 시각화.
run_nilearn_adhd_bsnet.py 의 출력 CSV를 입력으로 사용.

Usage:
    # Single-seed 4-panel figure
    python src/scripts/plot_adhd_results.py --csv data/adhd/results/adhd_bsnet_cc200.csv

    # Multi-seed 4-panel figure
    python src/scripts/plot_adhd_results.py --multi-seed-csv data/adhd/results/adhd_multiseed_cc200_10seeds.csv

    # Group comparison (ADHD vs Control)
    python src/scripts/plot_adhd_results.py --csv data/adhd/results/adhd_bsnet_cc200.csv --group-compare

    # Atlas comparison (CC200 vs CC400)
    python src/scripts/plot_adhd_results.py --atlas-compare

    # All plots at once
    python src/scripts/plot_adhd_results.py --all
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/adhd/results")


# ============================================================================
# Data loaders
# ============================================================================
def load_single_seed(csv_path: str) -> dict:
    """Load single-seed results CSV."""
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    return {
        "sub_idx": [int(r["sub_idx"]) for r in rows],
        "r_fc_raw": np.array([float(r["r_fc_raw"]) for r in rows]),
        "rho_hat_T": np.array([float(r["rho_hat_T"]) for r in rows]),
        "ci_lower": np.array([float(r["ci_lower"]) for r in rows]),
        "ci_upper": np.array([float(r["ci_upper"]) for r in rows]),
        "improvement": np.array([float(r["improvement"]) for r in rows]),
        "group": [r.get("group", "unknown") for r in rows],
        "n_rois": int(rows[0].get("n_rois", 0)),
        "atlas": rows[0].get("atlas", "unknown"),
    }


def load_multi_seed(csv_path: str) -> dict:
    """Load multi-seed results CSV."""
    with open(csv_path) as f:
        rows = list(csv_mod.DictReader(f))

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    return {
        "sub_idx": [int(r["sub_idx"]) for r in rows],
        "r_fc_raw": np.array([float(r["r_fc_raw"]) for r in rows]),
        "rho_mean": np.array([float(r["rho_mean"]) for r in rows]),
        "rho_std": np.array([float(r["rho_std"]) for r in rows]),
        "rho_min": np.array([float(r["rho_min"]) for r in rows]),
        "rho_max": np.array([float(r["rho_max"]) for r in rows]),
        "group": [r.get("group", "unknown") for r in rows],
        "n_rois": int(rows[0].get("n_rois", 0)),
        "atlas": rows[0].get("atlas", "unknown"),
    }


# ============================================================================
# Plot functions
# ============================================================================
def _setup_matplotlib():
    """Configure matplotlib for non-interactive backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_single_seed(data: dict, output_path: str) -> None:
    """4-panel figure: scatter, improvement, distribution, CI."""
    plt = _setup_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    atlas = data["atlas"].upper()
    n = len(data["sub_idx"])
    fig.suptitle(
        f"BS-NET Validation on ADHD-200 ({atlas}, N={n})",
        fontsize=14, fontweight="bold",
    )

    r_raw = data["r_fc_raw"]
    rho_hat = data["rho_hat_T"]
    improvement = data["improvement"]
    groups = data["group"]

    # Color map for groups
    color_map = {"adhd": "tomato", "control": "steelblue", "unknown": "gray"}

    # Panel A: Scatter r_FC vs ρ̂T colored by group
    ax = axes[0, 0]
    for grp, clr, label in [("control", "steelblue", "Control"),
                             ("adhd", "tomato", "ADHD")]:
        mask = [g == grp for g in groups]
        if any(mask):
            idx = np.where(mask)[0]
            ax.scatter(r_raw[idx], rho_hat[idx], c=clr, alpha=0.7,
                       edgecolors="k", lw=0.5, s=60, label=label)
    lims = [min(r_raw.min(), rho_hat.min()) - 0.05, 1.02]
    ax.plot(lims, lims, "k--", alpha=0.3, label="identity")
    ax.set_xlabel("r_FC (raw, 2min vs full)")
    ax.set_ylabel("ρ̂T (BS-NET)")
    ax.set_title("A. Raw FC vs BS-NET Prediction")
    ax.legend(fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Panel B: Improvement by group (box + strip)
    ax = axes[0, 1]
    grp_data = {}
    for g, imp in zip(groups, improvement):
        grp_data.setdefault(g, []).append(imp)
    grp_names = sorted(grp_data.keys())
    bp = ax.boxplot(
        [grp_data[g] for g in grp_names],
        tick_labels=[g.capitalize() for g in grp_names],
        patch_artist=True, vert=True,
    )
    for patch, name in zip(bp["boxes"], grp_names):
        patch.set_facecolor(color_map.get(name, "gray"))
        patch.set_alpha(0.5)
    # Strip plot overlay
    for i, name in enumerate(grp_names):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(grp_data[name]))
        ax.scatter(
            np.full(len(grp_data[name]), i + 1) + jitter,
            grp_data[name],
            c=color_map.get(name, "gray"), edgecolors="k",
            lw=0.5, s=40, alpha=0.7, zorder=5,
        )
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.set_ylabel("Δ (ρ̂T − r_FC)")
    ax.set_title("B. Improvement by Group")

    # Panel C: Distribution comparison
    ax = axes[1, 0]
    bins = np.linspace(0.5, 1.05, 25)
    ax.hist(r_raw, bins=bins, alpha=0.4, label="r_FC (raw)", color="gray", edgecolor="k")
    ax.hist(rho_hat, bins=bins, alpha=0.5, label="ρ̂T (BS-NET)", color="steelblue", edgecolor="k")
    ax.axvline(np.mean(r_raw), color="gray", ls="--", lw=1.5)
    ax.axvline(np.mean(rho_hat), color="steelblue", ls="--", lw=1.5)
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Count")
    ax.set_title("C. Distribution Comparison")
    ax.legend()

    # Panel D: Per-subject CI with group color
    ax = axes[1, 1]
    sort_idx = np.argsort(rho_hat)
    ci_lo = data["ci_lower"]
    ci_hi = data["ci_upper"]
    for pos, j in enumerate(sort_idx):
        clr = color_map.get(groups[j], "gray")
        ax.errorbar(
            rho_hat[j], pos,
            xerr=[[rho_hat[j] - ci_lo[j]], [ci_hi[j] - rho_hat[j]]],
            fmt="o", color=clr, ecolor="lightgray", capsize=2, ms=5,
        )
    ax.scatter(r_raw[sort_idx], range(len(sort_idx)),
               marker="x", color="gray", s=30, zorder=5, label="r_FC")
    ax.set_yticks(range(len(sort_idx)))
    ax.set_yticklabels([f"sub_{data['sub_idx'][j]:03d}" for j in sort_idx], fontsize=7)
    ax.set_xlabel("Correlation")
    ax.set_title("D. BS-NET with 95% CI (sorted)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved: {output_path}")
    plt.close()


def plot_multi_seed(data: dict, output_path: str) -> None:
    """4-panel figure: scatter ± seed error, group box, improvement, stability."""
    plt = _setup_matplotlib()

    rho_mean = data["rho_mean"]
    rho_std = data["rho_std"]
    r_raw = data["r_fc_raw"]
    groups = data["group"]
    n_subs = len(rho_mean)
    atlas = data["atlas"].upper()

    improvement = rho_mean - r_raw
    color_map = {"adhd": "tomato", "control": "steelblue", "unknown": "gray"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"BS-NET Multi-Seed — ADHD-200 ({atlas}, N={n_subs})",
        fontsize=14, fontweight="bold",
    )

    # Panel A: Scatter with seed error bars, colored by group
    ax = axes[0, 0]
    for grp, clr, label in [("control", "steelblue", "Control"),
                             ("adhd", "tomato", "ADHD")]:
        mask = np.array([g == grp for g in groups])
        if mask.any():
            ax.errorbar(
                r_raw[mask], rho_mean[mask], yerr=rho_std[mask] * 1.96,
                fmt="o", color=clr, ecolor=clr, capsize=3,
                ms=6, alpha=0.7, label=label,
            )
    lims = [min(r_raw.min(), rho_mean.min()) - 0.05, 1.05]
    ax.plot(lims, lims, "k--", alpha=0.3, label="identity")
    ax.set_xlabel("r_FC (raw)")
    ax.set_ylabel("ρ̂T (mean ± 1.96σ)")
    ax.set_title("A. Raw FC vs BS-NET (multi-seed)")
    ax.legend(fontsize=9)

    # Panel B: Group comparison (box + strip)
    ax = axes[0, 1]
    grp_rho = {}
    for g, rho in zip(groups, rho_mean):
        grp_rho.setdefault(g, []).append(rho)
    grp_names = sorted(grp_rho.keys())
    bp = ax.boxplot(
        [grp_rho[g] for g in grp_names],
        tick_labels=[g.capitalize() for g in grp_names],
        patch_artist=True, vert=True,
    )
    for patch, name in zip(bp["boxes"], grp_names):
        patch.set_facecolor(color_map.get(name, "gray"))
        patch.set_alpha(0.5)
    for i, name in enumerate(grp_names):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(grp_rho[name]))
        ax.scatter(
            np.full(len(grp_rho[name]), i + 1) + jitter,
            grp_rho[name],
            c=color_map.get(name, "gray"), edgecolors="k",
            lw=0.5, s=40, alpha=0.7, zorder=5,
        )
    ax.set_ylabel("ρ̂T (mean across seeds)")
    ax.set_title("B. ρ̂T by Group")

    # Panel C: Improvement distribution
    ax = axes[1, 0]
    for grp, clr in [("control", "steelblue"), ("adhd", "tomato")]:
        mask = [g == grp for g in groups]
        if any(mask):
            ax.hist(improvement[mask], bins=12, alpha=0.5,
                    label=grp.capitalize(), color=clr, edgecolor="k")
    ax.axvline(0, color="k", ls="--")
    ax.axvline(np.mean(improvement), color="black", ls="-", lw=2,
               label=f"mean={np.mean(improvement):.3f}")
    ax.set_xlabel("Δ (ρ̂T_mean − r_FC)")
    ax.set_ylabel("Count")
    ax.set_title("C. Improvement Distribution")
    ax.legend(fontsize=9)

    # Panel D: Seed stability per subject (sorted by std)
    ax = axes[1, 1]
    sort_idx = np.argsort(rho_std)
    bar_colors = [color_map.get(groups[j], "gray") for j in sort_idx]
    ax.barh(range(n_subs), rho_std[sort_idx], color=bar_colors, edgecolor="k", lw=0.5)
    ax.set_yticks(range(n_subs))
    ax.set_yticklabels([f"sub_{data['sub_idx'][j]:03d}" for j in sort_idx], fontsize=7)
    ax.set_xlabel("σ(ρ̂T) across seeds")
    ax.set_title(f"D. Seed Stability (mean σ={np.mean(rho_std):.4f})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved: {output_path}")
    plt.close()


def plot_atlas_comparison(output_path: str) -> None:
    """Compare CC200 vs CC400 results in a 2x2 figure."""
    plt = _setup_matplotlib()

    # Try to load both single-seed CSVs
    datasets = {}
    for atlas in ("cc200", "cc400"):
        csv_path = RESULTS_DIR / f"adhd_bsnet_{atlas}.csv"
        if csv_path.exists():
            datasets[atlas] = load_single_seed(str(csv_path))

    if len(datasets) < 2:
        logger.warning(
            f"Atlas comparison requires CC200 and CC400 results. "
            f"Found: {list(datasets.keys())}"
        )
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "BS-NET Atlas Comparison — ADHD-200 (CC200 vs CC400)",
        fontsize=14, fontweight="bold",
    )

    atlas_colors = {"cc200": "steelblue", "cc400": "coral"}

    # Panel A: ρ̂T distribution comparison
    ax = axes[0, 0]
    bins = np.linspace(0.6, 1.0, 20)
    for atlas, d in datasets.items():
        ax.hist(d["rho_hat_T"], bins=bins, alpha=0.5,
                label=f"{atlas.upper()} (μ={np.mean(d['rho_hat_T']):.3f})",
                color=atlas_colors[atlas], edgecolor="k")
    ax.set_xlabel("ρ̂T")
    ax.set_ylabel("Count")
    ax.set_title("A. ρ̂T Distribution by Atlas")
    ax.legend()

    # Panel B: Improvement distribution
    ax = axes[0, 1]
    for atlas, d in datasets.items():
        ax.hist(d["improvement"], bins=15, alpha=0.5,
                label=f"{atlas.upper()} (μ={np.mean(d['improvement']):+.3f})",
                color=atlas_colors[atlas], edgecolor="k")
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel("Δ (ρ̂T − r_FC)")
    ax.set_ylabel("Count")
    ax.set_title("B. Improvement by Atlas")
    ax.legend()

    # Panel C: Paired scatter CC200 vs CC400
    ax = axes[1, 0]
    d200 = datasets["cc200"]
    d400 = datasets["cc400"]
    # Match by sub_idx
    idx_set = set(d200["sub_idx"]) & set(d400["sub_idx"])
    rho200 = {s: r for s, r in zip(d200["sub_idx"], d200["rho_hat_T"])}
    rho400 = {s: r for s, r in zip(d400["sub_idx"], d400["rho_hat_T"])}
    paired_200 = np.array([rho200[s] for s in sorted(idx_set)])
    paired_400 = np.array([rho400[s] for s in sorted(idx_set)])
    groups_200 = {s: g for s, g in zip(d200["sub_idx"], d200["group"])}
    grp_colors = [
        "tomato" if groups_200.get(s) == "adhd" else "steelblue"
        for s in sorted(idx_set)
    ]

    ax.scatter(paired_200, paired_400, c=grp_colors, edgecolors="k",
               lw=0.5, s=60, alpha=0.7)
    lims = [min(paired_200.min(), paired_400.min()) - 0.02,
            max(paired_200.max(), paired_400.max()) + 0.02]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel("ρ̂T (CC200)")
    ax.set_ylabel("ρ̂T (CC400)")
    ax.set_title(f"C. Paired Comparison (r={np.corrcoef(paired_200, paired_400)[0,1]:.3f})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # Legend for groups
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=8, label="Control"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
               markersize=8, label="ADHD"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    # Panel D: Summary bar chart
    ax = axes[1, 1]
    atlas_names = list(datasets.keys())
    x = np.arange(len(atlas_names))
    r_means = [np.mean(datasets[a]["r_fc_raw"]) for a in atlas_names]
    rho_means = [np.mean(datasets[a]["rho_hat_T"]) for a in atlas_names]
    r_stds = [np.std(datasets[a]["r_fc_raw"]) for a in atlas_names]
    rho_stds = [np.std(datasets[a]["rho_hat_T"]) for a in atlas_names]

    w = 0.35
    ax.bar(x - w / 2, r_means, w, yerr=r_stds, label="r_FC (raw)",
           color="lightgray", edgecolor="k", capsize=5)
    ax.bar(x + w / 2, rho_means, w, yerr=rho_stds, label="ρ̂T (BS-NET)",
           color="steelblue", edgecolor="k", capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in atlas_names])
    ax.set_ylabel("Correlation")
    ax.set_title("D. Summary: r_FC vs ρ̂T by Atlas")
    ax.legend()
    n_rois = [datasets[a]["n_rois"] for a in atlas_names]
    for i, nr in enumerate(n_rois):
        ax.text(i, 0.55, f"{nr} ROIs", ha="center", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved: {output_path}")
    plt.close()


# ============================================================================
# CLI
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ADHD-200 BS-NET results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", default=None,
        help="Single-seed results CSV (e.g. data/adhd/results/adhd_bsnet_cc200.csv)",
    )
    parser.add_argument(
        "--multi-seed-csv", default=None,
        help="Multi-seed results CSV (e.g. data/adhd/results/adhd_multiseed_cc200_10seeds.csv)",
    )
    parser.add_argument(
        "--group-compare", action="store_true",
        help="Generate group comparison plot (requires --csv with group column)",
    )
    parser.add_argument(
        "--atlas-compare", action="store_true",
        help="Generate atlas comparison (CC200 vs CC400)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate all available plots",
    )
    parser.add_argument(
        "--output-dir", default="data/adhd/results",
        help="Output directory for figures (default: data/adhd/results)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    # --all: auto-detect available results
    if args.all:
        for atlas in ("cc200", "cc400"):
            csv_path = RESULTS_DIR / f"adhd_bsnet_{atlas}.csv"
            if csv_path.exists():
                data = load_single_seed(str(csv_path))
                fig_path = str(out_dir / f"adhd_bsnet_{atlas}.png")
                plot_single_seed(data, fig_path)
                generated.append(fig_path)

            ms_csv = sorted(RESULTS_DIR.glob(f"adhd_multiseed_{atlas}_*seeds.csv"))
            if ms_csv:
                ms_data = load_multi_seed(str(ms_csv[-1]))  # latest
                fig_path = str(out_dir / f"adhd_multiseed_{atlas}.png")
                plot_multi_seed(ms_data, fig_path)
                generated.append(fig_path)

        atlas_fig = str(out_dir / "adhd_atlas_comparison.png")
        plot_atlas_comparison(atlas_fig)
        generated.append(atlas_fig)

        logger.info(f"Generated {len(generated)} figures")
        return

    # Individual modes
    if args.csv:
        data = load_single_seed(args.csv)
        fig_path = str(out_dir / f"adhd_bsnet_{data['atlas']}.png")
        plot_single_seed(data, fig_path)
        generated.append(fig_path)

    if args.multi_seed_csv:
        ms_data = load_multi_seed(args.multi_seed_csv)
        fig_path = str(out_dir / f"adhd_multiseed_{ms_data['atlas']}.png")
        plot_multi_seed(ms_data, fig_path)
        generated.append(fig_path)

    if args.atlas_compare:
        fig_path = str(out_dir / "adhd_atlas_comparison.png")
        plot_atlas_comparison(fig_path)
        generated.append(fig_path)

    if not generated:
        parser.print_help()
        logger.warning("No plots generated. Specify --csv, --multi-seed-csv, --atlas-compare, or --all")


if __name__ == "__main__":
    main()
