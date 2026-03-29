#!/usr/bin/env python3
"""
ABIDE BS-NET 결과 시각화.

Usage:
    python src/scripts/plot_abide_results.py
    python src/scripts/plot_abide_results.py --csv data/abide/results/abide_bsnet_cc200_cpac_filt_noglobal.csv
    python src/scripts/plot_abide_results.py --multi-seed 10  # 10 seeds로 재실행 후 시각화
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)


def load_results(csv_path: str) -> dict:
    """Load results CSV into dict of arrays."""
    import csv as csv_mod

    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)

    return {
        "sub_id": [r["sub_id"] for r in rows],
        "site": [r["site"] for r in rows],
        "r_fc_raw": np.array([float(r["r_fc_raw"]) for r in rows]),
        "rho_hat_T": np.array([float(r["rho_hat_T"]) for r in rows]),
        "ci_lower": np.array([float(r["ci_lower"]) for r in rows]),
        "ci_upper": np.array([float(r["ci_upper"]) for r in rows]),
        "improvement": np.array([float(r["improvement"]) for r in rows]),
        "n_rois": int(rows[0]["n_rois"]) if rows else 0,
        "atlas": rows[0]["atlas"] if rows else "unknown",
    }


def _multiseed_worker(args: tuple) -> dict:
    """Worker for parallel multi-seed BS-NET per subject.

    Args:
        args: (subject_index, ts_path, tr, short_sec, n_seeds,
               orig_rho, orig_ci_lo, orig_ci_hi, correction_method).

    Returns:
        Dict with rho_all, ci_lo_all, ci_hi_all arrays for this subject.
    """
    from src.core.config import BSNetConfig
    from src.core.pipeline import run_bootstrap_prediction
    from src.data.data_loader import get_fc_matrix

    (i, ts_path, tr, short_sec, n_seeds,
     orig_rho, orig_ci_lo, orig_ci_hi, corr_method) = args

    rho = np.zeros(n_seeds)
    ci_lo = np.zeros(n_seeds)
    ci_hi = np.zeros(n_seeds)

    if not Path(ts_path).exists():
        rho[:] = orig_rho
        ci_lo[:] = orig_ci_lo
        ci_hi[:] = orig_ci_hi
        return {"idx": i, "rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi}

    ts = np.load(ts_path).astype(np.float64)
    valid = np.std(ts, axis=0) > 1e-8
    ts = ts[:, valid]

    n_rois = ts.shape[1]
    short_vols = int(short_sec / tr)
    ts_short = ts[:short_vols, :]
    fc_full_vec = get_fc_matrix(ts, vectorized=True, use_shrinkage=True)

    for s in range(n_seeds):
        seed = 42 + s * 7
        config = BSNetConfig(
            n_rois=n_rois, tr=tr,
            short_duration_sec=short_sec,
            target_duration_min=15,
            n_bootstraps=100, seed=seed,
        )
        result = run_bootstrap_prediction(
            ts_short, fc_full_vec, config,
            correction_method=corr_method,
        )
        rho[s] = float(result.rho_hat_T)
        ci_lo[s] = float(result.ci_lower)
        ci_hi[s] = float(result.ci_upper)

    return {"idx": i, "rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi}


def _resolve_n_jobs(n_jobs: int) -> int:
    """Resolve n_jobs to actual worker count."""
    cpu_count = os.cpu_count() or 1
    if n_jobs == -1:
        return cpu_count
    if n_jobs <= 0:
        return max(1, cpu_count + n_jobs)
    return min(n_jobs, cpu_count)


def run_multi_seed(
    csv_path: str,
    n_seeds: int = 10,
    output_dir: str | None = None,
    n_jobs: int = 1,
    correction_method: str = "original",
) -> dict:
    """Re-run BS-NET with multiple seeds and collect distributions.

    Args:
        csv_path: Original results CSV (for metadata).
        n_seeds: Number of random seeds.
        output_dir: Output directory for multi-seed CSV.
        n_jobs: Parallel workers (1=sequential, -1=all cores).
        correction_method: Attenuation correction method (see bootstrap.py).

    Returns:
        Dict with per-subject arrays: rho_hat_T_all (n_subjects, n_seeds), etc.
    """
    import csv as csv_mod

    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)

    # Locate time series cache
    base_dir = Path(csv_path).parents[1]
    atlas = rows[0]["atlas"]
    cache_dir = base_dir / "timeseries_cache" / atlas

    n_subs = len(rows)
    rho_all = np.zeros((n_subs, n_seeds))
    ci_lo_all = np.zeros((n_subs, n_seeds))
    ci_hi_all = np.zeros((n_subs, n_seeds))
    r_fc_raw = np.array([float(r["r_fc_raw"]) for r in rows])

    n_workers = _resolve_n_jobs(n_jobs)
    use_parallel = n_workers > 1

    # Build worker args
    worker_args = []
    for i, row in enumerate(rows):
        sub_id = row["sub_id"]
        tr = float(row["tr"])
        ts_path = str(cache_dir / f"{sub_id}_{atlas}.npy")
        worker_args.append((
            i, ts_path, tr, 120, n_seeds,
            float(row["rho_hat_T"]), float(row["ci_lower"]), float(row["ci_upper"]),
            correction_method,
        ))

    if use_parallel:
        logger.info(f"Multi-seed parallel mode: {n_workers} workers, {n_subs} subjects × {n_seeds} seeds")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_multiseed_worker, arg): arg[0]
                for arg in worker_args
            }
            for done, future in enumerate(as_completed(futures), 1):
                res = future.result()
                idx = res["idx"]
                rho_all[idx, :] = res["rho"]
                ci_lo_all[idx, :] = res["ci_lo"]
                ci_hi_all[idx, :] = res["ci_hi"]
                if done % 5 == 0 or done == 1:
                    sub_id = rows[idx]["sub_id"]
                    logger.info(
                        f"  [{done}/{n_subs}] {sub_id}: "
                        f"ρ̂T={np.mean(rho_all[idx]):.3f} ± {np.std(rho_all[idx]):.3f}"
                    )
    else:
        logger.info(f"Multi-seed sequential mode: {n_subs} subjects × {n_seeds} seeds")
        for arg in worker_args:
            res = _multiseed_worker(arg)
            idx = res["idx"]
            rho_all[idx, :] = res["rho"]
            ci_lo_all[idx, :] = res["ci_lo"]
            ci_hi_all[idx, :] = res["ci_hi"]
            if (idx + 1) % 5 == 0 or idx == 0:
                sub_id = rows[idx]["sub_id"]
                logger.info(
                    f"  [{idx+1}/{n_subs}] {sub_id}: "
                    f"ρ̂T={np.mean(rho_all[idx]):.3f} ± {np.std(rho_all[idx]):.3f}"
                )

    # Save multi-seed CSV
    out_path = Path(output_dir) if output_dir else Path(csv_path).parent
    ms_csv = out_path / f"abide_multiseed_{atlas}_{n_seeds}seeds.csv"

    with open(ms_csv, "w", newline="") as f:
        writer = csv_mod.writer(f)
        header = ["sub_id", "site", "r_fc_raw",
                  "rho_hat_T_mean", "rho_hat_T_std",
                  "rho_hat_T_min", "rho_hat_T_max",
                  "ci_lower_mean", "ci_upper_mean"]
        writer.writerow(header)
        for i, row in enumerate(rows):
            writer.writerow([
                row["sub_id"], row["site"], row["r_fc_raw"],
                f"{np.mean(rho_all[i]):.4f}", f"{np.std(rho_all[i]):.4f}",
                f"{np.min(rho_all[i]):.4f}", f"{np.max(rho_all[i]):.4f}",
                f"{np.mean(ci_lo_all[i]):.4f}", f"{np.mean(ci_hi_all[i]):.4f}",
            ])
    logger.info(f"Multi-seed CSV: {ms_csv}")

    return {
        "sub_id": [r["sub_id"] for r in rows],
        "site": [r["site"] for r in rows],
        "r_fc_raw": r_fc_raw,
        "rho_all": rho_all,
        "ci_lo_all": ci_lo_all,
        "ci_hi_all": ci_hi_all,
        "atlas": atlas,
        "n_seeds": n_seeds,
    }


def plot_single_seed(data: dict, output_path: str) -> None:
    """Generate 4-panel figure from single-seed results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"BS-NET Validation on ABIDE ({data['atlas'].upper()}, "
        f"N={len(data['sub_id'])})",
        fontsize=14, fontweight="bold",
    )

    r_raw = data["r_fc_raw"]
    rho_hat = data["rho_hat_T"]
    improvement = data["improvement"]

    # Panel A: Scatter r_fc_raw vs rho_hat_T
    ax = axes[0, 0]
    ax.scatter(r_raw, rho_hat, c="steelblue", alpha=0.7, edgecolors="k", lw=0.5, s=60)
    lims = [min(r_raw.min(), rho_hat.min()) - 0.05, 1.02]
    ax.plot(lims, lims, "k--", alpha=0.3, label="identity")
    ax.set_xlabel("r_FC (raw, 2min vs full)")
    ax.set_ylabel("ρ̂T (BS-NET)")
    ax.set_title("A. Raw FC vs BS-NET Prediction")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Panel B: Improvement bar chart per subject
    ax = axes[0, 1]
    idx = np.argsort(improvement)
    colors = ["forestgreen" if v > 0 else "tomato" for v in improvement[idx]]
    ax.barh(range(len(idx)), improvement[idx], color=colors, edgecolor="k", lw=0.5)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([data["sub_id"][j] for j in idx], fontsize=8)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Δ (ρ̂T − r_FC)")
    ax.set_title("B. Per-Subject Improvement")

    # Panel C: Distribution of r_fc_raw vs rho_hat_T
    ax = axes[1, 0]
    bins = np.linspace(0.5, 1.05, 25)
    ax.hist(r_raw, bins=bins, alpha=0.5, label="r_FC (raw)", color="gray", edgecolor="k")
    ax.hist(rho_hat, bins=bins, alpha=0.5, label="ρ̂T (BS-NET)", color="steelblue", edgecolor="k")
    ax.axvline(np.mean(r_raw), color="gray", ls="--", lw=1.5)
    ax.axvline(np.mean(rho_hat), color="steelblue", ls="--", lw=1.5)
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Count")
    ax.set_title("C. Distribution Comparison")
    ax.legend()

    # Panel D: Error bar (CI) per subject
    ax = axes[1, 1]
    ci_lo = data["ci_lower"]
    ci_hi = data["ci_upper"]
    y_pos = np.arange(len(rho_hat))
    ax.errorbar(
        rho_hat, y_pos,
        xerr=[rho_hat - ci_lo, ci_hi - rho_hat],
        fmt="o", color="steelblue", ecolor="gray", capsize=3, ms=5,
    )
    ax.scatter(r_raw, y_pos, marker="x", color="tomato", s=40, label="r_FC (raw)", zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["sub_id"], fontsize=8)
    ax.set_xlabel("Correlation")
    ax.set_title("D. BS-NET with 95% CI")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved: {output_path}")
    plt.close()


def plot_multi_seed(data: dict, output_path: str) -> None:
    """Generate 4-panel figure from multi-seed results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho_all = data["rho_all"]  # (n_subjects, n_seeds)
    r_raw = data["r_fc_raw"]
    n_subs = rho_all.shape[0]
    n_seeds = data["n_seeds"]

    rho_mean = np.mean(rho_all, axis=1)
    rho_std = np.std(rho_all, axis=1)
    improvement = rho_mean - r_raw

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"BS-NET Multi-Seed Validation ({data['atlas'].upper()}, "
        f"N={n_subs}, seeds={n_seeds})",
        fontsize=14, fontweight="bold",
    )

    # Panel A: Scatter with error bars
    ax = axes[0, 0]
    ax.errorbar(
        r_raw, rho_mean, yerr=rho_std * 1.96,
        fmt="o", color="steelblue", ecolor="lightblue",
        capsize=3, ms=6, alpha=0.8,
    )
    lims = [min(r_raw.min(), rho_mean.min()) - 0.05, 1.05]
    ax.plot(lims, lims, "k--", alpha=0.3, label="identity")
    ax.set_xlabel("r_FC (raw)")
    ax.set_ylabel("ρ̂T (mean ± 1.96σ)")
    ax.set_title(f"A. Raw FC vs BS-NET ({n_seeds} seeds)")
    ax.legend()

    # Panel B: Seed variability per subject (box plot)
    ax = axes[0, 1]
    ax.boxplot(
        [rho_all[i, :] for i in range(n_subs)],
        vert=False, patch_artist=True,
        boxprops=dict(facecolor="lightsteelblue"),
    )
    ax.scatter(r_raw, range(1, n_subs + 1), marker="x", color="tomato", s=40, zorder=5, label="r_FC")
    ax.set_yticklabels(data["sub_id"], fontsize=8)
    ax.set_xlabel("ρ̂T")
    ax.set_title("B. Seed Variability (box) + Raw FC (x)")
    ax.legend()

    # Panel C: Improvement distribution
    ax = axes[1, 0]
    ax.hist(improvement, bins=15, color="steelblue", edgecolor="k", alpha=0.7)
    ax.axvline(0, color="k", ls="--")
    ax.axvline(np.mean(improvement), color="tomato", ls="-", lw=2, label=f"mean={np.mean(improvement):.3f}")
    ax.set_xlabel("Δ (ρ̂T_mean − r_FC)")
    ax.set_ylabel("Count")
    ax.set_title("C. Improvement Distribution")
    ax.legend()

    # Panel D: Seed stability (std per subject)
    ax = axes[1, 1]
    idx = np.argsort(rho_std)
    ax.barh(range(n_subs), rho_std[idx], color="coral", edgecolor="k", lw=0.5)
    ax.set_yticks(range(n_subs))
    ax.set_yticklabels([data["sub_id"][j] for j in idx], fontsize=8)
    ax.set_xlabel("σ(ρ̂T) across seeds")
    ax.set_title("D. Prediction Stability")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Figure saved: {output_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ABIDE BS-NET results")
    parser.add_argument(
        "--csv", default="data/abide/results/abide_bsnet_cc200_cpac_filt_noglobal.csv",
        help="Results CSV path",
    )
    parser.add_argument("--multi-seed", type=int, default=0, help="Run N seeds (0=single seed plot)")
    parser.add_argument("--output-dir", default="data/abide/results", help="Output dir for figures")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel workers for multi-seed (1=sequential, -1=all cores)",
    )
    parser.add_argument(
        "--correction-method",
        choices=["original", "fisher_z", "partial", "soft_clamp"],
        default="original",
        help="Attenuation correction method (default: original)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    csv_path = args.csv
    if not Path(csv_path).exists():
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.multi_seed > 0:
        logger.info(f"Running multi-seed analysis ({args.multi_seed} seeds)...")
        ms_data = run_multi_seed(
            csv_path, n_seeds=args.multi_seed, n_jobs=args.n_jobs,
            correction_method=args.correction_method,
        )
        fig_path = str(out_dir / f"abide_bsnet_multiseed_{ms_data['atlas']}.png")
        plot_multi_seed(ms_data, fig_path)
    else:
        data = load_results(csv_path)
        fig_path = str(out_dir / f"abide_bsnet_{data['atlas']}.png")
        plot_single_seed(data, fig_path)


if __name__ == "__main__":
    main()
