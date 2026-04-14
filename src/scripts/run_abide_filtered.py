#!/usr/bin/env python3
"""ABIDE PCP filtered multi-seed BS-NET validation.

Quality-filtered ABIDE analysis with CONSORT-style progressive filtering:
  Stage 1: k ≥ 2.0  (reference ≥ short scan)
  Stage 2: Reference ≥ 3 min
  Stage 3: Total scan ≥ 5 min

Runs multi-seed BS-NET (Fisher z) on the filtered subset and saves:
  - CONSORT flowchart PNG
  - Filtered multi-seed CSV (same format as abide_multiseed_{atlas}_10seeds.csv)
  - Filtering summary JSON

Usage:
    # Default: CC200, 10 seeds, 8 workers, Fisher z
    PYTHONPATH=. python src/scripts/run_abide_filtered.py

    # CC400
    PYTHONPATH=. python src/scripts/run_abide_filtered.py --atlas cc400

    # Custom filter: only k >= 2 (skip ref/total filters)
    PYTHONPATH=. python src/scripts/run_abide_filtered.py --filter-mode liberal

    # Strict filter: ref >= 5 min
    PYTHONPATH=. python src/scripts/run_abide_filtered.py --filter-mode strict
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ── Project imports ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import BSNetConfig
from src.core.pipeline import run_bootstrap_prediction
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)

SHORT_TRS = 60  # BS-NET short scan = first 60 TRs


# ═══════════════════════════════════════════════════════════════════════
# Filtering
# ═══════════════════════════════════════════════════════════════════════

def load_subjects(atlas: str) -> list[dict]:
    """Load ABIDE subject metadata + scan lengths.

    Returns:
        List of dicts with keys: sub_id, site, group, tr, n_trs,
        total_s, short_s, ref_s, k, ts_path.
    """
    json_path = Path(f"data/abide/results/abide_subjects_{atlas}.json")
    with open(json_path) as f:
        raw = json.load(f)

    subjects = []
    for s in raw:
        ts_path = s["ts_path"]
        if not Path(ts_path).exists():
            continue
        ts = np.load(ts_path)
        n_trs = ts.shape[0]
        tr = s.get("tr", 2.0)
        subjects.append({
            "sub_id": s["sub_id"],
            "site": s["site"],
            "group": s.get("group", "Unknown"),
            "tr": tr,
            "n_trs": n_trs,
            "n_rois": ts.shape[1],
            "total_s": n_trs * tr,
            "short_s": SHORT_TRS * tr,
            "ref_s": (n_trs - SHORT_TRS) * tr,
            "k": n_trs / SHORT_TRS,
            "ts_path": ts_path,
        })
    return subjects


def apply_filters(
    subjects: list[dict],
    mode: str = "moderate",
) -> tuple[list[dict], list[dict]]:
    """Apply progressive quality filters.

    Args:
        subjects: Full subject list.
        mode: 'liberal' (k>=2 only), 'moderate' (k>=2 + ref>=3min + total>=5min),
              'strict' (k>=2 + ref>=5min).

    Returns:
        (filtered_subjects, consort_stages) — stages for flowchart.
    """
    stages = []
    current = subjects.copy()

    stages.append({
        "name": "Enrolled",
        "n": len(current),
        "n_sites": len(set(s["site"] for s in current)),
        "excluded": [],
        "reason": "",
    })

    # Stage 1: k >= 2
    excluded = [s for s in current if s["k"] < 2.0]
    current = [s for s in current if s["k"] >= 2.0]
    stages.append({
        "name": "k ≥ 2.0 (ref ≥ short)",
        "n": len(current),
        "n_sites": len(set(s["site"] for s in current)),
        "excluded": excluded,
        "reason": "k < 2 (reference shorter than short scan)",
        "excl_sites": dict(Counter(s["site"] for s in excluded)),
    })

    if mode in ("moderate", "strict"):
        # Stage 2: reference >= threshold
        ref_min = 300 if mode == "strict" else 180
        ref_label = "5 min" if mode == "strict" else "3 min"
        excluded = [s for s in current if s["ref_s"] < ref_min]
        current = [s for s in current if s["ref_s"] >= ref_min]
        stages.append({
            "name": f"Reference ≥ {ref_label}",
            "n": len(current),
            "n_sites": len(set(s["site"] for s in current)),
            "excluded": excluded,
            "reason": f"Reference < {ref_label}",
            "excl_sites": dict(Counter(s["site"] for s in excluded)),
        })

    if mode == "moderate":
        # Stage 3: total >= 5 min
        excluded = [s for s in current if s["total_s"] < 300]
        current = [s for s in current if s["total_s"] >= 300]
        stages.append({
            "name": "Total scan ≥ 5 min",
            "n": len(current),
            "n_sites": len(set(s["site"] for s in current)),
            "excluded": excluded,
            "reason": "Total scan < 5 min",
            "excl_sites": dict(Counter(s["site"] for s in excluded)),
        })

    return current, stages


# ═══════════════════════════════════════════════════════════════════════
# CONSORT Flowchart
# ═══════════════════════════════════════════════════════════════════════

def draw_consort(stages: list[dict], output_path: str, atlas: str) -> None:
    """Draw CONSORT-style flowchart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n_stages = len(stages)
    fig_h = max(10, 2.5 * n_stages + 1)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    box_w, box_h = 5.0, 1.5
    excl_w, excl_h = 4.0, 1.2
    center_x = 4.5
    excl_x = 11.0
    y_top = fig_h - 1.5
    y_step = 2.3

    def _draw_box(x, y, w, h, text, color="#E3F2FD", edge="#666", fs=9):
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor=edge, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                fontfamily="sans-serif", linespacing=1.4)

    def _arrow(x1, y1, x2, y2, color="#333", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                                     linestyle=ls))

    # Title
    ax.text(7.0, y_top + 0.8,
            f"ABIDE PCP Quality Filtering for BS-NET ({atlas.upper()})",
            ha="center", va="center", fontsize=13, fontweight="bold",
            fontfamily="sans-serif")

    for i, stage in enumerate(stages):
        y = y_top - i * y_step
        is_first = i == 0
        is_last = i == n_stages - 1

        # Main box
        color = "#E8F5E9" if is_first else ("#FFF9C4" if is_last else "#E3F2FD")
        text = f"{stage['name']}\nN = {stage['n']}, {stage['n_sites']} sites"
        if is_first:
            dur = [s["total_s"] for s in stage.get("excluded", [])]  # not useful here
            text = (f"ABIDE PCP (Controls)\nN = {stage['n']}\n"
                    f"{stage['n_sites']} sites, TR = 1.5–3.0 s")
        if is_last:
            text = f"Analysis Set\nN = {stage['n']}, {stage['n_sites']} sites"

        _draw_box(center_x, y, box_w, box_h, text, color=color, fs=9.5)

        # Arrow down
        if not is_last:
            _arrow(center_x, y - box_h / 2, center_x, y - y_step + box_h / 2)

        # Exclusion box (for stages after the first)
        if i > 0 and len(stage["excluded"]) > 0:
            excl_sites = stage.get("excl_sites", {})
            site_str = ", ".join(f"{s}({n})" for s, n in
                                sorted(excl_sites.items(), key=lambda x: -x[1]))
            excl_text = (f"Excluded: n = {len(stage['excluded'])}\n"
                         f"{stage['reason']}\n{site_str}")
            _draw_box(excl_x, y, excl_w, excl_h, excl_text,
                      color="#FFEBEE", edge="#E57373", fs=8)
            _arrow(center_x + box_w / 2, y, excl_x - excl_w / 2, y,
                   color="#999", ls="--")

    # Footnotes
    ax.text(0.3, 0.5,
            (f"Short scan = first {SHORT_TRS} TRs (90–180 s depending on TR)\n"
             f"k = total TRs / {SHORT_TRS} (Spearman-Brown extrapolation factor)\n"
             f"Reference FC = full-scan FC from remaining TRs"),
            ha="left", va="bottom", fontsize=7.5,
            fontfamily="sans-serif", color="#555", linespacing=1.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"CONSORT flowchart saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Multi-seed BS-NET worker
# ═══════════════════════════════════════════════════════════════════════

def _worker(args: tuple) -> dict:
    """Process one subject × n_seeds.

    Args:
        args: (idx, sub_dict, n_seeds, correction_method)

    Returns:
        Dict with idx, sub_id, site, r_fc_raw, rho array, ci arrays.
    """
    idx, sub, n_seeds, corr_method = args
    ts_path = sub["ts_path"]
    tr = sub["tr"]

    ts = np.load(ts_path).astype(np.float64)
    valid = np.std(ts, axis=0) > 1e-8
    ts = ts[:, valid]
    n_rois = ts.shape[1]

    short_vols = int(SHORT_TRS)  # 60 TRs
    ts_short = ts[:short_vols, :]
    fc_full_vec = get_fc_matrix(ts, vectorized=True, use_shrinkage=True, fisher_z=True)
    fc_short_vec = get_fc_matrix(ts_short, vectorized=True, use_shrinkage=True, fisher_z=True)
    r_fc_raw = float(np.corrcoef(fc_short_vec, fc_full_vec)[0, 1])

    rho = np.zeros(n_seeds)
    ci_lo = np.zeros(n_seeds)
    ci_hi = np.zeros(n_seeds)

    for s in range(n_seeds):
        seed = 42 + s * 7
        config = BSNetConfig(
            n_rois=n_rois, tr=tr,
            short_duration_sec=int(short_vols * tr),
            target_duration_min=15,
            n_bootstraps=100, seed=seed,
        )
        result = run_bootstrap_prediction(
            ts_short, fc_full_vec, config,
            correction_method=corr_method,
            fisher_z_fc=True,
        )
        rho[s] = float(result.rho_hat_T)
        ci_lo[s] = float(result.ci_lower)
        ci_hi[s] = float(result.ci_upper)

    return {
        "idx": idx,
        "sub_id": sub["sub_id"],
        "site": sub["site"],
        "r_fc_raw": r_fc_raw,
        "rho": rho,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ABIDE filtered multi-seed BS-NET validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--atlas", choices=["cc200", "cc400"], default="cc200")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--filter-mode", choices=["liberal", "moderate", "strict"],
        default="strict",
        help="liberal=k>=2, moderate=k>=2+ref>=3min+total>=5min, strict=k>=2+ref>=5min",
    )
    parser.add_argument(
        "--correction-method",
        choices=["original", "fisher_z", "partial", "soft_clamp"],
        default="fisher_z",
    )
    parser.add_argument("--output-dir", default="data/abide/results")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Step 1: Load & filter ──
    logger.info(f"Loading ABIDE subjects ({args.atlas})...")
    subjects = load_subjects(args.atlas)
    logger.info(f"  Total: {len(subjects)} subjects")

    filtered, stages = apply_filters(subjects, mode=args.filter_mode)
    logger.info(f"  After filtering ({args.filter_mode}): {len(filtered)} subjects")

    # Print CONSORT summary
    for i, stage in enumerate(stages):
        if i == 0:
            logger.info(f"  Stage 0: {stage['name']} — N={stage['n']}")
        else:
            n_excl = len(stage["excluded"])
            logger.info(f"  Stage {i}: {stage['name']} — N={stage['n']} "
                        f"(excluded {n_excl})")

    # ── Step 2: CONSORT flowchart ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    consort_path = str(out_dir.parent.parent / "docs" / "figure"
                       / f"ABIDE_CONSORT_{args.filter_mode}_{args.atlas}.png")
    # Fallback if docs/figure doesn't exist
    consort_dir = Path(consort_path).parent
    if not consort_dir.exists():
        consort_path = str(out_dir / f"ABIDE_CONSORT_{args.filter_mode}_{args.atlas}.png")

    draw_consort(stages, consort_path, args.atlas)

    # ── Step 3: Multi-seed BS-NET ──
    n_subs = len(filtered)
    n_seeds = args.n_seeds
    n_workers = min(args.n_jobs, os.cpu_count() or 1)

    logger.info(f"\nRunning BS-NET: {n_subs} subjects × {n_seeds} seeds, "
                f"{n_workers} workers, method={args.correction_method}")

    worker_args = [
        (i, sub, n_seeds, args.correction_method)
        for i, sub in enumerate(filtered)
    ]

    results = [None] * n_subs
    t_start = time.time()

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        logger.warning("tqdm not installed — falling back to log-based progress")

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_worker, arg): arg[0]
                for arg in worker_args
            }
            iterator = as_completed(futures)
            if has_tqdm:
                iterator = tqdm(iterator, total=n_subs,
                                desc="BS-NET multi-seed", unit="sub")
            for future in iterator:
                idx = futures[future]
                try:
                    res = future.result()
                    results[res["idx"]] = res
                    if has_tqdm:
                        rho_mean = np.mean(res["rho"])
                        iterator.set_postfix(  # type: ignore[union-attr]
                            sub=res["sub_id"], rho=f"{rho_mean:.3f}",
                            refresh=False,
                        )
                except Exception as e:
                    logger.error(f"Worker error idx={idx}: {e}")
    else:
        iterator = worker_args
        if has_tqdm:
            iterator = tqdm(iterator, total=n_subs,
                            desc="BS-NET multi-seed", unit="sub")
        for arg in iterator:
            res = _worker(arg)
            results[res["idx"]] = res
            if has_tqdm:
                iterator.set_postfix(  # type: ignore[union-attr]
                    sub=res["sub_id"], rho=f"{np.mean(res['rho']):.3f}",
                    refresh=False,
                )

    elapsed = time.time() - t_start

    # Remove failed entries
    results = [r for r in results if r is not None]
    logger.info(f"Completed: {len(results)}/{n_subs} subjects in {elapsed:.0f}s "
                f"({elapsed / max(len(results), 1):.1f}s/sub)")

    # ── Step 4: Save results ──
    suffix = f"filtered_{args.filter_mode}"
    csv_path = out_dir / f"abide_multiseed_{args.atlas}_{n_seeds}seeds_{suffix}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow([
            "sub_id", "site", "r_fc_raw",
            "rho_hat_T_mean", "rho_hat_T_std",
            "rho_hat_T_min", "rho_hat_T_max",
            "ci_lower_mean", "ci_upper_mean",
        ])
        for res in results:
            writer.writerow([
                res["sub_id"], res["site"], f"{res['r_fc_raw']:.4f}",
                f"{np.mean(res['rho']):.4f}", f"{np.std(res['rho']):.4f}",
                f"{np.min(res['rho']):.4f}", f"{np.max(res['rho']):.4f}",
                f"{np.mean(res['ci_lo']):.4f}", f"{np.mean(res['ci_hi']):.4f}",
            ])
    logger.info(f"Multi-seed CSV: {csv_path}")

    # Summary
    r_fc_arr = np.array([r["r_fc_raw"] for r in results])
    rho_arr = np.array([np.mean(r["rho"]) for r in results])
    rho_std_arr = np.array([np.std(r["rho"]) for r in results])
    improvement = rho_arr - r_fc_arr
    n_improved = int(np.sum(improvement > 0))
    ceiling = int(np.sum(rho_arr >= 0.999))

    summary = {
        "filter_mode": args.filter_mode,
        "atlas": args.atlas,
        "correction_method": args.correction_method,
        "n_seeds": n_seeds,
        "n_total": len(subjects),
        "n_filtered": len(filtered),
        "n_success": len(results),
        "n_sites": len(set(r["site"] for r in results)),
        "stages": [
            {"name": s["name"], "n": s["n"], "n_excluded": len(s["excluded"])}
            for s in stages
        ],
        "r_fc_raw": {
            "mean": round(float(np.mean(r_fc_arr)), 4),
            "std": round(float(np.std(r_fc_arr)), 4),
        },
        "rho_hat_T": {
            "mean": round(float(np.mean(rho_arr)), 4),
            "std": round(float(np.std(rho_arr)), 4),
        },
        "improvement": {
            "mean": round(float(np.mean(improvement)), 4),
            "std": round(float(np.std(improvement)), 4),
            "pct_improved": round(n_improved / len(results) * 100, 1),
            "n_improved": n_improved,
        },
        "seed_stability": {
            "mean_std": round(float(np.mean(rho_std_arr)), 4),
        },
        "ceiling": ceiling,
        "elapsed_sec": round(elapsed, 1),
    }

    summary_path = out_dir / f"abide_filtered_{args.filter_mode}_{args.atlas}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("")
    logger.info("━" * 55)
    logger.info(f" ABIDE Filtered Validation ({args.atlas.upper()}, {args.filter_mode})")
    logger.info("━" * 55)
    logger.info(f"  Subjects: {len(results)} (from {len(subjects)} total)")
    logger.info(f"  Sites: {summary['n_sites']}")
    logger.info(f"  r_FC (raw):  {summary['r_fc_raw']['mean']:.3f} ± "
                f"{summary['r_fc_raw']['std']:.3f}")
    logger.info(f"  ρ̂T (BS-NET): {summary['rho_hat_T']['mean']:.3f} ± "
                f"{summary['rho_hat_T']['std']:.3f}")
    logger.info(f"  Δ:           {summary['improvement']['mean']:+.3f} ± "
                f"{summary['improvement']['std']:.3f}")
    logger.info(f"  Improved:    {n_improved}/{len(results)} "
                f"({summary['improvement']['pct_improved']}%)")
    logger.info(f"  Seed σ:      {summary['seed_stability']['mean_std']:.4f}")
    logger.info(f"  Ceiling:     {ceiling}")
    logger.info(f"  Elapsed:     {elapsed:.0f}s")
    logger.info("━" * 55)


if __name__ == "__main__":
    main()
