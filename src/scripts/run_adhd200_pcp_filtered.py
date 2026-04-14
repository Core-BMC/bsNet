#!/usr/bin/env python3
"""ADHD-200 PCP filtered multi-seed BS-NET validation.

Quality-filtered ADHD-200 analysis (N=768 → strict filter → N≈399):
  Stage 1: Known DX only (exclude Unknown/test-set subjects)
  Stage 2: k ≥ 2.0  (reference ≥ short scan)
  Stage 3: Reference ≥ 5 min (Noble et al. 2019 criterion)

Runs multi-seed BS-NET (Fisher z) on the filtered subset and saves:
  - CONSORT flowchart PNG
  - Filtered multi-seed CSV
  - Filtering summary JSON

Usage:
    # Default: CC200, 10 seeds, 8 workers, Fisher z, strict
    PYTHONPATH=. python src/scripts/run_adhd200_pcp_filtered.py

    # Liberal filter (k>=2 only)
    PYTHONPATH=. python src/scripts/run_adhd200_pcp_filtered.py --filter-mode liberal

    # Fewer seeds for quick test
    PYTHONPATH=. python src/scripts/run_adhd200_pcp_filtered.py --n-seeds 3 --max-subjects 20
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

DATA_DIR = Path("data/adhd/pcp")
RESULTS_DIR = DATA_DIR / "results"


# ═══════════════════════════════════════════════════════════════════════
# Subject loading & filtering
# ═══════════════════════════════════════════════════════════════════════

def load_subjects() -> list[dict]:
    """Load ADHD-200 PCP subject metadata from convert output JSON.

    Returns:
        List of dicts with keys: sub_id, site, group, dx, age, sex, tr,
        n_trs, n_rois, n_sessions, total_s, ts_path, + computed k, ref_s, short_s.
    """
    json_path = RESULTS_DIR / "adhd200_subjects_cc200.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} not found. Run convert_adhd200_pcp.py first."
        )

    with open(json_path) as f:
        raw = json.load(f)

    subjects = []
    for s in raw:
        ts_path = s["ts_path"]
        if not Path(ts_path).exists():
            continue

        n_trs = s["n_trs"]
        tr = s["tr"]

        s["short_s"] = SHORT_TRS * tr
        s["ref_s"] = (n_trs - SHORT_TRS) * tr
        s["k"] = n_trs / SHORT_TRS
        subjects.append(s)

    return subjects


def apply_filters(
    subjects: list[dict],
    mode: str = "strict",
) -> tuple[list[dict], list[dict]]:
    """Apply progressive quality filters.

    Args:
        subjects: Full subject list (all 768).
        mode: 'liberal' (known DX + k>=2), 'moderate' (+ ref>=3min),
              'strict' (+ ref>=5min).

    Returns:
        (filtered_subjects, consort_stages).
    """
    stages = []
    current = subjects.copy()

    stages.append({
        "name": "Converted",
        "n": len(current),
        "n_sites": len(set(s["site"] for s in current)),
        "excluded": [],
        "reason": "",
    })

    # Stage 1: Known DX only (exclude Unknown = test set without labels)
    excluded = [s for s in current if s["group"] not in ("Control", "ADHD")]
    current = [s for s in current if s["group"] in ("Control", "ADHD")]
    stages.append({
        "name": "Known DX",
        "n": len(current),
        "n_sites": len(set(s["site"] for s in current)),
        "excluded": excluded,
        "reason": "Unknown DX (test set)",
        "excl_sites": dict(Counter(s["site"] for s in excluded)),
    })

    # Stage 2: k >= 2
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

    return current, stages


# ═══════════════════════════════════════════════════════════════════════
# BS-NET worker
# ═══════════════════════════════════════════════════════════════════════

def _worker(args: tuple) -> dict | None:
    """Process one subject × n_seeds."""
    (sub, n_seeds, n_bootstraps, correction_method) = args

    ts_path = sub["ts_path"]
    tr = sub["tr"]

    try:
        ts = np.load(ts_path)
    except Exception:
        return None

    n_trs = ts.shape[0]
    n_rois = ts.shape[1]
    short_trs = SHORT_TRS

    if n_trs <= short_trs + 10:
        return None

    # Short / reference split
    ts_short = ts[:short_trs]
    ts_ref = ts[short_trs:]

    # get_fc_matrix returns vectorized upper triangle (1D)
    fc_short_vec = get_fc_matrix(ts_short)
    fc_ref_vec = get_fc_matrix(ts_ref)

    # Raw FC correlation
    r_fc_raw = float(np.corrcoef(fc_short_vec, fc_ref_vec)[0, 1])

    # Multi-seed BS-NET
    rho_values = []
    ci_lowers = []
    ci_uppers = []

    for seed in range(n_seeds):
        cfg = BSNetConfig(
            n_rois=n_rois, tr=tr,
            n_bootstraps=n_bootstraps,
            short_duration_sec=int(short_trs * tr),
            target_duration_min=15,
            seed=seed * 1000 + 42,
        )
        result = run_bootstrap_prediction(
            ts_short, fc_ref_vec, cfg,
            correction_method=correction_method,
            fisher_z_fc=True,
        )
        rho_values.append(float(result.rho_hat_T))
        ci_lowers.append(float(result.ci_lower))
        ci_uppers.append(float(result.ci_upper))

    return {
        "sub_id": sub["sub_id"],
        "site": sub["site"],
        "group": sub["group"],
        "age": sub.get("age", -1),
        "n_trs": n_trs,
        "n_rois": n_rois,
        "n_sessions": sub.get("n_sessions", 1),
        "tr": tr,
        "r_fc_raw": round(r_fc_raw, 6),
        "rho_hat_T_mean": round(float(np.mean(rho_values)), 6),
        "rho_hat_T_std": round(float(np.std(rho_values)), 6),
        "rho_hat_T_min": round(float(np.min(rho_values)), 6),
        "rho_hat_T_max": round(float(np.max(rho_values)), 6),
        "ci_lower_mean": round(float(np.mean(ci_lowers)), 6),
        "ci_upper_mean": round(float(np.mean(ci_uppers)), 6),
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ADHD-200 PCP filtered multi-seed BS-NET validation",
    )
    parser.add_argument("--filter-mode", default="strict",
                        choices=["liberal", "moderate", "strict"])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-bootstraps", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--correction-method", default="fisher_z",
                        choices=["original", "fisher_z", "partial", "soft_clamp"])
    parser.add_argument("--max-subjects", type=int, default=0,
                        help="Limit subjects for testing (0=all)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load & filter ──
    logger.info("Loading ADHD-200 PCP subjects...")
    all_subjects = load_subjects()
    logger.info(f"Total subjects with timeseries: {len(all_subjects)}")

    filtered, stages = apply_filters(all_subjects, mode=args.filter_mode)
    logger.info(f"After {args.filter_mode} filter: {len(filtered)} subjects")

    for stg in stages:
        n_excl = len(stg.get("excluded", []))
        logger.info(f"  {stg['name']}: N={stg['n']} (−{n_excl})")

    if args.max_subjects > 0:
        filtered = filtered[:args.max_subjects]
        logger.info(f"Limited to {len(filtered)} subjects")

    if not filtered:
        logger.error("No subjects after filtering!")
        return

    # ── Multi-seed BS-NET ──
    n_seeds = args.n_seeds
    n_workers = min(args.n_workers, len(filtered))
    correction = args.correction_method

    logger.info(f"\nRunning BS-NET: {len(filtered)} subjects × {n_seeds} seeds, "
                f"correction={correction}, workers={n_workers}")

    worker_args = [
        (sub, n_seeds, args.n_bootstraps, correction)
        for sub in filtered
    ]

    results = []
    t0 = time.time()

    try:
        from tqdm import tqdm
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, wa): wa[0]["sub_id"] for wa in worker_args}
            pbar = tqdm(total=len(futures), desc="BS-NET", unit="sub")
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    results.append(res)
                pbar.update(1)
                pbar.set_postfix(ok=len(results))
            pbar.close()
    except ImportError:
        logger.info("(Install tqdm for progress bar)")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, wa): wa[0]["sub_id"] for wa in worker_args}
            for i, future in enumerate(as_completed(futures)):
                res = future.result()
                if res is not None:
                    results.append(res)
                if (i + 1) % 50 == 0:
                    logger.info(f"  Progress: {i + 1}/{len(futures)} (ok={len(results)})")

    elapsed = time.time() - t0
    results.sort(key=lambda r: r["sub_id"])

    # ── Save CSV ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"adhd200_multiseed_cc200_{n_seeds}seeds_filtered_{args.filter_mode}.csv"
    fieldnames = [
        "sub_id", "site", "group", "age", "n_trs", "n_rois", "n_sessions", "tr",
        "r_fc_raw", "rho_hat_T_mean", "rho_hat_T_std", "rho_hat_T_min", "rho_hat_T_max",
        "ci_lower_mean", "ci_upper_mean",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ── Summary statistics ──
    r_fc = np.array([r["r_fc_raw"] for r in results])
    rho = np.array([r["rho_hat_T_mean"] for r in results])
    improvement = rho - r_fc
    n_improved = int(np.sum(improvement > 0))
    ceiling = int(np.sum(rho >= 0.999))

    # Group stats
    group_stats = {}
    for g in sorted(set(r["group"] for r in results)):
        g_results = [r for r in results if r["group"] == g]
        g_rfc = np.array([r["r_fc_raw"] for r in g_results])
        g_rho = np.array([r["rho_hat_T_mean"] for r in g_results])
        group_stats[g] = {
            "n": len(g_results),
            "r_fc_raw": {"mean": round(float(np.mean(g_rfc)), 4), "std": round(float(np.std(g_rfc)), 4)},
            "rho_hat_T": {"mean": round(float(np.mean(g_rho)), 4), "std": round(float(np.std(g_rho)), 4)},
            "improvement": {"mean": round(float(np.mean(g_rho - g_rfc)), 4)},
        }

    # Site stats
    site_stats = {}
    for site in sorted(set(r["site"] for r in results)):
        s_results = [r for r in results if r["site"] == site]
        s_rho = np.array([r["rho_hat_T_mean"] for r in s_results])
        site_stats[site] = {
            "n": len(s_results),
            "rho_hat_T_mean": round(float(np.mean(s_rho)), 4),
            "rho_hat_T_std": round(float(np.std(s_rho)), 4),
        }

    summary = {
        "filter_mode": args.filter_mode,
        "correction_method": correction,
        "n_seeds": n_seeds,
        "n_total": len(all_subjects),
        "n_filtered": len(filtered),
        "n_success": len(results),
        "n_sites": len(set(r["site"] for r in results)),
        "stages": [
            {"name": s["name"], "n": s["n"], "n_excluded": len(s.get("excluded", []))}
            for s in stages
        ],
        "r_fc_raw": {"mean": round(float(np.mean(r_fc)), 4), "std": round(float(np.std(r_fc)), 4)},
        "rho_hat_T": {"mean": round(float(np.mean(rho)), 4), "std": round(float(np.std(rho)), 4)},
        "improvement": {
            "mean": round(float(np.mean(improvement)), 4),
            "std": round(float(np.std(improvement)), 4),
            "pct_improved": round(100 * n_improved / len(results), 1),
            "n_improved": n_improved,
        },
        "seed_stability": {
            "mean_std": round(float(np.mean([r["rho_hat_T_std"] for r in results])), 4),
        },
        "ceiling": ceiling,
        "group_stats": group_stats,
        "site_stats": site_stats,
        "elapsed_sec": round(elapsed, 1),
    }

    summary_path = RESULTS_DIR / f"adhd200_filtered_{args.filter_mode}_cc200_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print results ──
    logger.info(f"\n{'=' * 60}")
    logger.info(f"ADHD-200 PCP BS-NET Results ({args.filter_mode} filter)")
    logger.info(f"{'=' * 60}")
    logger.info(f"  N = {len(results)} ({args.filter_mode} filter, {correction})")
    logger.info(f"  r_FC:   {np.mean(r_fc):.4f} ± {np.std(r_fc):.4f}")
    logger.info(f"  ρ̂T:    {np.mean(rho):.4f} ± {np.std(rho):.4f}")
    logger.info(f"  Δ:      {np.mean(improvement):.4f} ± {np.std(improvement):.4f}")
    logger.info(f"  Improved: {n_improved}/{len(results)} ({100*n_improved/len(results):.1f}%)")
    logger.info(f"  Ceiling: {ceiling}")
    logger.info(f"  Seed σ: {np.mean([r['rho_hat_T_std'] for r in results]):.4f}")
    logger.info(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"\nGroup stats:")
    for g, gs in group_stats.items():
        logger.info(f"  {g}: N={gs['n']}, ρ̂T={gs['rho_hat_T']['mean']:.4f}±{gs['rho_hat_T']['std']:.4f}")
    logger.info(f"\nSite stats:")
    for site, ss in site_stats.items():
        logger.info(f"  {site}: N={ss['n']}, ρ̂T={ss['rho_hat_T_mean']:.4f}±{ss['rho_hat_T_std']:.4f}")
    logger.info(f"\n  CSV: {csv_path}")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
