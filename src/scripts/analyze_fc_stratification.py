"""FC Stratification Analysis: zero-inflation 및 연결 강도별 신뢰도 분석.

ρ̂T ≈ 0.8이 near-zero FC pair의 영팽창(zero inflation)에 의한 과대추정인지 검증.

분석 3종:
  Analysis 1 — FC 값 분포 시각화
      전체 ROI 쌍의 FC 값 분포 및 |FC| < 0.1 비율 확인.

  Analysis 2 — Quartile 층화 분석
      참조 FC 강도별 4분위(Q1–Q4)로 분류 후 각 층에서 short-scan vs. reference
      Spearman 상관 계산.  Q1(최약) 층이 Q4(최강)와 비슷하면 zero inflation 의심.

  Analysis 3 — 임계값 제거 분석
      |ref FC| > t (t = 0.0, 0.1, 0.2, 0.3) 조건에서 pair를 제한하고
      r_FC 재산출.  임계값 증가에 따라 r_FC가 크게 변하면 zero inflation 확인.

사용법:
    python src/scripts/analyze_fc_stratification.py \\
        --atlas 4s256parcels \\
        --short-sec 120 \\
        --n-subjects 20 \\
        --n-seeds 5

출력:
    artifacts/reports/fc_stratification_{atlas}_{short_sec}s.png
    artifacts/reports/fc_stratification_{atlas}_{short_sec}s.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.visualization.style import (
    CONDITION_PALETTE,
    FONT_AXIS,
    FONT_PANEL,
    FONT_TICK,
    apply_bsnet_theme,
    save_figure,
)

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────────
CACHE_DIR   = Path("data/ds000243/timeseries_cache_xcpd")
OUTPUT_DIR  = Path("artifacts/reports")
TR          = 2.5   # ds000243 TR (초)
TARGET_SEC  = 900   # 15분 참조 기준
THRESHOLDS  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_QUARTILES = 4
C_RAW   = CONDITION_PALETTE["raw"]    # Amber
C_BSNET = CONDITION_PALETTE["bsnet"]  # Blue
C_REF   = CONDITION_PALETTE["reference"]  # Gray


# ── 데이터 로딩 ────────────────────────────────────────────────────────────────
def _load_subjects(
    atlas: str,
    n_subjects: int,
    min_ref_sec: float = TARGET_SEC,
) -> list[dict]:
    """Load timeseries_cache_xcpd/{atlas}/*.npy files.

    Args:
        atlas: Atlas name (e.g. '4s256parcels').
        n_subjects: Max subjects to load (0 = all).
        min_ref_sec: Minimum total scan duration for reference FC (seconds).

    Returns:
        List of dicts: sub_id, ts (n_vols, n_rois), n_vols, total_sec.
    """
    cache = CACHE_DIR / atlas
    npy_files = sorted(cache.glob("*.npy"))
    if n_subjects > 0:
        npy_files = npy_files[:n_subjects]

    subjects = []
    for fp in npy_files:
        ts = np.load(fp)           # (n_vols, n_rois)
        n_vols, n_rois = ts.shape
        total_sec = n_vols * TR
        if total_sec < min_ref_sec:
            logger.debug(f"SKIP {fp.stem}: {total_sec:.0f}s < {min_ref_sec:.0f}s")
            continue
        subjects.append({
            "sub_id": fp.stem.split("_")[0],
            "ts": ts.astype(np.float64),
            "n_vols": n_vols,
            "total_sec": total_sec,
        })

    logger.info(f"Loaded {len(subjects)} subjects (atlas={atlas})")
    return subjects


# ── FC 계산 ────────────────────────────────────────────────────────────────────
def _pearson_fc(ts: np.ndarray) -> np.ndarray:
    """Compute Pearson FC vector (upper triangle) from timeseries.

    Zero-variance ROIs (constant signal, common in short scans) produce NaN
    in corrcoef; these are replaced with 0.0 before returning.

    Args:
        ts: (n_vols, n_rois) timeseries array.

    Returns:
        1D array of FC values, length = n_rois*(n_rois-1)/2.
    """
    fc = np.corrcoef(ts.T)                   # (n_rois, n_rois)
    fc = np.nan_to_num(fc, nan=0.0)          # zero-variance ROI → FC=0
    i, j = np.triu_indices_from(fc, k=1)
    return fc[i, j]


def _compute_fc_pair(
    ts: np.ndarray,
    short_vols: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """단기 FC 벡터와 참조 FC 벡터 반환.

    Args:
        ts: (n_vols, n_rois) 전체 시계열.
        short_vols: 단기 스캔 볼륨 수.
        seed: 무작위 시작 위치 시드.

    Returns:
        (fc_short, fc_ref): 각 FC 벡터 (n_pairs,).
    """
    rng = np.random.default_rng(seed)
    max_start = ts.shape[0] - short_vols
    start = int(rng.integers(0, max_start + 1))
    ts_short = ts[start: start + short_vols]
    return _pearson_fc(ts_short), _pearson_fc(ts)


# ── Analysis 1: FC 분포 ────────────────────────────────────────────────────────
def analysis_fc_distribution(
    subjects: list[dict],
    ax: plt.Axes,
) -> dict:
    """전체 피험자 참조 FC 분포 히스토그램.

    Args:
        subjects: 피험자 리스트.
        ax: matplotlib Axes.

    Returns:
        Summary statistics dict.
    """
    all_fc = []
    for sub in subjects:
        fc_ref = _pearson_fc(sub["ts"])
        all_fc.append(fc_ref)
    all_fc = np.concatenate(all_fc)

    near_zero_05  = np.mean(np.abs(all_fc) < 0.05) * 100
    near_zero_10  = np.mean(np.abs(all_fc) < 0.10) * 100
    near_zero_20  = np.mean(np.abs(all_fc) < 0.20) * 100
    positive_frac = np.mean(all_fc > 0.10) * 100

    ax.hist(all_fc, bins=80, color=C_BSNET, alpha=0.75, edgecolor="white", lw=0.4)
    ax.axvline(0, color="black", lw=1.0, ls="--")
    ax.axvline( 0.10, color=C_RAW, lw=1.2, ls=":", label="|FC|=0.10")
    ax.axvline(-0.10, color=C_RAW, lw=1.2, ls=":")

    txt = (
        f"|FC|<0.05: {near_zero_05:.1f}%\n"
        f"|FC|<0.10: {near_zero_10:.1f}%\n"
        f"|FC|<0.20: {near_zero_20:.1f}%\n"
        f"FC>0.10:   {positive_frac:.1f}%"
    )
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5,
            bbox=dict(fc="white", alpha=0.8, ec="none"))

    ax.set_xlabel("FC value (Pearson r)", **FONT_AXIS)
    ax.set_ylabel("Count", **FONT_AXIS)
    ax.set_title(
        f"A. FC Value Distribution\n(N={len(subjects)}, ref scan)",
        **FONT_PANEL,
    )
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=8.5)

    return {
        "n_subjects": len(subjects),
        "n_pairs": len(all_fc),
        "near_zero_05_pct": round(near_zero_05, 2),
        "near_zero_10_pct": round(near_zero_10, 2),
        "near_zero_20_pct": round(near_zero_20, 2),
        "positive_10_pct": round(positive_frac, 2),
        "fc_mean": round(float(np.mean(all_fc)), 4),
        "fc_sd": round(float(np.std(all_fc)), 4),
    }


# ── Analysis 2: Quartile 층화 분석 ────────────────────────────────────────────
def analysis_quartile_stratification(
    subjects: list[dict],
    short_sec: int,
    seeds: list[int],
    ax: plt.Axes,
) -> list[dict]:
    """참조 FC 강도별 4분위 층화 후 단기-참조 상관 분석.

    Args:
        subjects: 피험자 리스트.
        short_sec: 단기 스캔 시간(초).
        seeds: 난수 시드 리스트.
        ax: matplotlib Axes.

    Returns:
        List of dicts per quartile: q_label, r_fc_mean, r_fc_sd, n_pairs.
    """
    short_vols = int(short_sec / TR)

    # 피험자 × 시드 × 쌍 → 누적
    q_corrs: dict[int, list[float]] = {q: [] for q in range(N_QUARTILES)}

    for sub in subjects:
        fc_ref = _pearson_fc(sub["ts"])
        # Quartile 경계값 (참조 FC 절대값 기준)
        abs_ref = np.abs(fc_ref)
        q_bounds = np.quantile(abs_ref, np.linspace(0, 1, N_QUARTILES + 1))

        for seed in seeds:
            fc_short, _ = _compute_fc_pair(sub["ts"], short_vols, seed)

            for q in range(N_QUARTILES):
                lo, hi = q_bounds[q], q_bounds[q + 1]
                if q == N_QUARTILES - 1:
                    mask = (abs_ref >= lo) & (abs_ref <= hi)
                else:
                    mask = (abs_ref >= lo) & (abs_ref < hi)
                if mask.sum() < 10:
                    continue
                r, _ = spearmanr(fc_short[mask], fc_ref[mask])
                q_corrs[q].append(r)

    labels = [f"Q{q+1}\n({['Weakest','Weak','Strong','Strongest'][q]})" for q in range(N_QUARTILES)]
    means = [float(np.mean(q_corrs[q])) if q_corrs[q] else np.nan
             for q in range(N_QUARTILES)]
    sds   = [float(np.std(q_corrs[q])) if q_corrs[q] else np.nan
             for q in range(N_QUARTILES)]

    x = np.arange(N_QUARTILES)
    bars = ax.bar(x, means, color=[C_REF, C_REF, C_BSNET, C_BSNET],
                  alpha=0.80, width=0.55, edgecolor="white")
    ax.errorbar(x, means, yerr=sds, fmt="none",
                ecolor="#444444", elinewidth=1.3, capsize=4)

    # 값 표시
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8.5)

    # 전체 평균선
    all_r = [v for vals in q_corrs.values() for v in vals]
    ax.axhline(np.mean(all_r), color="#888888", lw=1.2, ls="--",
               label=f"Overall mean r={np.mean(all_r):.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_xlabel("|Reference FC| strength quartile", **FONT_AXIS)
    ax.set_ylabel("Spearman r (short vs. reference FC)", **FONT_AXIS)
    ax.set_title(
        f"B. Quartile Stratification\n(short={short_sec}s, {len(seeds)} seeds)",
        **FONT_PANEL,
    )
    ax.legend(fontsize=8.5)

    return [
        {
            "quartile": q + 1,
            "label": labels[q],
            "r_fc_mean": round(means[q], 4) if not np.isnan(means[q]) else None,
            "r_fc_sd": round(sds[q], 4) if not np.isnan(sds[q]) else None,
            "n_obs": len(q_corrs[q]),
        }
        for q in range(N_QUARTILES)
    ]


# ── Analysis 3: 임계값 제거 분석 ──────────────────────────────────────────────
def analysis_threshold_sweep(
    subjects: list[dict],
    short_sec: int,
    seeds: list[int],
    ax: plt.Axes,
) -> list[dict]:
    """|ref FC| > t 조건에서 pair 제한 후 r_FC 재산출.

    Args:
        subjects: 피험자 리스트.
        short_sec: 단기 스캔 시간(초).
        seeds: 난수 시드 리스트.
        ax: matplotlib Axes.

    Returns:
        List of dicts per threshold: threshold, r_fc_mean, r_fc_sd, n_pairs_pct.
    """
    short_vols = int(short_sec / TR)

    thresh_results: dict[float, list[float]] = {t: [] for t in THRESHOLDS}
    thresh_npairs:  dict[float, list[float]] = {t: [] for t in THRESHOLDS}

    for sub in subjects:
        fc_ref = _pearson_fc(sub["ts"])
        abs_ref = np.abs(fc_ref)
        n_total = len(fc_ref)

        for seed in seeds:
            fc_short, _ = _compute_fc_pair(sub["ts"], short_vols, seed)

            for t in THRESHOLDS:
                mask = abs_ref > t
                n_kept = mask.sum()
                if n_kept < 20:
                    continue
                r, _ = spearmanr(fc_short[mask], fc_ref[mask])
                thresh_results[t].append(r)
                thresh_npairs[t].append(n_kept / n_total * 100)

    means      = [np.mean(thresh_results[t]) if thresh_results[t] else np.nan
                  for t in THRESHOLDS]
    sds        = [np.std(thresh_results[t]) if thresh_results[t] else np.nan
                  for t in THRESHOLDS]
    pct_kept   = [np.mean(thresh_npairs[t]) if thresh_npairs[t] else np.nan
                  for t in THRESHOLDS]

    ax2 = ax.twinx()
    ax2.bar(THRESHOLDS, pct_kept, width=0.025, color="#dddddd",
            alpha=0.5, label="Pairs retained (%)", zorder=1)
    ax2.set_ylabel("Pairs retained (%)", fontsize=9.5, color="#999999")
    ax2.tick_params(axis="y", labelsize=9, colors="#999999")
    ax2.set_ylim(0, 130)

    ax.fill_between(THRESHOLDS,
                    [m - s for m, s in zip(means, sds)],
                    [m + s for m, s in zip(means, sds)],
                    color=C_BSNET, alpha=0.15, zorder=2)
    ax.plot(THRESHOLDS, means, color=C_BSNET, lw=2.5,
            marker="o", ms=7, label="r_FC (mean±SD)", zorder=3)
    ax.errorbar(THRESHOLDS, means, yerr=sds, fmt="none",
                ecolor=C_BSNET, elinewidth=1.2, capsize=4, zorder=3)

    ax.set_xlim(-0.01, 0.32)
    ax.set_ylim(0.4, 1.05)
    ax.set_xticks(THRESHOLDS)
    ax.set_xticklabels([str(t) for t in THRESHOLDS], fontsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.set_xlabel("|Reference FC| threshold t", **FONT_AXIS)
    ax.set_ylabel("Spearman r (short vs. reference FC)", **FONT_AXIS)
    ax.set_title(
        f"C. Threshold Exclusion Analysis\n(short={short_sec}s, {len(seeds)} seeds)",
        **FONT_PANEL,
    )
    ax.legend(fontsize=8.5, loc="lower left")

    ax.text(0.0, means[0] + 0.02, f"All\n{means[0]:.3f}",
            ha="center", va="bottom", fontsize=8, color=C_BSNET)

    return [
        {
            "threshold": t,
            "r_fc_mean": round(means[i], 4) if not np.isnan(means[i]) else None,
            "r_fc_sd": round(sds[i], 4) if not np.isnan(sds[i]) else None,
            "pct_pairs_kept": round(pct_kept[i], 1) if not np.isnan(pct_kept[i]) else None,
        }
        for i, t in enumerate(THRESHOLDS)
    ]


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="FC Stratification Analysis: zero-inflation and connection-strength-stratified reliability"
    )
    parser.add_argument("--atlas", default="4s256parcels",
                        help="Atlas name (default: 4s256parcels)")
    parser.add_argument("--short-sec", type=int, default=120,
                        help="Short scan duration in seconds (default: 120)")
    parser.add_argument("--n-subjects", type=int, default=20,
                        help="Max subjects (default: 20, 0=all)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of bootstrap seeds (default: 5)")
    parser.add_argument("--min-ref-sec", type=float, default=TARGET_SEC,
                        help=(
                            "Minimum total scan duration for reference FC in seconds "
                            f"(default: {TARGET_SEC}). Use 600 to match duration sweep inclusion criterion."
                        ))
    args = parser.parse_args()

    apply_bsnet_theme()
    seeds = list(range(42, 42 + args.n_seeds))

    subjects = _load_subjects(args.atlas, args.n_subjects, min_ref_sec=args.min_ref_sec)
    if not subjects:
        logger.error("No subject data found.")
        return

    fig = plt.figure(figsize=(17, 5.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    fig.suptitle(
        f"FC Stratification Analysis — ds000243 XCP-D "
        f"({args.atlas}, N={len(subjects)}, short={args.short_sec}s)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    dist_stats   = analysis_fc_distribution(subjects, axes[0])
    quartile_res = analysis_quartile_stratification(
        subjects, args.short_sec, seeds, axes[1]
    )
    thresh_res   = analysis_threshold_sweep(
        subjects, args.short_sec, seeds, axes[2]
    )

    plt.tight_layout(pad=2.5)

    out_name = f"fc_stratification_{args.atlas}_{args.short_sec}s.png"
    out_path = save_figure(fig, out_name)
    print(f"\nSaved: {out_path}")
    plt.close(fig)

    # CSV 저장
    csv_path = OUTPUT_DIR / f"fc_stratification_{args.atlas}_{args.short_sec}s.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "analysis", "label", "value_mean", "value_sd", "note"
        ])
        writer.writeheader()
        writer.writerow({
            "analysis": "distribution",
            "label": "near_zero_10_pct",
            "value_mean": dist_stats["near_zero_10_pct"],
            "value_sd": "",
            "note": "% of pairs with |FC|<0.10",
        })
        for row in quartile_res:
            writer.writerow({
                "analysis": "quartile",
                "label": row["label"].replace("\n", " "),
                "value_mean": row["r_fc_mean"],
                "value_sd": row["r_fc_sd"],
                "note": f"n_obs={row['n_obs']}",
            })
        for row in thresh_res:
            writer.writerow({
                "analysis": "threshold",
                "label": f"t={row['threshold']}",
                "value_mean": row["r_fc_mean"],
                "value_sd": row["r_fc_sd"],
                "note": f"kept={row['pct_pairs_kept']}%",
            })
    print(f"CSV saved: {csv_path}")

    # 콘솔 요약
    print(f"\n{'='*55}")
    print("FC Distribution")
    print(f"  |FC|<0.10: {dist_stats['near_zero_10_pct']}%")
    print(f"  FC>0.10:   {dist_stats['positive_10_pct']}%")
    def _fmt(v: float | None) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    print(f"\nQuartile Stratification (short={args.short_sec}s)")
    for row in quartile_res:
        label = row['label'].replace(chr(10), ' ')
        print(f"  {label}: r={_fmt(row['r_fc_mean'])}±{_fmt(row['r_fc_sd'])}")
    print(f"\nThreshold Exclusion Analysis")
    for row in thresh_res:
        kept = f"{row['pct_pairs_kept']:.0f}%" if row['pct_pairs_kept'] is not None else "N/A"
        print(f"  t={row['threshold']:.2f} ({kept} kept): "
              f"r={_fmt(row['r_fc_mean'])}±{_fmt(row['r_fc_sd'])}")
    print("="*55)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
