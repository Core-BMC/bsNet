#!/usr/bin/env python3
"""
BS-NET Temporal Self-Distillation (TSD) Ablation Experiment.

E0–E3 ablation을 통해 TSD 이론의 각 구성요소 기여도를 검증한다.

Experiment Design (from BS-NET_TSD_Theory_v2):
    E0: Baseline — 현재 BS-NET (bootstrap median, SB prophecy, attenuation correction)
    E1: w*_B — bootstrap ensemble FC를 명시적 distillation target으로 사용
    E2: w*_G — within-dataset GLM (short→long FC predictor, Ridge regression)
    E3: w*_B + w*_G — E1과 E2의 결합

Teacher-Student Structure (HCP-Free):
    Teacher: FC(T) = Ledoit-Wolf FC from full scan (T TR)
    Student: FC(t) = Ledoit-Wolf FC from short scan (t ≤ 120s)

Primary Dataset: ds000243 (WashU resting-state, 30min scan, N≈49)

Usage:
    # E0 baseline (현재 BS-NET)
    python src/scripts/run_tsd_ablation.py --dataset ds000243 --experiment E0

    # E1 bootstrap ensemble distillation
    python src/scripts/run_tsd_ablation.py --dataset ds000243 --experiment E1

    # E2 within-dataset GLM (subject-LOOCV)
    python src/scripts/run_tsd_ablation.py --dataset ds000243 --experiment E2

    # E3 combined
    python src/scripts/run_tsd_ablation.py --dataset ds000243 --experiment E3

    # All experiments
    python src/scripts/run_tsd_ablation.py --dataset ds000243 --experiment all

    # ABIDE secondary validation
    python src/scripts/run_tsd_ablation.py --dataset abide --experiment all

References:
    - Taxali et al. (2021): Multivariate ensemble → reliability. Cerebral Cortex.
    - Teeuw et al. (2021): Split-half measurement model. NeuroImage.
    - Guo et al. (2023): Short→long FC prediction via GLM. bioRxiv.
    - Ellis et al. (2024): SB convergence proof. Psychometrika.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.core.bootstrap import (
    block_bootstrap_indices,
    correct_attenuation,
    estimate_optimal_block_length,
    fisher_z,
    fisher_z_inv,
    spearman_brown,
)
from src.core.config import BSNetConfig
from src.data.data_loader import get_fc_matrix

logger = logging.getLogger(__name__)

ExperimentLevel = Literal["E0", "E1", "E2", "E3"]

# ============================================================================
# Data discovery
# ============================================================================

def discover_subjects_ds000243(
    cache_dir: str = "data/ds000243/timeseries_cache",
    atlas: str = "schaefer200",
) -> list[dict]:
    """ds000243 subject .npy 파일 탐색.

    파일 패턴: sub-XXX_{atlas}.npy (preprocess_ds000243.py 출력 형식)
    """
    cache_path = Path(cache_dir) / atlas
    if not cache_path.exists():
        available = [d.name for d in Path(cache_dir).iterdir() if d.is_dir()] if Path(cache_dir).exists() else []
        raise FileNotFoundError(
            f"Cache directory not found: {cache_path}. "
            f"Available atlases: {available}"
        )

    subjects = []
    for npy_file in sorted(cache_path.glob(f"sub-*_{atlas}.npy")):
        sub_id = npy_file.stem.replace(f"_{atlas}", "")
        subjects.append({"sub_id": sub_id, "ts_path": str(npy_file)})

    # Fallback: sub-*_ts.npy (XCP-D 4S 시리즈 출력 형식)
    if not subjects:
        for npy_file in sorted(cache_path.glob("sub-*_ts.npy")):
            sub_id = npy_file.stem.replace("_ts", "")
            subjects.append({"sub_id": sub_id, "ts_path": str(npy_file)})

    # Fallback: any sub-*.npy
    if not subjects:
        for npy_file in sorted(cache_path.glob("sub-*.npy")):
            sub_id = npy_file.stem.split("_")[0]
            subjects.append({"sub_id": sub_id, "ts_path": str(npy_file)})

    logger.info(f"ds000243: found {len(subjects)} subjects in {cache_path}")
    return subjects


def discover_subjects_abide(
    cache_dir: str = "data/abide/timeseries_cache",
    atlas: str = "cc200",
) -> list[dict]:
    """ABIDE PCP subject .npy 파일 탐색."""
    cache_path = Path(cache_dir) / atlas
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_path}")

    subjects = []
    for npy_file in sorted(cache_path.glob("*_*.npy")):
        sub_id = npy_file.stem
        subjects.append({"sub_id": sub_id, "ts_path": str(npy_file)})

    logger.info(f"ABIDE: found {len(subjects)} subjects in {cache_path}")
    return subjects


# ============================================================================
# TSD Experiment Core
# ============================================================================

@dataclass
class TSDResult:
    """Single-subject TSD experiment result."""

    sub_id: str
    experiment: str
    rho_hat_T: float
    r_fc_raw: float  # Corr(FC_short, FC_teacher) without any correction
    r_fc_teacher: float  # teacher quality: split-half reliability of FC_teacher
    n_tr_short: int
    n_tr_total: int
    k_factor: float
    seed: int
    # E1 추가 평가 지표 (E0/E2/E3에서는 NaN)
    rho_hat_T_vs_teacher: float = float("nan")  # E1의 fc_teacher 기준 ρ̂T (E0과 공정 비교용)
    r_fc_ensemble_teacher: float = float("nan")  # Corr(fc_ensemble, fc_teacher)


def compute_fc_vectorized(
    ts: np.ndarray, use_shrinkage: bool = True
) -> np.ndarray:
    """FC matrix 계산 후 upper-triangle 벡터 반환."""
    return get_fc_matrix(ts, vectorized=True, use_shrinkage=use_shrinkage)


def run_e0_baseline(
    ts_short: np.ndarray,
    fc_teacher: np.ndarray,
    config: BSNetConfig,
    correction_method: str = "fisher_z",
) -> float:
    """E0: 현재 BS-NET baseline.

    Bootstrap median + SB prophecy + attenuation correction.
    Teacher = full scan FC (passed as fc_teacher).
    """
    block_size = estimate_optimal_block_length(ts_short)
    t_samples = ts_short.shape[0]
    rho_hat_b = []
    empirical_prior = config.empirical_prior

    for _b in range(config.n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = ts_short[idx, :]

        fc_b = compute_fc_vectorized(ts_b, use_shrinkage=True)
        r_obs_t = np.corrcoef(fc_teacher, fc_b)[0, 1]

        # Split-half reliability
        n_split = ts_b.shape[0] // 2
        fc_h1 = compute_fc_vectorized(ts_b[:n_split], use_shrinkage=True)
        fc_h2 = compute_fc_vectorized(ts_b[n_split:], use_shrinkage=True)
        r_split = np.clip(np.corrcoef(fc_h1, fc_h2)[0, 1], 0.001, 0.999)

        rho_est = correct_attenuation(
            r_obs_t, config.reliability_coeff, r_split,
            k=config.k_factor, empirical_prior=empirical_prior,
            method=correction_method,
        )
        rho_hat_b.append(fisher_z(rho_est))

    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


def run_e1_ensemble_distill(
    ts_short: np.ndarray,
    fc_teacher: np.ndarray,
    config: BSNetConfig,
    correction_method: str = "fisher_z",
) -> tuple[float, float, float]:
    """E1: w*_B — Bootstrap ensemble FC를 명시적 distillation target으로 사용.

    Cat 1 (Statistical Distillation) + Cat 3 (Self-Distillation):
    1. B회 bootstrap → FC^(b) 집합 생성
    2. FC_ensemble = mean(FC^(b)) — 명시적 w*_B
    3. FC_ensemble을 reference로 사용하여 개별 bootstrap의 r_obs 재계산
    4. Attenuation correction + SB prophecy
    5. 별도로 fc_teacher 기준 ρ̂T도 산출 (E0과 공정 비교용)

    핵심 차이 (vs E0): E0은 fc_teacher (full scan)를 reference로 사용하지만,
    E1은 fc_ensemble (bootstrap mean)을 reference로 사용.
    fc_teacher 기준 평가는 공정 비교용으로 별도 산출.

    Returns:
        (rho_hat_T_self,      — fc_ensemble 기준 ρ̂T (self-distilled quality)
         rho_hat_T_vs_teacher, — fc_teacher 기준 ρ̂T (E0과 공정 비교용)
         r_fc_ensemble_teacher) — Corr(fc_ensemble, fc_teacher)
    """
    block_size = estimate_optimal_block_length(ts_short)
    t_samples = ts_short.shape[0]
    empirical_prior = config.empirical_prior

    # Phase 1: Bootstrap ensemble FC 생성 (w*_B)
    fc_bootstraps = []
    for _b in range(config.n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = ts_short[idx, :]
        fc_b = compute_fc_vectorized(ts_b, use_shrinkage=True)
        fc_bootstraps.append(fc_b)

    fc_ensemble = np.mean(fc_bootstraps, axis=0)  # w*_B

    # Phase 2a: fc_ensemble 기준 자기-평가 (self-distillation metric)
    rho_hat_b_self = []
    for _b in range(config.n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = ts_short[idx, :]
        fc_b = compute_fc_vectorized(ts_b, use_shrinkage=True)

        r_obs_t = np.corrcoef(fc_ensemble, fc_b)[0, 1]

        n_split = ts_b.shape[0] // 2
        fc_h1 = compute_fc_vectorized(ts_b[:n_split], use_shrinkage=True)
        fc_h2 = compute_fc_vectorized(ts_b[n_split:], use_shrinkage=True)
        r_split = np.clip(np.corrcoef(fc_h1, fc_h2)[0, 1], 0.001, 0.999)

        rho_est = correct_attenuation(
            r_obs_t, config.reliability_coeff, r_split,
            k=config.k_factor, empirical_prior=empirical_prior,
            method=correction_method,
        )
        rho_hat_b_self.append(fisher_z(rho_est))

    rho_hat_T_self = float(fisher_z_inv(np.nanmedian(rho_hat_b_self)))

    # Phase 2b: fc_teacher 기준 평가 (E0과 공정 비교용)
    rho_hat_b_teacher = []
    for _b in range(config.n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = ts_short[idx, :]
        fc_b = compute_fc_vectorized(ts_b, use_shrinkage=True)

        r_obs_t = np.corrcoef(fc_teacher, fc_b)[0, 1]

        n_split = ts_b.shape[0] // 2
        fc_h1 = compute_fc_vectorized(ts_b[:n_split], use_shrinkage=True)
        fc_h2 = compute_fc_vectorized(ts_b[n_split:], use_shrinkage=True)
        r_split = np.clip(np.corrcoef(fc_h1, fc_h2)[0, 1], 0.001, 0.999)

        rho_est = correct_attenuation(
            r_obs_t, config.reliability_coeff, r_split,
            k=config.k_factor, empirical_prior=empirical_prior,
            method=correction_method,
        )
        rho_hat_b_teacher.append(fisher_z(rho_est))

    rho_hat_T_vs_teacher = float(fisher_z_inv(np.nanmedian(rho_hat_b_teacher)))

    # fc_ensemble vs fc_teacher similarity
    r_fc_ensemble_teacher = float(np.corrcoef(fc_ensemble, fc_teacher)[0, 1])

    return rho_hat_T_self, rho_hat_T_vs_teacher, r_fc_ensemble_teacher


def run_e2_glm_distill(
    ts_short: np.ndarray,
    fc_teacher: np.ndarray,
    config: BSNetConfig,
    fc_predicted_g: np.ndarray | None = None,
    correction_method: str = "fisher_z",
) -> float:
    """E2: w*_G — Within-dataset GLM predictor 적용.

    Cat 2 (Knowledge Distillation) + Cat 5 (Temporal Distillation):
    1. fc_predicted_g는 LOOCV에서 사전 계산된 predicted long FC
    2. fc_predicted_g를 reference로 사용하여 bootstrap 평가

    Args:
        fc_predicted_g: Pre-computed predicted FC vector from Ridge LOOCV.
                        If None, falls back to E0 baseline.
    """
    if fc_predicted_g is None:
        logger.warning("fc_predicted_g not provided, falling back to E0 baseline")
        return run_e0_baseline(ts_short, fc_teacher, config, correction_method)

    # Bootstrap evaluation against fc_predicted_g
    block_size = estimate_optimal_block_length(ts_short)
    t_samples = ts_short.shape[0]
    empirical_prior = config.empirical_prior
    rho_hat_b = []

    for _b in range(config.n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = ts_short[idx, :]
        fc_b = compute_fc_vectorized(ts_b, use_shrinkage=True)

        r_obs_t = np.corrcoef(fc_predicted_g, fc_b)[0, 1]

        n_split = ts_b.shape[0] // 2
        fc_h1 = compute_fc_vectorized(ts_b[:n_split], use_shrinkage=True)
        fc_h2 = compute_fc_vectorized(ts_b[n_split:], use_shrinkage=True)
        r_split = np.clip(np.corrcoef(fc_h1, fc_h2)[0, 1], 0.001, 0.999)

        rho_est = correct_attenuation(
            r_obs_t, config.reliability_coeff, r_split,
            k=config.k_factor, empirical_prior=empirical_prior,
            method=correction_method,
        )
        rho_hat_b.append(fisher_z(rho_est))

    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


def run_e3_combined(
    ts_short: np.ndarray,
    fc_teacher: np.ndarray,
    config: BSNetConfig,
    fc_predicted_g: np.ndarray | None = None,
    correction_method: str = "fisher_z",
    alpha: float = 0.5,
) -> float:
    """E3: w*_B + w*_G — Bootstrap ensemble과 GLM predictor의 가중 결합.

    Cat 1+2+3+5 통합:
    1. w*_B: bootstrap ensemble FC (여기서 생성)
    2. fc_predicted_g: LOOCV에서 사전 계산된 predicted FC
    3. FC_combined = alpha * FC_ensemble + (1-alpha) * fc_predicted_g
    4. FC_combined를 reference로 bootstrap 평가

    Args:
        alpha: w*_B 가중치 (0=GLM only, 1=ensemble only, 0.5=equal blend)
    """
    if fc_predicted_g is None:
        logger.warning("fc_predicted_g not provided, falling back to E1")
        rho_self, _, _ = run_e1_ensemble_distill(ts_short, fc_teacher, config, correction_method)
        return rho_self

    block_size = estimate_optimal_block_length(ts_short)
    t_samples = ts_short.shape[0]
    empirical_prior = config.empirical_prior

    # Phase 1: w*_B — bootstrap ensemble
    fc_bootstraps = []
    for _b in range(config.n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = ts_short[idx, :]
        fc_b = compute_fc_vectorized(ts_b, use_shrinkage=True)
        fc_bootstraps.append(fc_b)
    fc_ensemble = np.mean(fc_bootstraps, axis=0)

    # Phase 2: Combine ensemble + GLM prediction
    fc_combined = alpha * fc_ensemble + (1 - alpha) * fc_predicted_g

    # Phase 3: Bootstrap evaluation against fc_combined
    rho_hat_b = []
    for _b in range(config.n_bootstraps):
        idx = block_bootstrap_indices(
            t_samples, block_size, n_blocks=max(1, t_samples // block_size)
        )
        ts_b = ts_short[idx, :]
        fc_b = compute_fc_vectorized(ts_b, use_shrinkage=True)

        r_obs_t = np.corrcoef(fc_combined, fc_b)[0, 1]

        n_split = ts_b.shape[0] // 2
        fc_h1 = compute_fc_vectorized(ts_b[:n_split], use_shrinkage=True)
        fc_h2 = compute_fc_vectorized(ts_b[n_split:], use_shrinkage=True)
        r_split = np.clip(np.corrcoef(fc_h1, fc_h2)[0, 1], 0.001, 0.999)

        rho_est = correct_attenuation(
            r_obs_t, config.reliability_coeff, r_split,
            k=config.k_factor, empirical_prior=empirical_prior,
            method=correction_method,
        )
        rho_hat_b.append(fisher_z(rho_est))

    return float(fisher_z_inv(np.nanmedian(rho_hat_b)))


# ============================================================================
# w*_G Training (Ridge Regression, Subject-LOOCV)
# ============================================================================

def train_w_star_g_loocv(
    subjects: list[dict],
    short_sec: int = 120,
    tr: float = 2.5,
    alpha_ridge: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Within-dataset Ridge regression: short FC → long FC (Guo 2023 설계).

    Subject-level LOOCV로 각 subject에 대한 predicted FC를 생성한다.
    정보 누수 방지: test subject는 학습에서 제외.

    메모리 최적화: Ridge coef_ (n_edges × n_edges, ~3GB/subject) 대신
    predicted FC vector (n_edges, ~160KB/subject)만 저장.

    Args:
        subjects: list of {"sub_id", "ts_path"} dicts.
        short_sec: Student duration in seconds.
        tr: Repetition time.
        alpha_ridge: Ridge regularization strength.

    Returns:
        fc_predicted_per_sub: {sub_id: predicted long FC vector} for each held-out subject.
        glm_r2_per_sub: {sub_id: prediction R²} on held-out subject.
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        raise ImportError("sklearn required for w*_G training: pip install scikit-learn")

    short_samples = int(short_sec / tr)

    # Precompute FC pairs (short, long) for all subjects
    fc_pairs = []
    valid_subs = []
    for sub_info in subjects:
        ts = np.load(sub_info["ts_path"]).astype(np.float64)
        if ts.shape[0] < short_samples + 10:
            logger.warning(f"Skipping {sub_info['sub_id']}: too short ({ts.shape[0]} TR)")
            continue

        fc_short = compute_fc_vectorized(ts[:short_samples], use_shrinkage=True)
        fc_long = compute_fc_vectorized(ts, use_shrinkage=True)
        fc_pairs.append((fc_short, fc_long))
        valid_subs.append(sub_info)

    logger.info(f"w*_G training: {len(valid_subs)} valid subjects, "
                f"n_edges={fc_pairs[0][0].shape[0]}")

    # Subject-level LOOCV — store predicted FC only (not coef_)
    fc_predicted_per_sub: dict[str, np.ndarray] = {}
    glm_r2_per_sub: dict[str, float] = {}

    for i, sub_info in enumerate(valid_subs):
        # Train on all except i
        X_train = np.array([fc_pairs[j][0] for j in range(len(valid_subs)) if j != i])
        y_train = np.array([fc_pairs[j][1] for j in range(len(valid_subs)) if j != i])

        ridge = Ridge(alpha=alpha_ridge, fit_intercept=True)
        ridge.fit(X_train, y_train)

        # Predict held-out subject
        fc_short_test = fc_pairs[i][0].reshape(1, -1)
        fc_long_test = fc_pairs[i][1]
        fc_predicted = ridge.predict(fc_short_test).flatten()

        r2 = np.corrcoef(fc_predicted, fc_long_test)[0, 1] ** 2

        # Store predicted FC only (not coef_ — saves ~3GB/subject)
        fc_predicted_per_sub[sub_info["sub_id"]] = fc_predicted
        glm_r2_per_sub[sub_info["sub_id"]] = float(r2)

        # Explicitly free the Ridge model
        del ridge, X_train, y_train

        logger.debug(f"  {sub_info['sub_id']}: R²={r2:.4f}")

    mean_r2 = np.mean(list(glm_r2_per_sub.values()))
    logger.info(f"w*_G LOOCV mean R²={mean_r2:.4f}")

    return fc_predicted_per_sub, glm_r2_per_sub


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment(
    subjects: list[dict],
    experiment: ExperimentLevel,
    short_sec: int = 120,
    tr: float = 2.5,
    n_bootstraps: int = 100,
    n_seeds: int = 10,
    correction_method: str = "fisher_z",
    fc_predicted_dict: dict[str, np.ndarray] | None = None,
) -> list[TSDResult]:
    """단일 experiment level을 전체 subject에 대해 실행."""
    short_samples = int(short_sec / tr)
    results: list[TSDResult] = []

    for sub_info in subjects:
        ts_full = np.load(sub_info["ts_path"]).astype(np.float64)

        if ts_full.shape[0] < short_samples + 10:
            logger.warning(f"Skipping {sub_info['sub_id']}: insufficient length")
            continue

        ts_short = ts_full[:short_samples]
        fc_teacher = compute_fc_vectorized(ts_full, use_shrinkage=True)

        # Teacher quality: split-half reliability
        n_half = ts_full.shape[0] // 2
        fc_h1 = compute_fc_vectorized(ts_full[:n_half], use_shrinkage=True)
        fc_h2 = compute_fc_vectorized(ts_full[n_half:], use_shrinkage=True)
        r_teacher = float(np.corrcoef(fc_h1, fc_h2)[0, 1])

        # Raw FC similarity (no BS-NET correction)
        fc_short_raw = compute_fc_vectorized(ts_short, use_shrinkage=True)
        r_fc_raw = float(np.corrcoef(fc_short_raw, fc_teacher)[0, 1])

        target_samples = ts_full.shape[0]
        k = target_samples / short_samples

        for seed in range(n_seeds):
            np.random.seed(seed)

            config = BSNetConfig(
                n_rois=ts_full.shape[1],
                tr=tr,
                short_duration_sec=short_sec,
                target_duration_min=int(target_samples * tr / 60),
                n_bootstraps=n_bootstraps,
                seed=seed,
            )

            # E1-specific extra metrics
            rho_hat_T_vs_teacher = float("nan")
            r_fc_ensemble_teacher = float("nan")

            if experiment == "E0":
                rho = run_e0_baseline(ts_short, fc_teacher, config, correction_method)
            elif experiment == "E1":
                rho, rho_hat_T_vs_teacher, r_fc_ensemble_teacher = (
                    run_e1_ensemble_distill(ts_short, fc_teacher, config, correction_method)
                )
            elif experiment == "E2":
                fc_pred = fc_predicted_dict.get(sub_info["sub_id"]) if fc_predicted_dict else None
                rho = run_e2_glm_distill(ts_short, fc_teacher, config, fc_pred, correction_method)
            elif experiment == "E3":
                fc_pred = fc_predicted_dict.get(sub_info["sub_id"]) if fc_predicted_dict else None
                rho = run_e3_combined(ts_short, fc_teacher, config, fc_pred, correction_method)
            else:
                raise ValueError(f"Unknown experiment: {experiment}")

            results.append(TSDResult(
                sub_id=sub_info["sub_id"],
                experiment=experiment,
                rho_hat_T=rho,
                r_fc_raw=r_fc_raw,
                r_fc_teacher=r_teacher,
                n_tr_short=short_samples,
                n_tr_total=target_samples,
                k_factor=k,
                seed=seed,
                rho_hat_T_vs_teacher=rho_hat_T_vs_teacher,
                r_fc_ensemble_teacher=r_fc_ensemble_teacher,
            ))

        logger.info(
            f"  {sub_info['sub_id']}: {experiment} "
            f"ρ̂T={np.mean([r.rho_hat_T for r in results if r.sub_id == sub_info['sub_id']]):.4f} "
            f"(raw r_FC={r_fc_raw:.4f})"
        )

    return results


def save_results(results: list[TSDResult], output_path: str) -> None:
    """결과를 CSV로 저장."""
    import csv

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sub_id", "experiment", "rho_hat_T", "r_fc_raw", "r_fc_teacher",
        "n_tr_short", "n_tr_total", "k_factor", "seed",
        "rho_hat_T_vs_teacher", "r_fc_ensemble_teacher",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "sub_id": r.sub_id,
                "experiment": r.experiment,
                "rho_hat_T": f"{r.rho_hat_T:.6f}",
                "r_fc_raw": f"{r.r_fc_raw:.6f}",
                "r_fc_teacher": f"{r.r_fc_teacher:.6f}",
                "n_tr_short": r.n_tr_short,
                "n_tr_total": r.n_tr_total,
                "k_factor": f"{r.k_factor:.4f}",
                "seed": r.seed,
                "rho_hat_T_vs_teacher": f"{r.rho_hat_T_vs_teacher:.6f}" if not np.isnan(r.rho_hat_T_vs_teacher) else "",
                "r_fc_ensemble_teacher": f"{r.r_fc_ensemble_teacher:.6f}" if not np.isnan(r.r_fc_ensemble_teacher) else "",
            })

    logger.info(f"Saved {len(results)} results to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BS-NET TSD Ablation Experiment (E0–E3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=["ds000243", "abide"], default="ds000243",
        help="Dataset to use (default: ds000243)",
    )
    parser.add_argument(
        "--experiment", choices=["E0", "E1", "E2", "E3", "all"], default="all",
        help="Experiment level (default: all)",
    )
    parser.add_argument("--atlas", default="schaefer200", help="Atlas name (ds000243: schaefer200/cc200/4s256parcels/...)")
    parser.add_argument("--cache-dir", default=None, help="Override timeseries cache directory")
    parser.add_argument("--short-sec", type=int, default=120, help="Short scan duration (sec)")
    parser.add_argument("--tr", type=float, default=2.5, help="TR in seconds")
    parser.add_argument("--n-bootstraps", type=int, default=100, help="Bootstrap iterations")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument("--correction-method", default="fisher_z", help="Correction method")
    parser.add_argument("--alpha-ridge", type=float, default=1.0, help="Ridge alpha for w*_G")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    # Discover subjects
    if args.dataset == "ds000243":
        cache_dir = args.cache_dir or "data/ds000243/timeseries_cache"
        subjects = discover_subjects_ds000243(cache_dir=cache_dir, atlas=args.atlas)
        default_output_dir = "data/ds000243/results"
    elif args.dataset == "abide":
        cache_dir = args.cache_dir or "data/abide/timeseries_cache"
        subjects = discover_subjects_abide(cache_dir=cache_dir, atlas=args.atlas)
        default_output_dir = "data/abide/results"
        args.tr = 2.0  # ABIDE typical TR
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    output_dir = args.output_dir or default_output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine experiments to run
    experiments: list[ExperimentLevel] = (
        ["E0", "E1", "E2", "E3"] if args.experiment == "all"
        else [args.experiment]
    )

    # Pre-train w*_G if E2 or E3 is requested
    fc_predicted_dict = None
    glm_r2_dict = None
    if "E2" in experiments or "E3" in experiments:
        logger.info("=== Training w*_G (Ridge LOOCV) ===")
        t0 = time.time()
        fc_predicted_dict, glm_r2_dict = train_w_star_g_loocv(
            subjects,
            short_sec=args.short_sec,
            tr=args.tr,
            alpha_ridge=args.alpha_ridge,
        )
        logger.info(f"w*_G training completed in {time.time() - t0:.1f}s")

        # Save GLM R² results
        import csv
        r2_path = Path(output_dir) / f"tsd_glm_r2_{args.dataset}_{args.atlas}.csv"
        with open(r2_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sub_id", "glm_r2"])
            for sub_id, r2 in sorted(glm_r2_dict.items()):
                writer.writerow([sub_id, f"{r2:.6f}"])
        logger.info(f"GLM R² saved to {r2_path}")

    # Run experiments
    all_results: list[TSDResult] = []
    for exp in experiments:
        logger.info(f"=== Running {exp} ===")
        t0 = time.time()

        results = run_experiment(
            subjects=subjects,
            experiment=exp,
            short_sec=args.short_sec,
            tr=args.tr,
            n_bootstraps=args.n_bootstraps,
            n_seeds=args.n_seeds,
            correction_method=args.correction_method,
            fc_predicted_dict=fc_predicted_dict,
        )
        all_results.extend(results)

        # Per-experiment summary
        rhos = [r.rho_hat_T for r in results]
        raws = [r.r_fc_raw for r in results]
        logger.info(
            f"{exp}: ρ̂T={np.mean(rhos):.4f}±{np.std(rhos):.4f}, "
            f"raw r_FC={np.mean(raws):.4f}±{np.std(raws):.4f}, "
            f"Δ={np.mean(rhos) - np.mean(raws):.4f}, "
            f"time={time.time() - t0:.1f}s"
        )

    # Save all results
    tag = f"{args.dataset}_{args.atlas}"
    runs_path = Path(output_dir) / f"tsd_ablation_runs_{tag}.csv"
    save_results(all_results, str(runs_path))

    # Summary table
    logger.info("\n=== TSD Ablation Summary ===")
    logger.info(f"{'Exp':>4} | {'ρ̂T mean':>10} | {'ρ̂T std':>8} | {'r_FC raw':>10} | {'Δ':>8} | {'vs_teacher':>12} | {'N':>5}")
    logger.info("-" * 75)
    for exp in experiments:
        exp_results = [r for r in all_results if r.experiment == exp]
        rhos = [r.rho_hat_T for r in exp_results]
        raws = [r.r_fc_raw for r in exp_results]
        # E1: teacher-compared ρ̂T 별도 표시
        vs_teacher = [r.rho_hat_T_vs_teacher for r in exp_results if not np.isnan(r.rho_hat_T_vs_teacher)]
        vs_str = f"{np.mean(vs_teacher):>12.4f}" if vs_teacher else f"{'—':>12}"
        logger.info(
            f"{exp:>4} | {np.mean(rhos):>10.4f} | {np.std(rhos):>8.4f} | "
            f"{np.mean(raws):>10.4f} | {np.mean(rhos) - np.mean(raws):>8.4f} | "
            f"{vs_str} | {len(exp_results):>5}"
        )

    # E0 vs E1 공정 비교 (E1의 teacher-compared metric 사용)
    e0_results = [r for r in all_results if r.experiment == "E0"]
    e1_results = [r for r in all_results if r.experiment == "E1"]
    if e0_results and e1_results:
        e0_rhos = [r.rho_hat_T for r in e0_results]
        e1_vs_teacher = [r.rho_hat_T_vs_teacher for r in e1_results if not np.isnan(r.rho_hat_T_vs_teacher)]
        if e1_vs_teacher:
            delta_e0_e1 = np.mean(e1_vs_teacher) - np.mean(e0_rhos)
            logger.info(f"\n=== E0 vs E1 Fair Comparison (both vs fc_teacher) ===")
            logger.info(f"E0 ρ̂T = {np.mean(e0_rhos):.4f} ± {np.std(e0_rhos):.4f}")
            logger.info(f"E1 ρ̂T (vs teacher) = {np.mean(e1_vs_teacher):.4f} ± {np.std(e1_vs_teacher):.4f}")
            logger.info(f"Δ(E1-E0) = {delta_e0_e1:+.4f}")
            # fc_ensemble quality
            ens_teacher = [r.r_fc_ensemble_teacher for r in e1_results if not np.isnan(r.r_fc_ensemble_teacher)]
            if ens_teacher:
                logger.info(f"Corr(fc_ensemble, fc_teacher) = {np.mean(ens_teacher):.4f} ± {np.std(ens_teacher):.4f}")

    # Save summary
    summary_path = Path(output_dir) / f"tsd_ablation_summary_{tag}.csv"
    import csv
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "rho_hat_T_mean", "rho_hat_T_std",
            "r_fc_raw_mean", "delta", "n_obs",
            "rho_hat_T_vs_teacher_mean", "r_fc_ensemble_teacher_mean",
        ])
        for exp in experiments:
            exp_results = [r for r in all_results if r.experiment == exp]
            rhos = [r.rho_hat_T for r in exp_results]
            raws = [r.r_fc_raw for r in exp_results]
            vs_teacher = [r.rho_hat_T_vs_teacher for r in exp_results if not np.isnan(r.rho_hat_T_vs_teacher)]
            ens_teacher = [r.r_fc_ensemble_teacher for r in exp_results if not np.isnan(r.r_fc_ensemble_teacher)]
            writer.writerow([
                exp,
                f"{np.mean(rhos):.6f}",
                f"{np.std(rhos):.6f}",
                f"{np.mean(raws):.6f}",
                f"{np.mean(rhos) - np.mean(raws):.6f}",
                len(exp_results),
                f"{np.mean(vs_teacher):.6f}" if vs_teacher else "",
                f"{np.mean(ens_teacher):.6f}" if ens_teacher else "",
            ])
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
