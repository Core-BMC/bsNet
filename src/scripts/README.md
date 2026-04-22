# BS-NET Scripts Index

스크립트를 **10개 카테고리**로 분류한 인덱스. 각 스크립트의 용도, CLI, 입출력을 명시한다.

---

## 카테고리 요약

| # | 카테고리 | 스크립트 수 | 설명 |
|---|----------|------------|------|
| 1 | Validation (실증 검증) | 10+ | ABIDE, ADHD, fMRIPrep, Keane 기반 실데이터 검증 + 분류 |
| 2 | Defense Experiments (방어 실험) | 8 | Track A–G 방어 실험 + failure analysis |
| 3 | Data Acquisition (데이터 수집/변환) | 5 | OpenNeuro 인덱싱, 다운로드, 형식 변환 |
| 4 | Preprocessing (전처리) | 3 | ds007535, ds000243, setup_and_preprocess |
| 5 | Convergence & Ablation | 3 | Convergence validation, progressive ablation, ABIDE duration sweep |
| 6 | Downstream Analysis | 3 | Downstream suite, reliability-aware clustering/classification |
| 7 | Visualization (시각화) | 2+17 | src/scripts/ plotting + src/visualization/ 논문 figure |
| 8 | Simulation (시뮬레이션) | 2 | Synthetic baseline + duration sweep |
| 9 | Utility | 2 | Atlas inspection, FC threshold visualization |
| 10 | Pipeline Orchestration (파이프라인) | 14 (.sh) | fMRIPrep, XCP-D, Keane streaming, 환경 설정 |
| 11 | TSD (Temporal Self-Distillation) | 1 | E0–E3 ablation 실험 |

---

## 1. Validation (실증 검증)

### `run_abide_bsnet.py` — ABIDE PCP 검증 (주력)
- **용도**: ABIDE I 전체 N=468 (CC200/CC400) BS-NET 검증. Pre-extracted ROI TS 사용.
- **CLI**: `--atlas {cc200,cc400}`, `--correction-method {original,fisher_z,partial,soft_clamp}`, `--n-jobs`, `--max-subjects`, `--group`, `--pipeline`, `--strategy`, `-v`
- **입력**: nilearn `fetch_abide_pcp` (자동 다운로드)
- **출력**: `data/abide/results/abide_bsnet_{atlas}_{pipeline}_{strategy}.csv`, `data/abide/timeseries_cache/`
- **예시**:
  ```bash
  python src/scripts/run_abide_bsnet.py --atlas cc200 --correction-method fisher_z --n-jobs 8
  python src/scripts/run_abide_bsnet.py --atlas cc400 --correction-method fisher_z --n-jobs -1
  ```

### `run_nilearn_adhd_bsnet.py` — ADHD-200 검증
- **용도**: nilearn ADHD 40명 (20 ADHD + 20 HC). Atlas parcellation 직접 수행 → 다중 atlas 비교 가능.
- **CLI**: `--atlas {cc200,cc400,all}`, `--correction-method`, `--n-jobs`, `--max-subjects`, `--group`, `-v`
- **입력**: nilearn `fetch_adhd` (자동 다운로드)
- **출력**: `data/adhd/results/adhd_bsnet_{atlas}.csv`, `data/adhd/results/adhd_summary_{atlas}.json`
- **예시**:
  ```bash
  python src/scripts/run_nilearn_adhd_bsnet.py --atlas all --correction-method fisher_z --n-jobs 8
  ```

### `run_adhd_classification.py` — Track H: ADHD vs Control 분류
- **용도**: ADHD-200 N=40에서 3가지 FC 조건 (Raw, BS-NET, Reference)으로 Linear SVM 분류. Per-fold 결과 CSV 저장.
- **CLI**: `--atlas {cc200,cc400}`, `--short-sec 120`, `--n-repeats 10`, `--n-bootstraps 100`, `--correction-method`, `--seed`, `-v`
- **입력**: `data/adhd/ts_cache/{atlas}/`, `data/adhd/results/adhd_bsnet_{atlas}.csv`
- **출력**: `data/adhd/results/adhd_classification_{atlas}.csv` (summary), `adhd_classification_{atlas}_folds.csv` (per-fold)
- **예시**:
  ```bash
  python src/scripts/run_adhd_classification.py --atlas cc200 cc400 --n-bootstraps 100
  ```

### `run_reliability_aware_clustering.py` — ρ̂T 기반 비지도 환자 분리 분석 (신규)
- **용도**: ADHD-200 PCP strict subset에서 `ρ̂T` tertile(T1/T2/T3)별 HC/Patients 분리도를 비지도 방식으로 정량화.
- **FC 방법**: nilearn `ConnectivityMeasure` (`correlation`, `partial correlation`, `tangent`)
- **클러스터링**: KMeans(주력), GMM, Spectral
- **평가 지표**: ARI, NMI, silhouette, balanced accuracy(라벨 flip 최적 매핑)
- **CLI**: `--n-repeats`, `--random-seed`, `--pca-var`, `--min-subjects`
- **출력**:
  - `data/adhd/pcp/results/adhd200_reliability_clustering_runs.csv`
  - `data/adhd/pcp/results/adhd200_reliability_clustering_summary.csv`
- **예시**:
  ```bash
  python src/scripts/run_reliability_aware_clustering.py --n-repeats 20 --random-seed 42
  ```

### `run_reliability_aware_classification.py` — ρ̂T 기반 감독 분류 (신규)
- **용도**: ADHD-200 PCP strict subset에서 ADHD vs Control 분류를 reliability-aware 방식으로 평가.
- **평가 설계**: LOSO(기본) 또는 Stratified K-fold.
- **FC 방법**: nilearn `ConnectivityMeasure` (`correlation`, `partial correlation`, `tangent`)
- **모델**: `logistic_l2`, `linear_svm`
- **보강 요소**:
  - repeat-wise class-balance downsampling
  - age/sex/site covariates (fold-safe)
  - `rho_hat_T` 기반 sample weighting (`--rho-weight-gamma`)
  - permutation p-value (`--n-permutations`, primary-only 옵션)
  - repeat-level parallelism (`--n-jobs`)
- **출력**:
  - `data/adhd/pcp/results/adhd200_reliability_classification_runs.csv`
  - `data/adhd/pcp/results/adhd200_reliability_classification_summary.csv`
- **예시**:
  ```bash
  python src/scripts/run_reliability_aware_classification.py \
    --eval-scheme loso \
    --balance-classes \
    --n-repeats 20 \
    --n-permutations 1000 \
    --permute-primary-only \
    --primary-fc tangent \
    --primary-model logistic_l2 \
    --n-jobs 8
  ```

### `convert_keane_restfc_to_npz.py` — ds003404/ds005073 restFC 통합 변환 (신규)
- **용도**: OpenNeuro `ds003404`(HC) + `ds005073`(BP/SZ) derivatives의 `.mat` FC를 통합 NPZ로 변환.
- **입력**:
  - `data/ds003404/derivatives/restFCArray.mat`
  - `data/ds005073/derivatives/restFCArray_BP.mat`
  - `data/ds005073/derivatives/restFCArray_SZ.mat`
  - 각 `participants.tsv`
- **출력**:
  - `data/ds005073/results/keane_restfc_combined.npz`
  - `data/ds005073/results/keane_restfc_metadata.csv`
- **주의**: derivatives FC만 사용하므로 BS-NET 보정 자체는 수행하지 않음 (raw BOLD 필요).

### `run_keane_fc_classification.py` — Keane FC-only 분류 (신규)
- **용도**: 통합 NPZ에서 exploratory 분류 실행.
- **태스크**:
  - `hc_vs_psychosis` (HC vs BP+SZ)  ※ dataset confound 주의
  - `bp_vs_sz` (within-ds005073)
- **모델**: `logistic_l2`, `linear_svm`
- **평가**: Repeated Stratified K-fold + permutation p-value
- **출력**:
  - `data/ds005073/results/keane_fc_classification_runs.csv`
  - `data/ds005073/results/keane_fc_classification_summary.csv`

### `run_keane_bsnet_recompute.py` — Keane TS 기반 BS-NET 재산출 (신규)
- **용도**: `data/derivatives/bsnet/sub-*/sub-*_ts.npy`에서 BS-NET 지표를 재계산.
- **핵심**: `--correction-method fisher_z`로 ceiling 패턴 완화된 ρ̂T 재산출.
- **출력**:
  - `data/keane/results/keane_bsnet_recomputed{_tag}.csv`
  - `data/keane/results/keane_bsnet_features{_tag}.npz`

### `run_keane_bsnet_classification.py` — Keane BP vs SZ 분류 (신규)
- **용도**: `run_keane_bsnet_recompute.py`의 feature NPZ + `rho_hat_T` CSV로 BP vs SZ 분류 수행.
- **특징**:
  - reliability gate (`none` / `hard quantile` / `soft weighting`)
  - train-fold 기준 threshold 산정(누수 방지)
  - confirmatory(primary 1개) + exploratory family 분리
  - permutation p-value + Holm/FDR 보정
- **출력**:
  - `data/keane/results/keane_bsnet_bp_sz_gated_runs{_tag}.csv`
  - `data/keane/results/keane_bsnet_bp_sz_gated_summary{_tag}.csv`

### `run_abide_filtered.py` — ABIDE 필터링 검증
- **용도**: ABIDE PCP에서 조건 필터링 후 BS-NET 검증.
- **출력**: `data/abide/results/`

### `run_adhd200_pcp_filtered.py` — ADHD-200 PCP 전체 검증 (N=399)
- **용도**: ADHD-200 PCP strict subset N=399 (6 sites) BS-NET 검증.
- **출력**: `data/adhd/pcp/results/`

### `run_duration_sweep.py` — Duration sweep 검증
- **용도**: τ_short별 ρ̂T 변화 측정. ds000243, ds007535 지원.
- **CLI**: `--dataset`, `--n-seeds`, `--n-jobs`
- **출력**: `data/*/results/duration_sweep_*.csv`

### `run_held_out_validation.py` — Held-out 검증
- **용도**: Held-out split 기반 BS-NET 검증.
- **출력**: `artifacts/reports/held_out_validation.csv`

### `run_fmriprep_bsnet.py` — fMRIPrep/XCP-D 기반 검증
- **용도**: XCP-D (기본) 또는 fMRIPrep-direct (레거시) 처리 결과에서 BS-NET 실행. Schaefer 100/400.
- **CLI**: `--subject`, `--run-all`, `--input-mode {xcpd,fmriprep}`, `--parcels {100,400}`, `--xcpd-dir`, `--fmriprep-dir`, `-v`
- **입력**: `data/derivatives/xcp-d/` 또는 `data/derivatives/fmriprep/`
- **출력**: `data/derivatives/bsnet/{sub_id}/` (FC, TS, JSON)

---

## 2. Defense Experiments (방어 실험 Track A–G)

BS-NET의 검증 로직(bootstrap → SB → prior → attenuation correction)은 데이터 소스(synthetic/real)에 무관하게 동작한다. 모든 방어 스크립트는 `--input-npy` 옵션으로 전처리된 실데이터(.npy)를 받을 수 있다. 생략 시 synthetic data로 자동 생성.

```bash
# Synthetic (기본)
python src/scripts/run_component_necessity.py

# Real data (.npy, shape: n_samples × n_rois)
python src/scripts/run_component_necessity.py --input-npy data/abide/timeseries_cache/cc200/50033_cc200.npy
```

| 스크립트 | Track | 기본 데이터 | 용도 | 출력 |
|----------|-------|-------------|------|------|
| `run_sensitivity_analysis.py` | A | synthetic | Hyperparameter robustness (reliability_coeff × observation_var, prior_mean × prior_var) | `artifacts/reports/sensitivity_analysis_{phase}.csv` |
| `run_ablation_study.py` | B | synthetic | 5-level incremental contribution (L0→L5) | `artifacts/reports/ablation_results.csv` |
| `run_stationarity_test.py` | C | synthetic | ADF + ICC for SB 가정 검증. optional: statsmodels | `artifacts/reports/stationarity_test.csv` |
| `run_shrinkage_comparison.py` | D | synthetic | LW vs OAS vs Pearson. optional: sklearn | `artifacts/reports/shrinkage_comparison.csv` |
| `run_component_necessity.py` | E | synthetic | Leave-one-out (SB/LW/boot/prior/atten 제거) | `artifacts/reports/component_necessity.csv` |
| `run_noise_degradation.py` | F | synthetic | 3D sweep: noise × ROI × duration | `artifacts/reports/noise_degradation.csv` |
| `analyze_ceiling_effect.py` | G | **real (ABIDE 캐시)** | 4-method ceiling comparison. `run_abide_bsnet.py` 선행 필요 | `artifacts/reports/ceiling_effect_*.csv`, PNG |
| `run_failure_analysis.py` | — | synthetic | 실패 피험자 특성 (SNR, reliability, AR(1)) | `artifacts/reports/failure_analysis.csv` |

**공통 옵션**: `--input-npy PATH` (전처리된 .npy), `-v` (verbose). Track G만 자체 ABIDE 캐시 로딩.

---

## 3. Data Acquisition & Conversion (데이터 수집/변환)

### `index_openneuro_hc.py`
- **용도**: OpenNeuro GraphQL API로 7개 데이터셋의 HC 피험자 인덱싱.
- **CLI**: `--cache-dir`
- **출력**: `data/hc_adult_index.csv`

### `download_hc_100.py`
- **용도**: 인덱스 기반 100명 HC 다운로드 (7개 데이터셋에서 균형 샘플링).
- **CLI**: `--n-subjects`, `--seed`, `--dry-run`
- **출력**: `data/openneuro/<ds_id>/sub-*/`

### `download_adhd200_pcp.py`
- **용도**: ADHD-200 PCP 데이터 다운로드.
- **출력**: `data/adhd/pcp/`

### `convert_adhd200_pcp.py`
- **용도**: ADHD-200 PCP 데이터를 BS-NET 입력 형식으로 변환.
- **출력**: `data/adhd/pcp/timeseries_cache/`

### `convert_xcpd_to_npy.py`
- **용도**: XCP-D 출력을 BS-NET 입력용 .npy로 변환.
- **입력**: `data/derivatives/xcp-d/`
- **출력**: `data/*/timeseries_cache/`

### `extract_xcpd_timeseries.py`
- **용도**: XCP-D parcellated time series 추출 유틸리티.

---

## 4. Preprocessing (전처리)

### `setup_and_preprocess.py`
- **용도**: Atlas 설정 + raw BOLD 전처리 + FC 추출.
- **CLI**: `--test-one`, `--run-all`
- **출력**: `data/atlas/`, `data/derivatives/sub-*/`

### `preprocess_ds007535.py`
- **용도**: ds007535 (SpeechHemi) task-residual FC 전처리. HRF-convolved task regressors + 36P.
- **CLI**: `--input-dir`, `--n-jobs`
- **출력**: `data/ds007535/timeseries_cache/`

### `preprocess_ds000243.py`
- **용도**: ds000243 (WashU resting-state) 36P confound regression. Multi-run concat 지원, TR=2.5s.
- **CLI**: `--input-dir`, `--n-jobs`
- **출력**: `data/ds000243/timeseries_cache/`

*레거시 스크립트는 `backup/legacy_scripts_20260329/`로 아카이브됨.*

---

## 5. Convergence & Ablation (수렴 검증/삭감 실험)

### `run_convergence_validation.py`
- **용도**: τ_short별 ρ̂T vs r_FC gap 검증. ds000243 N=49, 18 τ_short points.
- **CLI**: `--dataset`, `--n-seeds`, `--n-jobs`
- **출력**: `data/ds000243/results/convergence_*.csv`

### `run_progressive_ablation.py`
- **용도**: L0→L5 cumulative 6-level 실데이터 ablation.
- **CLI**: `--dataset`, `--n-seeds`, `--n-jobs`
- **출력**: `data/*/results/progressive_ablation_*.csv`

### `run_abide_duration_sweep.py`
- **용도**: ABIDE 데이터셋 대상 duration sweep.
- **출력**: `data/abide/results/duration_sweep_*.csv`

### `run_tsd_ablation.py` — TSD E0–E3 Ablation (신규)
- **용도**: Temporal Self-Distillation 이론 검증. E0(baseline BS-NET) → E1(w*_B bootstrap ensemble) → E2(w*_G Ridge LOOCV) → E3(combined).
- **CLI**: `--dataset {ds000243,abide}`, `--experiment {E0,E1,E2,E3,all}`, `--atlas`, `--short-sec`, `--tr`, `--n-bootstraps`, `--n-seeds`, `--correction-method`, `--alpha-ridge`, `--n-jobs`
- **입력**: `data/{ds000243,abide}/timeseries_cache/` (.npy time series)
- **출력**: `data/*/results/tsd_ablation_runs_*.csv`, `tsd_ablation_summary_*.csv`, `tsd_glm_r2_*.csv`
- **예시**:
  ```bash
  python src/scripts/run_tsd_ablation.py --dataset ds000243 --experiment all --n-seeds 10 --n-jobs 8
  python src/scripts/run_tsd_ablation.py --dataset abide --experiment E0 E1 --n-seeds 5
  ```

---

## 6. Downstream Analysis (하류 분석)

### `run_downstream_analysis.py`
- **용도**: 7-analysis suite (FC similarity, connectome, Cohen's d, SVM, graph metrics, ρ̂T-stratified, fingerprinting).
- **CLI**: `--n-jobs`
- **출력**: `data/adhd/pcp/results/downstream_*.csv`

### `run_reliability_aware_clustering.py`
- **용도**: ρ̂T tertile별 비지도 환자 분리 분석 (KMeans, GMM, Spectral).
- **CLI**: `--n-repeats`, `--random-seed`, `--n-jobs`
- **출력**: `data/adhd/pcp/results/adhd200_reliability_clustering_*.csv`

### `run_reliability_aware_classification.py`
- **용도**: LOSO/stratified k-fold 감독 분류. tangent FC, permutation p-value.
- **CLI**: `--eval-scheme`, `--n-repeats`, `--n-permutations`, `--n-jobs`
- **출력**: `data/adhd/pcp/results/adhd200_reliability_classification_*.csv`

---

## 7. Visualization (시각화)

### `plot_abide_results.py` (src/scripts/)
- **용도**: ABIDE 단일시드/멀티시드 결과 4-panel figure.
- **CLI**: `--csv`, `--multi-seed N`, `--correction-method`, `--n-jobs`, `-v`
- **출력**: `data/abide/results/abide_bsnet_{multiseed_}*.png`, multi-seed CSV

### src/visualization/ (논문 Figure 전용)

| 스크립트 | Figure | 내용 |
|----------|--------|------|
| `plot_figure1_combined.py` | Fig 1 | Method Overview — Pipeline Schematic + Convergence Validation + τ_min Estimation |
| `plot_figure2_component.py` | Fig 2 | Component Necessity — LOO + Progressive 4-level + Cross-dataset + Distribution |
| `plot_figure3_abide.py` | Fig 3 | ABIDE Validation — N=468, multi-seed, Fisher z, violin+boxplot |
| `plot_figure4_adhd.py` | Fig 4 | ADHD Validation — cross-dataset generalization, violin+boxplot |
| `plot_figure5_structure.py` | Fig 5 | Network Structure Preservation — topology/community analysis |
| `plot_figure6_classification.py` | Fig 6 | ADHD Classification — Linear SVM, 3 FC conditions × 2 atlases |
| `plot_figure_s1_progressive_full.py` | Fig S1 | 6-level progressive ablation × k-group (3×2 layout) |
| `plot_figure_s2_k_stratification.py` | Fig S2 | k-stratification dose-response + per-site summary |
| `plot_figure_s3_abide_filtered_consort.py` | Fig S3 | ABIDE filtered CONSORT flowchart |
| `plot_patient_utility_clustering.py` | Fig S6 | ρ̂T strata 비지도 분리도(ARI/BalAcc) + FC method heatmap (EXPLORATORY) |
| `plot_patient_utility_classification.py` | Fig S7 | LOSO supervised discrimination(BalAcc/AUC) + permutation p panel |
| `plot_component_necessity.py` | — | Component necessity bars + Δρ waterfall (standalone) |
| `plot_convergence_validation.py` | — | Convergence validation standalone plotting |
| `plot_fc_intuition.py` | — | FC intuition visualization (legacy concept) |
| `plot_figure0_conceptual.py` | — | Conceptual framework figure |
| `plot_held_out_validation.py` | — | Held-out validation plotting |
| `plot_network_visualization.py` | — | Network visualization utility |
| `style.py` | — | 공용 matplotlib 스타일 설정 (library) |

*구 버전 스크립트(plot_figure1_ds007535 등)는 `src/visualization/legacy/`에 보관.*

---

## 8. Simulation (시뮬레이션)

### `run_synthetic_baseline.py`
- **용도**: BS-NET 기본 동작 확인. TR=1s, 50 ROIs, 100 bootstraps.
- **CLI**: 없음
- **출력**: stdout (파일 출력 없음)

### `sweep_simulation.py`
- **용도**: Duration sweep (10 seeds). 예측 신뢰도 vs 스캔 시간.
- **CLI**: 없음
- **출력**: `artifacts/reports/duration_sweep_seeds_results.csv`

---

## 9. Utility (유틸리티)

### `inspect_craddock_atlas.py`
- **용도**: Craddock CC200/CC400 atlas ROI 검사 및 시각화.

### `visualize_fc_threshold.py`
- **용도**: FC matrix threshold 시각화 유틸리티.

### `analyze_fc_stratification.py`
- **용도**: FC 값 분포별 stratification 분석.

---

## 10. Pipeline Orchestration (Shell Scripts — 14개)

| 스크립트 | 용도 | 주요 옵션 |
|----------|------|-----------|
| `run_all_pipeline.sh` | 6-stage 마스터 오케스트레이터 (인덱싱→다운로드→fMRIPrep→XCP-D→BS-NET→요약) | `--step`, `--dry-run`, `--singularity` |
| `run_dataset_pipeline.sh` | Per-dataset 점진적 실행 (ds* 단위) | `--dataset`, `--auto`, `--skip-download` |
| `run_fmriprep_batch.sh` | fMRIPrep 배치 (Docker/Singularity) | `--subject`, `--csv`, `--all`, `--singularity` |
| `run_fmriprep_manual.sh` | fMRIPrep v25.2.5 수동 실행 + 진단 | `--check`, `--subject`, `--batch` |
| `run_fmriprep_keane.sh` | Keane(ds003404/ds005073) REST 전용 fMRIPrep | `--dataset`, `--subject`, `--dry-run` |
| `run_keane_streaming_pipeline.sh` | Keane subject 단위 streaming (datalad→fMRIPrep→BS-NET→cleanup) | `--dataset`, `--subject`, `--cleanup-level` |
| `run_xcpd_batch.sh` | XCP-D 배치 (36P, scrubbing, Schaefer) | `--subject`, `--parcels`, `--singularity` |
| `run_xcpd_ds000243.sh` | ds000243 전용 XCP-D 실행 | — |
| `run_ds000243_batch.sh` | ds000243 전체 배치 (preprocess + 6 atlas sweep) | — |
| `run_ds007535_batch.sh` | ds007535 전체 배치 | — |
| `run_component_necessity_batch.sh` | Component necessity 배치 실행 | — |
| `setup_local_env.sh` | 환경 초기화 (conda/venv + pip) | `--venv` |
| `setup_keane_datalad.sh` | Keane 데이터셋 DataLad clone 설정 | — |
| `install_datalad.sh` | DataLad + git-annex 설치 | — |

---

## 아카이브된 레거시 스크립트

`backup/legacy_scripts_20260329/`에 보관. 상세는 `MANIFEST.md` 참조.

| 원본 | 대체 | 사유 |
|------|------|------|
| `scripts/preprocess_bold.py` | `run_fmriprep_batch.sh` + `run_xcpd_batch.sh` | ANTsPy 직접 → fMRIPrep/XCP-D |
| `scripts/run_real_figure4.py` | `src/visualization/plot_figure4_subnetworks.py` | 시각화 모듈 통합 |
| `scripts/setup_env.sh` | `src/scripts/setup_local_env.sh` | 중복 |
| `src/scripts/run_real_data.py` | `run_fmriprep_bsnet.py` | MoBSE 의존 |
| `src/scripts/run_real_data_scale.py` | `run_fmriprep_bsnet.py --run-all` | MoBSE 의존 |
| `src/scripts/preprocess_real_data.py` | `run_fmriprep_batch.sh` + `run_xcpd_batch.sh` | MoBSE 의존 |

---

## 공통 CLI 패턴

모든 validation/analysis 스크립트는 다음 패턴을 따른다:

```bash
# 병렬화
--n-jobs 8          # 8 workers
--n-jobs -1         # all cores

# Correction method (Fisher z 권장)
--correction-method fisher_z

# Atlas 선택
--atlas cc200       # Craddock 200
--atlas cc400       # Craddock 400
--atlas all         # 모든 atlas 순차 실행

# 디버깅
-v / --verbose      # DEBUG 레벨 로깅
--max-subjects 5    # 소수 피험자로 빠른 테스트
```

---

## 의존성 그래프

```
Validation scripts ──→ src.core (bootstrap, config, pipeline)
                    ──→ src.data (data_loader)
                    ──→ nilearn, nibabel, numpy, scipy

Defense experiments ──→ src.core (bootstrap, config, pipeline, simulate)
                    ──→ numpy, scipy (외부 데이터 불필요)

Visualization       ──→ src.core, src.data
                    ──→ matplotlib, seaborn

Shell pipelines     ──→ Docker/Singularity (fMRIPrep, XCP-D)
                    ──→ Python scripts (chained)
```
