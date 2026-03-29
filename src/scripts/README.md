# BS-NET Scripts Index

스크립트를 **7개 카테고리**로 분류한 인덱스. 각 스크립트의 용도, CLI, 입출력을 명시한다.

---

## 카테고리 요약

| # | 카테고리 | 스크립트 수 | 설명 |
|---|----------|------------|------|
| 1 | Validation (실증 검증) | 3 | ABIDE, ADHD, fMRIPrep 기반 실데이터 BS-NET 검증 |
| 2 | Defense Experiments (방어 실험) | 8 | Track A–G 방어 실험 + failure analysis |
| 3 | Data Acquisition (데이터 수집) | 2 | OpenNeuro 인덱싱 및 다운로드 |
| 4 | Preprocessing (전처리) | 1 | Atlas parcellation, confound regression |
| 5 | Visualization (시각화) | 1+5 | ABIDE plotting + src/visualization/ 논문 figure |
| 6 | Simulation (시뮬레이션) | 2 | Synthetic baseline + duration sweep |
| 7 | Pipeline Orchestration (파이프라인) | 6 (.sh) | fMRIPrep, XCP-D 배치, 환경 설정 |

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

## 3. Data Acquisition (데이터 수집)

### `index_openneuro_hc.py`
- **용도**: OpenNeuro GraphQL API로 7개 데이터셋의 HC 피험자 인덱싱.
- **CLI**: `--cache-dir`
- **출력**: `data/hc_adult_index.csv`

### `download_hc_100.py`
- **용도**: 인덱스 기반 100명 HC 다운로드 (7개 데이터셋에서 균형 샘플링).
- **CLI**: `--n-subjects`, `--seed`, `--dry-run`
- **의존**: `index_openneuro_hc.py` 출력
- **출력**: `data/openneuro/<ds_id>/sub-*/`

---

## 4. Preprocessing (전처리)

### `setup_and_preprocess.py`
- **용도**: Atlas 설정 + raw BOLD 전처리 + FC 추출.
- **CLI**: `--test-one`, `--run-all`
- **출력**: `data/atlas/`, `data/derivatives/sub-*/`

*레거시 스크립트 (`run_real_data.py`, `run_real_data_scale.py`, `preprocess_real_data.py`)는 `backup/legacy_scripts_20260329/`로 아카이브됨.*

---

## 5. Visualization (시각화)

### `plot_abide_results.py` (src/scripts/)
- **용도**: ABIDE 단일시드/멀티시드 결과 4-panel figure.
- **CLI**: `--csv`, `--multi-seed N`, `--correction-method`, `--n-jobs`, `-v`
- **출력**: `data/abide/results/abide_bsnet_{multiseed_}*.png`, multi-seed CSV

### src/visualization/ (논문 Figure 전용)

| 스크립트 | Figure | 내용 |
|----------|--------|------|
| `plot_figure1_combined.py` | Fig 1 | Prediction accuracy vs duration, marginal gain, uncertainty decay |
| `plot_figure2_validation.py` | Fig 2 | N=100 validation (scatter, KDE, error, pass rate) |
| `plot_figure3_topology.py` | Fig 3 | Small-worldness + degree variance |
| `plot_figure4_subnetworks.py` | Fig 4 | Jaccard overlap + modularity |
| `plot_component_necessity.py` | Fig 5 | Component necessity bars + Δρ waterfall |
| `style.py` | — | 공용 matplotlib 스타일 설정 (library) |

---

## 6. Simulation (시뮬레이션)

### `run_synthetic_baseline.py`
- **용도**: BS-NET 기본 동작 확인. TR=1s, 50 ROIs, 100 bootstraps.
- **CLI**: 없음
- **출력**: stdout (파일 출력 없음)

### `sweep_simulation.py`
- **용도**: Duration sweep (10 seeds). 예측 신뢰도 vs 스캔 시간.
- **CLI**: 없음
- **출력**: `artifacts/reports/duration_sweep_seeds_results.csv`

---

## 7. Pipeline Orchestration (Shell Scripts)

| 스크립트 | 용도 | 주요 옵션 |
|----------|------|-----------|
| `run_all_pipeline.sh` | 6-stage 마스터 오케스트레이터 (인덱싱→다운로드→fMRIPrep→XCP-D→BS-NET→요약) | `--step`, `--dry-run`, `--singularity` |
| `run_dataset_pipeline.sh` | Per-dataset 점진적 실행 (ds* 단위) | `--dataset`, `--auto`, `--skip-download` |
| `run_fmriprep_batch.sh` | fMRIPrep 배치 (Docker/Singularity) | `--subject`, `--csv`, `--all`, `--singularity` |
| `run_fmriprep_manual.sh` | fMRIPrep v25.2.5 수동 실행 + 진단 | `--check`, `--subject`, `--batch` |
| `run_xcpd_batch.sh` | XCP-D 배치 (36P, scrubbing, Schaefer 100/400) | `--subject`, `--parcels`, `--singularity` |
| `setup_local_env.sh` | 환경 초기화 (conda/venv + pip) | `--venv` |

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
