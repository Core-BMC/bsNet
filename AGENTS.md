# AGENTS.md — BS-NET Project

## Co-Author

이 프로젝트의 리팩토링, 문서화, 시각화 작업은 각 에이전트 모델 (LLM Agent,   Gemini, GPT, Claude 등)의 지원을 받아 수행되었습니다.

## Git Policy

- 커밋 메시지에 Co-Authored-By 트레일러를 포함하지 않는다.
- Co-author 정보는 이 문서에만 기록한다.

## Script Execution Policy

- 스크립트는 **사용자가 직접 실행**한다.
- Agent는 스크립트 작성 후 **실행 명령어만 알려준다**.
- 실행 환경: 항상 `.venv` 기준 (`source .venv/bin/activate` 또는 `(.venv)` 상태에서 `python3`)
- **예외**: 사용자가 명시적으로 "돌려줘"라고 요청한 경우에만 Agent가 직접 실행.
- **BLAS threading**: `ProcessPoolExecutor` + NumPy 병렬 실행 시 반드시 아래 환경변수 설정:
  ```bash
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  ```

  미설정 시 worker × BLAS threads 경합으로 CPU 100% 포화 + 극심한 성능 저하 발생.

## Git Commit Policy

- `git commit`은 **사용자가 직접 실행**한다.
- Agent는 staging(`git add`)까지만 처리하고, **커밋 메시지만 제공**한다.
- 커밋 메시지는 코드블록으로 복사 가능하게 제공한다.
- **이유**: Cowork sandbox의 FUSE 파일시스템에서 `.git/index.lock` 제거 불가로 `git commit` 실행 불가.

## Session Protocol

> 모든 AI 에이전트(Claude, Codex 등)는 세션 시작/종료 시 이 프로토콜을 따른다.

### 세션 시작 (Recall)

1. **이 파일** (`CLAUDE.md` 또는 `AGENTS.md`) 읽기 → 프로젝트 개요, 정책, 현재 상태 파악
2. **`docs/dev/`** 디렉토리에서 **가장 최근 dev log** 읽기 → 직전 세션의 상세 작업 내역 + TODO 확인
3. `git status -s` + `git log --oneline -5` → 커밋 이후 미반영 변경 사항 파악
4. 직전 dev log의 **TODO 섹션**을 오늘 작업의 시작점으로 삼기

### 세션 종료 (Relay)

1. **dev log 갱신** (`docs/dev/YY-MM-DD-HH-MM_dev_logs.md`)
   - 오늘 날짜 파일이 있으면 새 Session 섹션 추가, 없으면 신규 생성
   - 형식: Added / Changed / Fixed / Experiments / TODO (상세 지침: `docs/dev/README.md`)
   - TODO는 **반드시** 작성 — 다음 세션의 시작점
2. **이 파일의 Current Status** 갱신 (간략 요약만, 상세는 dev log에 위임)
   - 날짜와 세션 번호 업데이트
   - 핵심 변경 사항 1–3줄 추가 또는 수정
3. **CLAUDE.md ↔ AGENTS.md 동기화** — 두 파일의 내용이 다르면 최신 쪽으로 통일
4. **Pending Tasks** 갱신 — 완료 항목 체크, 신규 항목 추가
5. **Next Session TODO** 갱신 — 다음에 바로 착수할 구체적 작업 목록

### 기록 원칙

- **CLAUDE.md / AGENTS.md**: 간결한 현황 + 지침 (What & Where)
- **docs/dev/*.md**: 상세 이력 + 의사결정 근거 (Why & How)
- 중복 기록 금지 — 상세 내용은 항상 dev log에, 여기서는 참조만

---

## Project Overview

**BS-NET** (Bootstrap reliability-adjusted FC Network Estimation)는 2분 rsfMRI 스캔으로부터
15분 상당의 FC(Functional Connectivity)를 외삽(extrapolate)하는 방법론이다.

핵심 파이프라인: Short scan → Ledoit-Wolf shrinkage FC → Block bootstrap resampling
→ Spearman-Brown prophecy (k=7.5) → Bayesian empirical prior → Attenuation correction
→ **Fisher z-space bounding** → ρ̂T

## Current Status (2026-04-22, updated)

> 상세 이력: `docs/dev/` 참조 (세션별 Added/Changed/Fixed/TODO 기록)

- **Session 1–4 요약**: 코드 리팩토링, 방어 실험 Track A–H, ABIDE/ADHD 실증, ds000243 파이프라인 완료
- **Session 5**: Fig 1 (FC Intuition) 3×3 figure 완성, Figure legend 상세 작성
- **Session 6**: Figure 번호 체계 재구성 (7→6), Fig 1–6 전체 재생성, Fig 2 스타일 변경
- **Session 7**: ADHD-200 PCP 전체 검증 (N=399) + Downstream Analysis + τ_min 개념
- **Session 8**: Convergence Validation 실험 + 시각화 + τ_min empirical estimation
- **Session 9**: Fig 2 완성 + Supplementary S1/S2 신규 작성 + progressive ablation 스크립트
- **04-17 Sessions 1–4**: Fig 3 ABIDE main/supplementary 분리, Fig3/FigS3/Fig4 스타일 통일 (violin+boxplot+outlier)
- **04-17 Sessions 5–7**: Reliability-aware clustering 파이프라인 (`run_reliability_aware_clustering.py`, `plot_patient_utility_clustering.py`), 퍼뮤테이션 p-value, class-balance 보정
- **04-17 Session 8**: FigS6 guardrails (EXPLORATORY 워터마크, p-value annotations)
- **04-17 Session 9**: ADHD discrimination citation guide (`docs/5.13`)
- **04-17 Session 10**: Reliability-aware classification 파이프라인 (`run_reliability_aware_classification.py`) — LOSO + stratified k-fold, tangent FC, permutation p-values
- **04-18 Session 1**: FigS7 supervised utility figure (`plot_patient_utility_classification.py`), PCP N=399 S6/S7 legends 갱신
- **04-18 Sessions 2–4**: **Keane ds003404/ds005073 파이프라인** — FC 변환, FC-only 분류, REST fMRIPrep 스크립트, streaming pipeline
- **04-18 이후 커밋**: Keane BS-NET 재산출 (`run_keane_bsnet_recompute.py`), BP vs SZ 분류 (`run_keane_bsnet_classification.py`) — confirmatory/exploratory family 분리, Holm/BH-FDR 보정, hard gate fold-train-only threshold, min-valid-splits 옵션
- **Figure 번호 체계**: Main 1–6 확정, Supplementary S1–S3, S6–S7 추가
  - **Fig 1: Method Overview** (A: Pipeline Schematic, B1–B3: Convergence Validation, C: τ_min Estimation)
  - **Fig 2: Component Necessity** (A: LOO, B: Progressive 4-level, C: Cross-dataset, D: Distribution) — **완료**
  - **Fig 3: ABIDE Validation** — main/supplementary 분리 완료
  - **Fig 4: ADHD Validation** — 스타일 통일 완료
  - Fig 5: Structure | Fig 6: Classification
  - **Fig S1: 6-Level Progressive by k-Group** (3×2) — **완료**
  - **Fig S2: k-Stratification Analysis** — **완료**
  - **Fig S3: ABIDE Filtered CONSORT** — **완료**
  - **Fig S6: Reliability-Aware Clustering** (EXPLORATORY 워터마크) — **완료**
  - **Fig S7: Reliability-Aware Classification** (permutation p annotation) — **완료**
- **Keane ds003404/ds005073**: BP vs SZ classification, HC vs Psychosis, streaming pipeline (datalad→fMRIPrep→BS-NET→cleanup)
- **ADHD-200 PCP 검증**: N=399 (6 sites), r_FC=0.525→ρ̂T=0.725, 100% improved, ceiling=0
- **Convergence Validation**: ds000243 N=49, 18 τ_short points, gap 100% positive, ρ̂T peak=0.774 at 150s
- **τ_min**: empirical plateau [90, 180]s, 95% peak at 40s, 실용 권장 60-120s
- **Downstream Analysis**: 7-analysis suite 완료 (FC sim, connectome, Cohen's d, SVM, graph, ρ̂T-stratified, fingerprint)
- **ρ̂T Dose-Response**: 3/3 monotonicity 확인 (T1<T2<T3), BS-NET = reliability estimator로 확정
- **τ_min 개념**: Minimum Common Bootstrap Duration — hemodynamic low-freq cycle 기반 이론적 하한
- **스토리라인**: Method+Proof(Fig1) → Mechanism(Fig2) → Validation(Fig3) → Cross-Dataset(Fig4) → Safety(Fig5) → Utility(Fig6)
- **04-22 Session 1**: TSD(Temporal Self-Distillation) 이론 검토 + E0–E3 ablation 실험 스크립트 작성
  - TSD 이론: BS-NET을 5-category distillation 프레임워크로 재해석 (Statistical/Knowledge/Self/Cross-Architecture/Temporal)
  - `run_tsd_ablation.py`: E0(baseline) → E1(w*_B bootstrap ensemble) → E2(w*_G Ridge LOOCV) → E3(combined)
  - TSD 이론 검토 6건 권고사항 도출 (Cat 3 reframing, distillation qualifier, ds000243 non-stationarity 등)
- **04-22 Session 2**: TSD 이론 문서 docx → markdown 변환 (`docs/5.16_tsd_theory_v2.md`)
  - 6건 권고사항 전량 반영: Cat 3 Teeuw reframing, analytic distillation qualifier, Xiang 저자명, Guo Tier 1 승격, E0→E1 우선 전략, ds000243 non-stationarity
  - docx tracked changes는 보고서 최종 출력 시 별도 처리 예정
- **04-22 Session 3**: TSD E0–E2 실험 완료 (ds000243, N=52, schaefer200, 10 seeds)
  - E0=0.816, E1(vs teacher)=0.816, E2=0.725 → **E0≈E1, E2<E0**
  - 결론: implicit distillation 포화 + GLM prediction이 actual teacher보다 열등 → E3 스킵
  - BS-NET은 이미 최적 teacher(full scan FC)를 보유 → TSD는 이론적 렌즈, 실용 개선 불가

## Pending Tasks

### Tier 1 (reject 방지 — 필수)

- [X] `reliability_coeff=0.98`을 "within-session scanner reliability"로 명시적 정의 — config.py/bootstrap.py에 Friedman 2008 주석 완비
- [X] BCa vs percentile CI 불일치 해결 — percentile로 통일 완료 (5.4/5.5 문서 확인, 코드에 BCa 미구현)
- [X] "ground truth" → "reference" 용어 변경 — graph_metrics.py, run_held_out_validation.py, test_graph_metrics.py, 1.2_arch_pipeline.md 완료
- [X] 전처리 파이프라인 상세 기술 (Methods 섹션용) — 5.5 문서 코드 일치 완료
- [ ] Sensitivity analysis 재설계 (oracle noise 패턴 문제 해결)
- [X] Ceiling effect 보정 — Fisher z-space correction 구현 및 4-method 비교 (Track G)

### Tier 2 (major revision 방지)

- [ ] 9% 실패 피험자 특성 분석
- [ ] Stationarity test 결과 논문 반영 방식 확정 (ICC=-0.16, Cheng et al. 2021 선례 활용)
- [X] docs/INDEX.md에 신규 문서 반영
- [X] ABIDE 전체 N=468 Fisher z multi-seed 실행 완료 (ρ̂T=0.843±0.036, ceiling 0%)
- [X] CC400 atlas 비교 완료 (ρ̂T=0.834±0.037, 97.4% improved)
- [X] ADHD 40명 검증 실행 — Fisher z, CC200 ρ̂T=0.866, CC400(356 ROIs) ρ̂T=0.855
- [ ] Track E ABIDE 실데이터 결과 분석 (N=468, CC200+CC400 배치 실행 중)

### Figure 1 — Method Overview (Session 8에서 확정)

- [X] 구 Fig 1 (FC Intuition 3×3) 폐기 결정: "FC denoiser" 오해 + Panel H/I 설계 결함
- [X] 새 Fig 1 구성 확정: A(Pipeline Schematic) + B1-B3(Convergence) + C(τ_min)
- [X] `plot_figure1_combined.py` 작성 — matplotlib 블록도 + convergence + τ_min 통합
- [X] `Fig1_Method_Overview.png` 생성 완료
- [X] Panel A: 최종 논문용 Illustrator 교체 (현재 matplotlib placeholder)
- [X] style.py에 Fig 3-7 Gray/Amber/Blue 3색 스키마 공식 등록 + DOT_COLOR/ACCENT_COLORS 추가, Fig 3–7 하드코딩 정리

### ds007535 (SpeechHemi) — 보류

- [X] ds007535 선정 및 전처리 스크립트 작성 완료
- [ ] ds007535 bold.nii.gz 다운로드 — Fig 1 교체로 **우선순위 하락**, Supplementary 후보

### ds000243 (WashU resting-state) — 주력 검증 데이터셋

- [X] `preprocess_ds000243.py` 작성 (36P confound regression only, no task regression)
- [X] `run_duration_sweep.py`에 ds000243 지원 추가 (`discover_subjects_ds000243`)
- [X] `run_ds000243_batch.sh` 작성 (6-atlas batch sweep)
- [X] `data/ds000243/{raw,timeseries_cache,results}/` 디렉토리 생성
- [X] ds000243 fMRIPrep 출력물 업로드 완료 (N=52, MNI152NLin6Asym, TR=2.5s 확인)
- [X] `preprocess_ds000243.py` 리팩토링: multi-run concat, confounds path 버그 수정, TR=2.5s, `--n-jobs 8`
- [X] `run_ds000243_batch.sh` preprocess 단계 통합 (find pipefail 버그 수정)
- [X] `run_ds000243_batch.sh` 실행 완료 (6 atlases × 52 subjects × 10 seeds, ~8.25h)
- [X] Figure 1 (Method Overview): ds000243 convergence data 기반 `plot_figure1_combined.py` 완료
- [X] Figure 2 (Component Necessity): ds000243 실데이터 버전 추가 (현재 ABIDE N=468 기반)
- [ ] Figure 4 (Network Structure Preservation): ds000243 기반 topology/community 분석

### Supplementary Figures
- [X] Fig S1: 6-level progressive ablation × k-group (3×2 layout) — `plot_figure_s1_progressive_full.py`
- [X] Fig S2: k-stratification dose-response + per-site summary — `plot_figure_s2_k_stratification.py`
- [X] `run_progressive_ablation.py`: L0→L5 cumulative 6-level 실데이터 ablation 스크립트
- [X] Fig S3: ABIDE Filtered CONSORT — `plot_figure_s3_abide_filtered_consort.py`
- [X] Fig S6: Reliability-Aware Clustering — `plot_patient_utility_clustering.py` (EXPLORATORY 워터마크)
- [X] Fig S7: Reliability-Aware Classification — `plot_patient_utility_classification.py` (permutation p annotation)

### Keane ds003404/ds005073 — 신규 검증 데이터셋
- [X] `convert_keane_restfc_to_npz.py`: .mat FC → NPZ/metadata 변환
- [X] `run_keane_fc_classification.py`: FC-only exploratory 분류 (HC vs Psychosis, BP vs SZ)
- [X] `run_fmriprep_keane.sh`: REST-only fMRIPrep 실행 스크립트
- [X] `run_keane_streaming_pipeline.sh`: subject 단위 streaming (datalad→fMRIPrep→BS-NET→cleanup)
- [X] `run_keane_bsnet_recompute.py`: ts.npy 기반 BS-NET 재산출
- [X] `run_keane_bsnet_classification.py`: confirmatory(Holm) + exploratory(BH-FDR), hard gate train-only, min-valid-splits
- [ ] Keane 실행 결과 수집 및 분석 (BP vs SZ accuracy, permutation p-value)
- [ ] Keane 결과를 논문 Supplementary/Discussion에 반영

### Reliability-Aware Pipelines
- [X] `run_reliability_aware_clustering.py`: permutation p-value, class-balance 보정, parallel workers
- [X] `run_reliability_aware_classification.py`: LOSO + stratified k-fold, tangent FC, permutation p-values
- [X] `docs/5.12_patient_utility_reliability_aware_clustering.md`
- [X] `docs/5.13_adhd_discrimination_citation_guide.md`

### TSD (Temporal Self-Distillation) — 이론 + 실험
- [X] TSD 이론 문서 검토 (BS-NET_TSD_Theory_v2.docx) — 5-category distillation mapping
- [X] `run_tsd_ablation.py`: E0–E3 ablation 실험 스크립트 (ds000243/ABIDE 지원)
- [X] TSD 이론 6건 권고사항 반영 — `docs/5.16_tsd_theory_v2.md`로 clean markdown 저장
  - Cat 3 Teeuw reframing, analytic distillation qualifier, Xiang 저자명, Guo Tier 1 승격, E0→E1 전략, ds000243 non-stationarity
- [X] docx tracked changes 작성 (BS-NET_TSD_Theory_v2_reviewed.docx) — 최종 보고서 출력 시 사용
- **E0–E2 Distillation Ablation 실험** (완료, ds000243 N=52 schaefer200):
  - [x] **E0 baseline**: ρ̂T=0.816±0.028, r_FC=0.716, Δ=+0.100
  - [x] **E1 w\*_B**: ρ̂T_self=0.954 (vs ensemble), ρ̂T_vs_teacher=0.816 — **E0≈E1 확인**
  - [x] **E0 vs E1 분석**: Δ(E1-E0)= -0.0003 → implicit distillation 포화
  - [x] **E2 w\*_G**: ρ̂T=0.725±0.036, GLM R²≈0.45–0.59 — **E2 < E0 (Δ=-0.091)**
  - [x] **E3 스킵 결정**: E2 < E0이므로 E1+E2 blend가 E0을 초과할 수 없음
  - [ ] TSD 실험 결과 논문 반영 — **negative result로 보고, 이론적 렌즈로서의 가치 강조**
- **실험 결론**:
  - E0≈E1: BS-NET의 기존 파이프라인이 bootstrap ensemble 정보를 이미 완전히 활용
  - E2<E0: GLM predicted FC(R²≈0.5)는 actual full-scan FC보다 열등한 reference
  - BS-NET은 이미 최적 teacher(같은 subject의 full scan)를 보유 → 외부 prediction 불필요
  - TSD 프레임워크 = BS-NET의 작동 원리를 설명하는 이론적 렌즈 (실용적 개선 아님)

### 논문 작성

- [ ] Abstract, Introduction, Methods, Discussion, Limitations 집필
- [ ] Methods에 Fisher z-space correction 기술 포함
- [ ] Methods에 task-residual FC 방법론 기술 (ds007535, Cole 2014, Gratton 2018)
- [ ] Supplementary: 4-method ceiling comparison table/figure

## Key Experimental Findings

- **Component Necessity — Synthetic** (n_rois=50, 900/120): SB 제거 시 Δρ=-0.307, Prior 제거 시 Δρ=-0.187 → 두 구성요소가 핵심
- **Component Necessity — ABIDE pilot** (N=3, CC200, Fisher z): SB 제거 Δρ=-0.092, Prior 제거 Δρ=-0.046 — 실데이터에서도 동일 패턴 확인
- **Noise Degradation**: noise≈1.0 (SNR=1:1)에서 성능 경계 확인
- **Shrinkage**: LW와 OAS 성능 차이 미미 (짧은 시계열에서)
- **Stationarity**: ICC=-0.16으로 parallel test 가정 불충족 → Cheng et al. (2021) 선례로 방어
- **ABIDE PCP 실증 — Original** (N=468, CC200): r_FC=0.771±0.071 → ρ̂T=0.967±0.058, ceiling 53.6%
- **ABIDE PCP 실증 — Fisher z** (N=468, CC200): r_FC=0.771±0.071 → ρ̂T=0.843±0.036, ceiling 0%
- **Ceiling Effect (Track G)**: Original 85% ceiling → Fisher z 0% (pilot N=20)
  - Original: ρ̂T=0.993 (overcorrection via hard clip)
  - Fisher z: ρ̂T=0.868 (principled, no ceiling)
  - Partial α=0.5: ρ̂T=0.885 (damped correction)
  - Soft clamp: ρ̂T=0.798 (tanh compression, over-dampened)
- **ABIDE PCP — CC400 Fisher z** (N=468): r_FC=0.757±0.071 → ρ̂T=0.834±0.037, 97.4% improved
- **Atlas robustness**: CC200 ρ̂T=0.843 vs CC400 ρ̂T=0.834 — ROI 수 2배 증가에도 성능 동등
- **Multi-seed stability (Fisher z, 10 seeds)**: ρ̂T=0.843±0.036, seed std=0.005, ceiling 0%
- **Multi-seed stability (Original, 10 seeds)**: ρ̂T=0.967±0.058, seed std=0.006, ceiling 53.6%
- **ADHD-200 실증 — Fisher z** (N=40, CC200→195 ROIs): r_FC=0.819±0.090 → ρ̂T=0.866±0.047, 77.5% improved
- **ADHD-200 실증 — Fisher z** (N=40, CC400→356 ROIs): r_FC=0.803±0.098 → ρ̂T=0.855±0.051
- **ADHD-200 multi-seed (10 seeds, Fisher z)**:
  - CC200: ρ̂T=0.866±0.047, seed std=0.0043, ceiling 0%
  - CC400: ρ̂T=0.855±0.051, seed std=0.0046, ceiling 0%
- **ADHD group comparison (Fisher z, multi-seed)**:
  - CC200: ADHD(n=20) ρ̂T=0.868 vs Control(n=20) ρ̂T=0.864 — 그룹 무관 일관 보정
  - CC400: ADHD(n=20) ρ̂T=0.857 vs Control(n=20) ρ̂T=0.854
- **Cross-dataset consistency**: ABIDE (N=468) ρ̂T=0.843 vs ADHD (N=40) ρ̂T=0.866 — 독립 데이터셋 간 일관된 개선
- **ADHD-200 PCP Full (N=399, CC200, Fisher z, 10 seeds)**: r_FC=0.525±0.087 → ρ̂T=0.725±0.049, Δ=0.201, 100% improved
  - 6 sites: NYU(79), NeuroIMAGE(48), OHSU(78), Peking_1(85), Peking_2(67), Peking_3(42)
  - Group: ADHD(186) ρ̂T=0.721 vs Control(213) ρ̂T=0.729
- **Downstream Analysis (N=399)**: LW shrinkage 단독 개선 미미 (Δ<0.01 전 항목)
  - SVM: raw=0.560, LW=0.560, ref=0.600
  - Fingerprinting: 83.0% ID rate (2min scan → N=399 중 자기 자신 식별)
- **ρ̂T Dose-Response (N=399, tertile stratification)**:
  - T1(ρ̂T=0.670): FC_sim=0.430, Cohen_d_r=0.317, SVM=0.502
  - T2(ρ̂T=0.730): FC_sim=0.534, Cohen_d_r=0.408, SVM=0.550
  - T3(ρ̂T=0.775): FC_sim=0.610, Cohen_d_r=0.538, SVM=0.556
  - **3/3 monotonicity 확인** → BS-NET = reliability estimator, not FC denoiser
- **ds007535 (SpeechHemi)**: OpenNeuro, 56 subjects, task fMRI (speech lateralization), 450 vols × TR=2s = 900s (15min)
  - fMRIPrep 25.1.4 전처리 완료 상태로 제공 (MNI152NLin2009cAsym)
  - 데이터 구조: `sub-XX/func/` 하위에 직접 배치 (derivatives 폴더 없음)
  - `bold.nii.gz` = 전처리된 MNI space BOLD, `desc-confounds_timeseries.tsv` = fMRIPrep confounds
  - Task-residual FC 접근: HRF-convolved task regressors + 36P regress out → residual FC ≈ rest FC (r>0.9)
  - 근거: Cole et al. (2014, DOI: 10.1016/j.neuron.2014.05.014), Gratton et al. (2018, DOI: 10.1016/j.neuron.2018.03.035)
- **τ_min (Minimum Common Bootstrap Duration)**: hemodynamic 0.01 Hz의 2 cycles ≥ 200s, 실용적 하한 ~120s
  - ρ̂T(τ) plateau에서 τ_min 정의 가능 — duration sweep으로 empirical estimation
  - 코호트/조건별 가변적이나 BS-NET이 판단 도구 제공
- **Track H: ADHD vs Control Classification** (N=40, Linear SVM, 5-fold×5 repeats):
  - CC200: Raw Acc=0.710, BS-NET Acc=0.720 (+1.0pp), Reference Acc=0.625
  - CC400: Raw Acc=0.685, BS-NET Acc=0.700 (+1.5pp), Reference Acc=0.610
  - Reference FC paradox: 짧은 총 스캔(154–352s) + LW regularization이 SVM 고차원 분류에 유리
- **Component Necessity — ABIDE 실데이터** (N=468, CC200, 10 seeds):
  - Full Pipeline: ρ̂T=0.860±0.061
  - w/o Spearman-Brown: Δ=−0.116 (CRITICAL)
  - w/o Bayesian Prior: Δ=−0.051 (CRITICAL)
  - w/o Ledoit-Wolf: Δ=−0.002 (negligible)
  - Synthetic과 동일 패턴: SB > Prior > Attenuation > Bootstrap > LW
- **Progressive Ablation — ds000243** (N=52, CC200, 10 seeds):
  - L0(Raw)=0.698 → L1(+LW)=0.699 → L2(+Boot)=0.645 → L3(+SB+Att)=0.795 → L4(+Prior)=0.802 → L5(Full)=0.806
  - L2 dip: bootstrap resampling 효과 (effective sample shortening)
  - 4-level monotonic: L0→L1→L3→L5
- **Progressive Ablation — ABIDE** (N=468, CC200, 10 seeds):
  - L0=0.830 → L5=0.859, k≥3 필터링 후 monotonic pattern 확인
  - k<2: Δ≈0 (short≈total), k≥4: Δ scales with k (dose-response)
  - ABIDE k range: 1.3 (OHSU) – 5.3 (UM_2), 20 sites heterogeneity
- **k-Filtering Decision**: Main Fig 2에 k≥3 적용 (N=223), k<3은 Supplementary S1/S2에서 분석
- **Reliability-Aware Clustering** (ADHD-200 PCP N=399): ρ̂T 기반 strata별 k-means, permutation p-value, EXPLORATORY 가드레일
- **Reliability-Aware Classification** (ADHD-200 PCP N=399): LOSO + stratified k-fold, tangent FC, ρ̂T hard gate (fold-train-only threshold)
- **Keane ds003404/ds005073**: BP vs SZ classification pipeline
  - Confirmatory (Primary) + Exploratory family 분리, Holm/BH-FDR multiple comparison correction
  - Hard gate: fold별 train-only threshold 적용 (data leakage 방지)
  - min-valid-splits 옵션: 유효 split 부족 시 설정 자동 스킵
  - non-finite ρ̂T 대상 사전 드롭 + 경고 로그
- **Convergence Validation** (ds000243, N=49, 4S256Parcels, 100bs × 10seeds):
  - τ_short=60s: ρ̂T=0.756±0.032, r_FC peak=0.566 (300s), gap=0.190, 49/49 positive
  - τ_short=120s: ρ̂T=0.771±0.032, r_FC peak=0.638 (240s), gap=0.134, 25/25 positive
  - τ_short=180s: ρ̂T=0.769±0.036, r_FC peak=0.657 (180s), gap=0.112, 46/46 positive
  - Non-stationarity: r_FC peaks at τ_ref≈240-300s then declines — ρ̂T bypasses this
- **τ_min Empirical** (18-point fine-grained):
  - Peak ρ̂T=0.774 at τ_short=150s
  - 95% threshold at 40s, plateau [90, 180]s
  - Seed SD decreases monotonically: 0.011 (30s) → 0.003 (150s)
  - Reference artifact: ρ̂T drops at τ_short≥240s (insufficient τ_ref for short-scan subjects)
- **TSD (Temporal Self-Distillation) 이론 검토** (상세: `docs/5.16_tsd_theory_v2.md`):
  - BS-NET을 5-category distillation으로 재해석: Cat 1(Statistical-bootstrap), Cat 2(Knowledge-Guo GLM), Cat 3(Self-temporal split), Cat 4(Cross-Architecture-SB+attenuation), Cat 5(Temporal-SB prophecy)
  - w*_B: bootstrap ensemble FC median = implicit distillation target (Cat 1+3)
  - w*_G: within-dataset Ridge regression short→long FC predictor (Guo 2023 design, Cat 2)
  - E0–E3 ablation: E0(baseline BS-NET) → E1(w*_B explicit) → E2(w*_G) → E3(combined)
  - 6건 권고사항 반영 완료: Cat 3 Teeuw reframing, analytic distillation qualifier, Xiang 저자명, Guo Tier 1 승격, E0→E1 전략, ds000243 non-stationarity
  - **ds000243 non-stationarity**: r_FC peaks at τ_ref≈240-300s then declines → 30분 FC ≠ 최적 teacher. ds000243의 가치 = 넓은 τ_short sweep 범위. ρ̂T는 이 non-stationarity를 SB prophecy로 bypass

## Conventions

### Code Style

- Python 3.9+, PEP 8, Black (88 cols), isort, ruff
- Type hints, Google Style docstrings, snake_case
- N806/N815 exemption: neuroimaging domain variables (fc_true_T, rho_hat_T, G, C, L)
- Tests: pytest with mock-based dry-run (nilearn/nibabel not required)

### Metric Naming Convention

| Context | 외삽 신뢰도 (pipeline output) | FC 일치도 (validation) |
| ------- | ----------------------------- | ---------------------- |
| LaTeX   | `\hat{\rho}_T`              | `r_{\mathrm{FC}}`    |
| Inline  | ρ̂T                         | rFC                    |
| Code    | `rho_hat_T`                 | `r_fc`               |

### Ruff Configuration (pyproject.toml)

```toml
[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM"]
ignore = ["E501", "N815"]

[tool.ruff.lint.per-file-ignores]
"tests/*.py" = ["N802", "N803", "N806"]
```

## Project Structure

```
bsNet/
├── src/
│   ├── core/          # config, pipeline, bootstrap (4 correction methods), stats, simulate
│   ├── data/          # data_loader, synthetic data generator
│   ├── scripts/       # see src/scripts/README.md
│   │   ├── [Validation]  run_abide_bsnet.py, run_nilearn_adhd_bsnet.py, run_fmriprep_bsnet.py, run_duration_sweep.py
│   │   ├── [Defense]     run_{sensitivity,ablation,stationarity,shrinkage,...}.py (Track A–G)
│   │   ├── [Convergence] run_convergence_validation.py, run_progressive_ablation.py, run_abide_duration_sweep.py
│   │   ├── [Downstream]  run_downstream_analysis.py, run_reliability_aware_{classification,clustering}.py
│   │   ├── [Keane]       run_keane_bsnet_{recompute,classification}.py, run_keane_fc_classification.py, convert_keane_restfc_to_npz.py
│   │   ├── [TSD]         run_tsd_ablation.py (E0–E3 Temporal Self-Distillation ablation)
│   │   ├── [Data]        index_openneuro_hc.py, download_hc_100.py, download_adhd200_pcp.py, convert_{adhd200_pcp,xcpd_to_npy}.py
│   │   ├── [Viz]         plot_abide_results.py, plot_adhd_results.py, analyze_{ceiling_effect,fc_stratification}.py
│   │   ├── [Simulation]  run_synthetic_baseline.py, sweep_simulation.py
│   │   ├── [Utility]     inspect_craddock_atlas.py, visualize_fc_threshold.py
│   │   ├── [Preprocess]  preprocess_ds007535.py, preprocess_ds000243.py, setup_and_preprocess.py
│   │   └── [Pipeline]    run_fmriprep_{batch,keane,manual}.sh, run_xcpd_{batch,ds000243}.sh,
│   │                      run_keane_streaming_pipeline.sh, run_ds000243_batch.sh, run_all_pipeline.sh,
│   │                      setup_keane_datalad.sh, install_datalad.sh, setup_local_env.sh
│   └── visualization/ # Fig 1–6, FigS1–S3/S6–S7, style.py, legacy/
├── tests/             # pytest (74 tests)
├── docs/              # 6-category docs (1.x theory ~ 6.x ops), see docs/INDEX.md
│   └── figure/        # Fig1–6, FigS1–S3, FigS6–S7 PNG files
├── data/abide/        # ABIDE PCP cached time series + results
├── data/adhd/         # ADHD-200 cached time series + results
├── data/ds007535/     # SpeechHemi: raw/ (DataLad), timeseries_cache/, results/
├── data/ds000243/     # WashU resting-state: raw/, timeseries_cache/, results/
├── data/ds005073/     # Keane BP/SZ: results/ (keane_restfc_combined.npz, classification CSVs)
├── artifacts/reports/ # experiment result CSVs
└── pyproject.toml
```

## Next Session TODO

### Priority 1: 논문 작성
1. **논문 Methods 초고**: convergence validation + τ_min 수식 + TSD 프레임워크 기술 (negative result 포함)
2. **논문 Discussion**: τ_min + "Why ~2 minutes?" + non-stationarity + TSD 이론적 의의

### Priority 2: Keane + 추가 검증
3. **Keane 결과 수집** (서버): `run_keane_bsnet_classification.py` 실행 결과 확인 (BP vs SZ accuracy, permutation p-values)
4. **Keane 결과 논문 반영**: Supplementary 또는 Discussion에 cross-disorder generalization 결과 기술

### Priority 3: Figure + 기타
5. Figure 4 (Network Structure Preservation): ds000243 기반 topology/community 분석
6. TSD docx 최종 보고서 출력 (tracked changes 반영)

## Key References

- Cheng et al. (2021): Split-half + CTT framework on HCP N=1003, DOI: 10.1016/j.neuroimage.2021.118005
- Friedman et al. (2008): Within-session ICC > 0.95, DOI: 10.1016/j.neuroimage.2008.02.005
- Pitsialis et al. (2022): LW vs OAS shrinkage, DOI: 10.1016/j.neuroimage.2022.119244
- Noble et al. (2019): Cross-session ICC=0.29 (다른 개념임에 주의)
- Shou et al. (2014): Reliability correction for FC (z-space attenuation), DOI: 10.1016/j.neuroimage.2015.10.011
- Teeuw et al. (2021): Reliability modelling of resting-state FC, DOI: 10.1016/j.neuroimage.2021.117842
- Zimmerman (2007): Correction with biased reliability estimates, DOI: 10.1177/0013164406299132
- Cole et al. (2014): Task-evoked vs intrinsic FC architecture, DOI: 10.1016/j.neuron.2014.05.014
- Gratton et al. (2018): FC dominated by stable individual factors, DOI: 10.1016/j.neuron.2018.03.035
- Guo et al. (2023): GLM-based short→long FC prediction (within-dataset), DOI: 10.1093/cercor/bhac519
- Ellis (2024): SB prophecy convergence without parallel test assumption, DOI: 10.1007/s11336-024-09960-9
- Hinton et al. (2015): Knowledge distillation, arXiv:1503.02531
- Pieper et al. (2023): data2vec EMA self-distillation for brain decoding

## Correction Method Selection Guide

BS-NET `correct_attenuation()` 함수의 `method` 파라미터:

| Method             | 코드             | 설명                                 | 논문 근거                 | Ceiling 해소      |
| ------------------ | ---------------- | ------------------------------------ | ------------------------- | ----------------- |
| Original           | `"original"`   | 표준 CTT 보정 + hard clip            | Spearman (1904)           | ✗ (85%)          |
| **Fisher z** | `"fisher_z"`   | z-space에서 가법 보정 → tanh 역변환 | Shou (2014), Teeuw (2021) | **✓ (0%)** |
| Partial            | `"partial"`    | α=0.5 감쇠 보정                     | Zimmerman (2007)          | ✓ (0%)           |
| Soft clamp         | `"soft_clamp"` | tanh 압축 (순위 보존)                | —                        | ✓ (0%)           |

**권장**: `"fisher_z"` (학술적으로 가장 방어 가능, ceiling 완전 해소, 의미 있는 improvement 유지)

## XCP-D Atlas 명칭 가이드 (v26.x NIfTI 모드)

XCP-D v26.x NIfTI 처리 시 `Schaefer200`/`Schaefer400` 이름은 인식되지 않음.
내장 아틀라스는 **Schaefer (피질) + Tian (피질하) 결합 4S 시리즈**로 제공됨.

| BS-NET 아틀라스 | XCP-D 4S 이름    | 피질 ROI | 피질하 ROI | 총 ROI |
| --------------- | ---------------- | -------- | ---------- | ------ |
| schaefer200     | `4S256Parcels` | 200      | 56         | 256    |
| schaefer400     | `4S456Parcels` | 400      | 56         | 456    |
| schaefer100     | `4S156Parcels` | 100      | 56         | 156    |
| schaefer300     | `4S356Parcels` | 300      | 56         | 356    |

**XCP-D Docker 실행 옵션** (ds000243, NIfTI 모드):

```bash
docker run --rm \
  -v /path/to/fmriprep:/data:ro \
  -v /path/to/xcpd:/out \
  -v /path/to/work:/work \
  pennlinc/xcp_d:latest \
  /data /out participant \
  --mode linc --input-type fmriprep \
  --file-format nifti \
  -p 36P --fd-thresh 0.5 \
  --lower-bpf 0.01 --upper-bpf 0.1 \
  --smoothing 0 --combine-runs y \
  --atlases 4S256Parcels 4S456Parcels \
  --skip connectivity --min-time 120 \
  --nprocs 8 --mem-mb 16000 -w /work --notrack
```

**필수 전처리**: fMRIPrep 출력에 native-space T1w symlink 필요

```bash
# fMRIPrep이 MNI-space T1w만 출력한 경우
cd data/derivatives/fmriprep/sub-XXX/anat/
ln -s sub-XXX_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz \
      sub-XXX_desc-preproc_T1w.nii.gz
ln -s sub-XXX_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz \
      sub-XXX_desc-brain_mask.nii.gz
```

**convert_xcpd_to_npy.py ATLAS_NAME_MAP** — 4S 시리즈 키:

- `"4S156Parcels"` → `"4s156parcels"`
- `"4S256Parcels"` → `"4s256parcels"`
- `"4S356Parcels"` → `"4s356parcels"`
- `"4S456Parcels"` → `"4s456parcels"`
