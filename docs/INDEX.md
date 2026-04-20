# BS-NET Documentation Directory

이 폴더(`docs/`)는 BS-NET 파이프라인의 핵심 이론부터 최종 논문화 보고서까지 모든 문서를 **6개 카테고리(1~6) × 하위 목차(X.Y)** 형태로 구조화한 인덱스입니다.

---

## 1. 기저 이론 및 파이프라인 아키텍처 (Theory & Architecture)
- **`1.1_theory_concept.md`**: Spearman-Brown 기반 상관계수 예측 가설 및 동적 블록 부트스트래핑 등 알고리즘의 수학적 이론 증명.
- **`1.2_arch_pipeline.md`**: 원시 fMRI 전처리(Nuisance Regression)부터 공분산 분석 모델까지의 파이프라인 구조 설계도 및 참조 링크 모음.

## 2. 연구 실험 및 디버깅 로그 (Experiment Logs)
- **`2.1_log_experiment_20260326.md`**: 1차 튜닝. 단일 가상 모델 지표 산출, 로컬 캐시 구조 개선, 120초 기반 예측 시계열 최적화 등 초기 시뮬레이터 가동 기록.
- **`2.2_log_experiment_20260327.md`**: 2차 실증. N=100 대규모 Scale-Up(OpenNeuro) 검증 로직 가동 기록, KDE Bias 편향 보정, Figure 2 시각화 완료 로그.
- **`2.3_log_sessions.log`**: 토폴로지 검증(Phase 4~7) 세션별 실행 이력. Schaefer 400 전환, ARI/Jaccard 산출, Violin plot 생성 등 단문 기록.
- **`2.4_log_experiment_20260328_30.md`**: 3차 실증 (3일분 통합). 방어 실험 Track A–G 완성, Ceiling effect 발견 및 Fisher z 해소, ABIDE N=468 CC200/CC400 검증, ADHD-200 N=40 그룹 비교, Track E 배치 리팩토링.
- **`2.5_log_experiment_20260401.md`**: 4차 작업 (세션 3). Figure 1/2/7 재설계, style.py ATLAS_META 통합, ds000243 파이프라인 인프라 구축, run_duration_sweep.py ds000243 지원 추가.

## 3. 정량 분석 결과 및 시각화 캡션 (Validation Results & Legends)
- **`3.1_res_figure_legends.md`**: 메인 논문 Figure 1~7 canonical 레전드. (resting-state 중심, Figure 4 sliding-window 포함)
- **`3.2_res_abide_figure_legends.md`**: ABIDE 확장 보충 캡션(Supplementary S*). CC400 확장, ceiling-method 비교.
- **`3.3_res_adhd_figure_legends.md`**: ADHD 확장 보충 캡션(Supplementary S*). PCP N=399 기반 S6(비지도 utility), S7(LOSO supervised utility) 포함.
- **`3.4_res_classification_legend.md`**: Main Figure 7(Track H) 상세 레전드. Linear SVM 3-condition × 2-atlas 비교, Reference FC paradox 해석.

## 4. 퍼블리케이션 및 최종 보고서 (Final Reports)
- **`4.1_pub_report_academic.md`**: BS-NET의 Phase 1~4(이론 증명, 대규모 실증, 토폴로지 검증) 전 과정을 피어리뷰 학술 저널 형식으로 정리한 영문 학술 보고서.
- **`4.2_pub_report_general.md`**: 비전문 대중/경영진 대상의 한국어 브리핑 보고서. 2분 스캔으로 15분 결과를 재현하는 BS-NET의 임상적 의의를 풀어 설명.

## 5. 개발 계획 및 유지보수 (Development Plans)
- **`5.1_plan_refactoring.md`**: 코드 구조 분석 및 리팩토링 계획서. 중복 코드 추출, 매직 넘버 중앙화, 시각화 통합, 프로젝트 정리 등 4단계 로드맵.
- **`5.2_review_critical_analysis.md`**: Critical review — 통계, 실험 설계, 스토리라인 전반의 13개 이슈 (CRITICAL 4, MAJOR 7, MINOR 2) 및 DOI 16건.
- **`5.3_defense_plan.md`**: 방어 실험 계획서. Track A–D 4개 병렬 트랙 설계.
- **`5.4_defense_responses.md`**: 방어 실험 결과 및 코멘트 대응. Track A–F 6개 실험 결과, 해석, 논문 반영 방향.
- **`5.5_methods_preprocessing.md`**: Methods 섹션용 전처리 파이프라인 상세 기술 (fMRIPrep, XCP-D, Schaefer 400, LW shrinkage). DOI 8건.
- **`5.6_failure_characterization.md`**: 9% 실패 피험자 특성 분석. N=300 (3 noise levels) 시뮬레이션, SNR≈1:1까지 100% pass 확인.
- **`5.7_stationarity_discussion.md`**: Stationarity test 결과 논문 반영 방식 확정. Cheng et al. (2021) 선례 기반 방어 논증 3단계.
- **`5.8_ceiling_effect_correction.md`**: Track G — Ceiling effect 원인 분석 및 보정. Fisher z-space correction(Shou 2014, Teeuw 2021)을 포함한 4-method 비교. ABIDE N=468 실증에서 original 85% ceiling → Fisher z 0% 해소. DOI 5건.
- **`5.9_manuscript_why_questions_and_figure_map.md`**: 논문 집필 전용 why 질문 체크리스트 + 메인 Figure 1–7 canonical map + 번호 충돌 정리 규칙.
- **`5.10_figure_design_audit_20260410.md`**: Figure 디자인 재점검 리포트. 메인/추가 Figure 최종 수정일·수정 횟수·완성도 및 공통 디자인 수정안(P0/P1/P2) 포함.
- **`5.11_storyline_figure_decision_matrix.md`**: 논문 스토리라인 우선 고정 + Figure별 재활용/신규/아카이브 의사결정 매트릭스.
- **`5.12_patient_utility_reliability_aware_clustering.md`**: 환자 데이터 활용성 보강 계획. `ρ̂T` strata 기반 HC/Patients 분리(비지도) 분석 설계, nilearn FC 방법 비교, 결과 figure 설계.
- **`5.13_adhd_discrimination_citation_guide.md`**: ADHD vs HC discrimination을 위한 인용 중심 작성 가이드. tangent-linear 권장 조합, site/confound 통제, permutation/LOSO 권장, claim-to-citation 매핑 및 본문 템플릿 포함.
- **`5.14_keane_reliability_gated_discrimination_design.md`**: Keane(BP/SZ) reliability-gated 분류 설계 문서. confirmatory/exploratory family, train-only threshold(누수 방지), split validity(최소 클래스 수), Holm/FDR 보정 규칙 포함.

## 6. 운영 및 파이프라인 가이드 (Operations & Pipeline)
- **`6.1_ops_local_setup.md`**: 로컬 환경 설정 가이드. Python venv/conda, 의존성 설치, Schaefer atlas 배치, FreeSurfer 라이선스 등.
- **`6.2_ops_real_data_pipeline.md`**: OpenNeuro HC 100명 실증 분석 전체 파이프라인. Step 0(환경) ~ Step 5(요약) 및 환경별 워크플로우(단일 머신, 랩탑+워크스테이션, HPC).

---

### 스크립트 인덱스
- **`src/scripts/README.md`**: 전체 스크립트 분류 인덱스 (7개 카테고리, CLI 레퍼런스, 의존성 그래프)

---

### 부속 자산 (Assets)
- **`figure/`**: 최종 렌더링 이미지 (PNG). 배포용 사본.

#### 메인 Figure (논문 본문용, 7개)

| # | 파일명 | 내용 | 스크립트 |
|---|--------|------|---------|
| Fig 1 | `Figure1_ds000243_DurationSweep.png` | Duration Sweep — resting-state(ds000243) 기준 메인 정본 | `plot_figure1_ds000243.py` |
| Fig 2 | `Figure2_Validation_ds007535.png` | Empirical Validation — ds007535 N=30 (6 atlas, full BS-NET pipeline, 4-panel) | `plot_figure2_validation.py` |
| Fig 3 | `Figure3_ComponentNecessity.png` | Component Necessity — ABIDE N=468 실데이터 violin+jitter (SB/Prior/LW leave-one-out) | `plot_figure3_component.py` |
| Fig 4 | `Figure4_Structure_Preservation.png` | Network Structure Preservation + sliding-window stability (본문 포함) | `plot_figure4_structure.py` |
| Fig 5 | `Figure5_ABIDE_Validation.png` | ABIDE Empirical Validation (N=468, CC200, 10 seeds, Fisher z, 4-panel) | `plot_figure5_abide_v2.py` |
| Fig 6 | `Figure6_ADHD_Validation.png` | Cross-Dataset Generalization (N=40, CC200, 10 seeds, Fisher z, 4-panel) | `plot_figure6_adhd_v2.py` |
| Fig 7 | `Figure7_ADHD_Classification.png` | Clinical Classification — grouped bar (CC200/CC400 hatching, 3 FC conditions) | `plot_figure7_classification.py` |

#### 보충 Figure (Supplementary)

| # | 파일명 | 내용 |
|---|--------|------|
| Fig S1 | *(생성 예정)* | ABIDE CC400 멀티시드 확장 |
| Fig S2 | *(생성 예정)* | ABIDE ceiling-method 비교 (CC200) |
| Fig S3 | *(생성 예정)* | ABIDE ceiling-method 비교 (CC400) |
| Fig S6 | `FigS6_Reliability_Aware_Clustering.png` | ADHD-200 PCP reliability-aware clustering utility (exploratory) |
| Fig S7 | `FigS7_Reliability_Aware_Classification.png` | ADHD-200 PCP LOSO supervised discrimination utility (exploratory) |
