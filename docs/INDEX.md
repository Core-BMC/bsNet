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

## 3. 정량 분석 결과 및 시각화 캡션 (Validation Results & Legends)
- **`3.1_res_figure_legends.md`**: Figure 1~4 전체에 대한 통합 아카데믹 레전드. 패널별 축·통계량·해석을 포함한 마스터 캡션 문서.
- **`3.2_res_abide_figure_legends.md`**: ABIDE PCP 실증 Figure 5~7b 레전드. Multi-seed (CC200/CC400), Ceiling effect 4-method 비교.
- **`3.3_res_adhd_figure_legends.md`**: ADHD-200 실증 Figure 8~12 레전드. Single/multi-seed, group comparison (ADHD vs Control), atlas comparison.

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

## 6. 운영 및 파이프라인 가이드 (Operations & Pipeline)
- **`6.1_ops_local_setup.md`**: 로컬 환경 설정 가이드. Python venv/conda, 의존성 설치, Schaefer atlas 배치, FreeSurfer 라이선스 등.
- **`6.2_ops_real_data_pipeline.md`**: OpenNeuro HC 100명 실증 분석 전체 파이프라인. Step 0(환경) ~ Step 5(요약) 및 환경별 워크플로우(단일 머신, 랩탑+워크스테이션, HPC).

---

### 스크립트 인덱스
- **`src/scripts/README.md`**: 전체 스크립트 분류 인덱스 (7개 카테고리, CLI 레퍼런스, 의존성 그래프)

---

### 부속 자산 (Assets)
- **`figure/`**: 최종 렌더링 이미지 (PNG). 배포용 사본.

| Figure | 파일명 | 내용 |
|--------|--------|------|
| Fig 1 | `Figure1_Combined.png` | 최적 스캔 시간 증명 (Marginal Gain, CI Decay, Signal Overlay) |
| Fig 2 | `Figure2_Validation.png` | N=100 코호트 실증 (Scatter, KDE, Error, Pass Rate) |
| Fig 3 | `Figure3_Topology.png` | Schaefer 400 토폴로지 보존 (Degree, Path Length) |
| Fig 4 | `Figure4_*.png` | ARI, Modularity, Subnetwork Jaccard (Yeo-7 / Yeo-100) |
| — | `Figure_ComponentNecessity.png` | Component Necessity Analysis — ρ̂T per condition + Δρ waterfall |
| Fig 5a | `Figure5_ABIDE_MultiSeed_CC200.png` | ABIDE multi-seed CC200 (N=468, 10 seeds, Fisher z) |
| Fig 5b | `Figure5_ABIDE_MultiSeed_CC400.png` | ABIDE multi-seed CC400 (N=468, 10 seeds, Fisher z) |
| Fig 5c | `Figure5_ABIDE_Ceiling_CC200.png` | Ceiling effect 4-method comparison CC200 |
| Fig 5d | `Figure5_ABIDE_Ceiling_CC400.png` | Ceiling effect 4-method comparison CC400 |
| Fig 6a | `Figure6_ADHD_SingleSeed_CC200.png` | ADHD single-seed CC200 (N=40, group colored) |
| Fig 6b | `Figure6_ADHD_SingleSeed_CC400.png` | ADHD single-seed CC400 (N=40, group colored) |
| Fig 6c | `Figure6_ADHD_MultiSeed_CC200.png` | ADHD multi-seed CC200 (N=40, 10 seeds, group) |
| Fig 6d | `Figure6_ADHD_MultiSeed_CC400.png` | ADHD multi-seed CC400 (N=40, 10 seeds, group) |
| Fig 6e | `Figure6_ADHD_Atlas_Comparison.png` | ADHD atlas comparison CC200 vs CC400 |
