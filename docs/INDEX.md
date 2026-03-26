# BS-NET Documentation Directory

이 폴더(`docs/`)는 BS-NET 파이프라인의 핵심 이론부터 최종 논문화 보고서까지 모든 문서들을 **4개의 카테고리(1~4) 및 하위 목차 형태(1.1, 2.1 등)로 묶어 주제별, 용도별로 구조화한 전용 인덱스**입니다.

---

## 🧠 1. 기저 이론 및 파이프라인 아키텍처 (Theory & Architecture)
- **`1.1_theory_concept.md`**: Spearman-Brown 기반 상관계수 예측 가설 및 동적 블록 부트스트래핑 등 알고리즘 기반이 되는 수학적 이론 증명.
- **`1.2_arch_pipeline.md`**: 원시 fMRI 데이터 전처리(Nuisance Regression)부터 메인 공분산 분석 모델까지의 파이프라인 데이터 파이프 구조 설계도.

## 📓 2. 연구 실험 및 디버깅 로그 (Experiment Logs)
- **`2.1_log_experiment_20260326.md`**: 1차 튜닝. 단일 가상 모델 지표 산출, 로컬 캐시 구조 개선, 120초 기반 예측 시계열 최적화 로그 등 초기 시뮬레이터 가동 기록.
- **`2.2_log_experiment_20260327.md`**: 2차 실증. 100명 대규모 Scale-Up(OpenNeuro) 검증 로직 가동 기록 및 데이터 밀도분포(KDE) Bias 편향 보정 추가 등 실데이터 가동 로그.

## 📊 3. 정량 분석 결과 및 시각화 캡션 (Validation Results & Legends)
- **`3.1_res_optimal_duration.md`**: 한계 효용(Marginal Gain)과 불확실성 감소 분석을 통해 120초가 최적점임을 이론적으로 증명하고, 이를 기반으로 진행한 100명 코호트에 대한 결과(Figure 2)를 정리한 종합 실증 보고서.
- **`3.2_res_figure1_legends.md`**: 최적 스캔 시간 규명 및 정확도 상승 분석 등 임계점 한계를 매핑한 **Figure 1** 전용 아카데믹 레전드.
- **`3.3_res_figure2_legends.md`**: N=100 코호트 실증(91% 임상 통과율, 0.48 R^2) 밀도/오차 분석을 시각화한 대규모 결과물 **Figure 2** 전용 아카데믹 레전드.

## 🎓 4. 퍼블리케이션 및 최종 보고서 (Final Reports)
- **`4.1_pub_report_academic.md`**: BS-NET 모델링 기법의 이론, 증명, 결과를 피어리뷰(Peer-Review) 학술 저널에 즉각 게재할 수 있도록 정형화하여 완성한 영문 학술용 프레임워크 문서.
- **`4.2_pub_report_general.md`**: BS-NET 파이프라인 기술의 시장성과 임상적, 사회적 파급력을 일반 대중이나 비전문 경영진도 즉각적으로 이해할 수 있도록 쉽게 풀어 쓴 브리핑/기자용 요약본.

---

*이외에 생성된 최종 시각화 패널 파일(Figure 1, Figure 2)은 `docs/figure/` 폴더 내에 이미지 타겟으로 내보내기/저장됩니다.*
