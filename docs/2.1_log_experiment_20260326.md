# bsNet Experiments Note (1차 정리, 2026-03-26)

## 1) 목적과 범위
- **목적**: BS-NET PoC(Proof of Concept)의 단시간(2분) fMRI 데이터 기반 장시간(15분) FC 상관계수 예측 파이프라인 최적화 및 타당성 검증.
- **범위**: 
  - 1단계: 선형 가상 시뮬레이터(AR 모델) 테스트를 통한 통계적 최적화(Shrinkage, Bayesian)
  - 2단계: 실제 오픈 데이터(OpenNeuro `ds000030`, `ds000243`) 기반 100건 대상으로의 무작위 스케일업(Scale-up) 확장 검증.

## 2) 시계열별 실험 수행 기록
| 수행 내용 | 결과 및 지표 요약 |
| --------- | ---- |
| **초기 예측 시뮬레이션** | 단순 Spearman-Brown 공식만 적용 시 예측 정확도가 57~66% 수준에 그침을 발견 |
| **성능 최적화 달성** | Ledoit-Wolf Shrinkage, 동적 블록(Dynamic Block) 역산, 베이지안 Empirical Prior(0.25) 결합 적용 → **예측 정확성 86.9% 돌파 (오류 0.028)** |
| **Phase 2 (단일 심층/오픈)** | OpenNeuro 로컬 캐시(`MoBSE` 레퍼런스 데이터)를 활용한 1호 피험자(sub-10159) 교차 검증 → **예측도 83.1% 달성** |
| **Phase 3 (다중 스케일업)** | 코호트 스크립트(`run_real_data_scale.py`) 배포 및 무작위 다중 검증 → **9건 코호트 검증 결과 평균 정확도 86.8%, 80% 통과율 100% 달성** |
| **Clean Fetching 재개** | 로컬 캐시 참조(치팅 논란) 배제 조치. OpenNeuro GraphQL에서 100명을 실시간으로 새로 다운로드 하도록 코드 리팩터링 |

## 3) 핵심 알고리즘 통합 내역
### A. 저신뢰도 노이즈 제어
- `bootstrap.py`: 기존 경험적 고정 블록 길이에 의존하지 않고, 데이터 자체의 AR(1) 자기상관성 붕괴 시간(Lag)을 직접 추적하여 블록을 분할하는 Politis & White(2004) 모듈 탑재.
- `data_loader.py`: 짧은 샘플 구조에서 분산 노이즈가 폭발하는 것을 정규화(압축)시키는 Ledoit-Wolf 수축 도구 도입.
- `main.py`: Empirical Bayesian Prior 통계치를 삽입하여 쪼개기(Split-half) 샘플의 붕괴를 하드웨어가 방어하도록 수정.

### B. 실 데이터 수집 & 병목 돌파
- `run_real_data_scale.py`: `MoBSE`의 OpenNeuro GraphQL 인덱서를 이식. `participants.tsv` 누락 등 메타데이터 불량을 회피(`strict_hc=False`)하고 `normal, healthy control, ctrl, hc` 등 광범위한 진단명으로 필터 정밀 타격 지원.
- 전처리 파이프라인 정착: `build_paper_nuisance_confounds` 함수를 이식해 GSR(Global Signal Regression), CompCor 상위 컴포넌트 축출, Polynomial Detrending 등을 한 번에 묶어 `Schaefer(n_rois=100)` 아틀라스에 투입.

## 4) 직면했던 문제들과 돌파 전략 (Troubleshooting)
1. **과도하게 억눌린 상관관계 부스팅 통계오류**
   - **대응**: 부스팅 승수가 낮아지는 것을 막기 위해, 초기 상관 행렬 공식에 대규모 데이터의 경험적 사전 확률(Empirical Prior) `(0.25, 0.05)`을 가중 타격하여 예측 성능 보존.
2. **MoBSE 연동 간 의존성(Dependency) 격리 충돌**
   - **대응**: `bsNet` 가상 컨테이너 환경(`venv`)에 의존성을 일일이 복사하는 대신 뼈대가 되는 `MoBSE` 로컬 패키지를 `-e`로 직접 Install 시켜 원활한 API 호출 인프라 확보.
3. **Nilearn Schaefer ROI 인자 거부**
   - **대응**: `nilearn` 마스커의 최소 요구치가 100 노드부터 시작됨을 포착, 50에서 100(기본 임상단위)으로 즉각 보정(`n_rois=100`).
4. **로컬 데이터 캐시 의존 회피 (Clean-Run 명시)**
   - **대응**: 0베이스 테스트 구축을 위해, MoBSE 공유 캐시 폴더(`cache_dir = Path("/Users/hwon/GitHub/MoBSE/...)`)를 차단. 빈 폴더(`data/openneuro`)로부터 OpenNeuro API 통신 패킷을 새롭게 개시하도록 백그라운드 데몬 가동 지시.

## 5) 현재 상태 및 관찰 지점
- 현재 완전히 깨끗한 새 스크립트 환경이 백그라운드 프로세스(터미널 백그라운드)에서 GraphQL 트리를 순차 연산하며 BOLD NIfTI 및 Confounds 데이터를 다운로드 가동 중입니다. 
- 이 크롤링 연산이 수 시간 뒤 무사히 안착하면, 코드는 별도의 지시 없이 자동으로 100회 부트스트래핑과 Schaefer 검증을 통과시킬 것입니다. 완료 후 결과는 산출 경로(`artifacts/reports/scale_up_100_results_[TIME].csv`)에 통계 파일로 적재됩니다.
