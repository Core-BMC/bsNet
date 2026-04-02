#!/usr/bin/env bash
# run_ds000243_batch.sh — ds000243 preprocess + duration sweep, all atlases
#
# 사용법:
#   cd /path/to/bsNet
#   source .venv/bin/activate
#   bash src/scripts/run_ds000243_batch.sh
#
# 옵션 (환경변수로 오버라이드):
#   N_SEEDS=10         bash src/scripts/run_ds000243_batch.sh
#   N_JOBS=4           bash src/scripts/run_ds000243_batch.sh
#   PREPROC_JOBS=16    bash ...  # preprocess 병렬수 (기본 16)
#   ATLASES="schaefer200 schaefer400"  bash ...
#   FORCE=1            bash ...  # 기존 결과 덮어쓰기
#
# 기본 동작:
#   1. atlas별 timeseries_cache가 없으면 preprocess_ds000243.py 실행
#   2. 이미 집계 CSV가 있으면 sweep skip

set -euo pipefail

# ── 설정 ─────────────────────────────────────────────────────────────────
DATASET="ds000243"
ATLASES="${ATLASES:-schaefer200 schaefer400 cc200 cc400 aal harvard_oxford}"
N_SEEDS="${N_SEEDS:-10}"
N_JOBS="${N_JOBS:-4}"
PREPROC_JOBS="${PREPROC_JOBS:-8}"
FORCE="${FORCE:-0}"
MIN_TOTAL_SEC=600
TARGET_DUR_MIN=15

FMRIPREP_DIR="data/${DATASET}/results/fmrirep"
CACHE_DIR="data/${DATASET}/timeseries_cache"
RESULTS_DIR="data/${DATASET}/results"
PREPROC_SCRIPT="src/scripts/preprocess_ds000243.py"
SWEEP_SCRIPT="src/scripts/run_duration_sweep.py"

# ── 색상 출력 ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=== DS000243 Preprocess + Duration Sweep Batch ===${NC}"
echo    "  Dataset      : ${DATASET} (WashU resting-state, N=52)"
echo    "  Atlases      : ${ATLASES}"
echo    "  N_SEEDS      : ${N_SEEDS}"
echo    "  N_JOBS(sweep): ${N_JOBS}"
echo    "  PREPROC_JOBS : ${PREPROC_JOBS}"
echo    "  FORCE        : ${FORCE}"
echo ""

TOTAL_START=$(date +%s)
DONE=0
SKIPPED=0

for ATLAS in ${ATLASES}; do
    CACHE_ATLAS_DIR="${CACHE_DIR}/${ATLAS}"
    AGG_CSV="${RESULTS_DIR}/${DATASET}_duration_sweep_${ATLAS}_aggregated.csv"
    RAW_CSV="${RESULTS_DIR}/${DATASET}_duration_sweep_${ATLAS}.csv"

    # ── Step 1: preprocess (atlas별 timeseries_cache 없으면 실행) ─────────
    if [[ -d "${CACHE_ATLAS_DIR}" ]]; then
        N_CACHED=$(find "${CACHE_ATLAS_DIR}" -name "*.npy" | wc -l | tr -d ' ')
    else
        N_CACHED=0
    fi
    if [[ "${FORCE}" == "1" || "${N_CACHED}" -lt 50 ]]; then
        echo -e "${GREEN}[PREPROC]${NC} ${ATLAS} (cached: ${N_CACHED}/52) ..."
        PREPROC_START=$(date +%s)
        FORCE_FLAG=""
        [[ "${FORCE}" == "1" ]] && FORCE_FLAG="--force"
        python3 "${PREPROC_SCRIPT}" \
            --input-dir   "${FMRIPREP_DIR}" \
            --atlas       "${ATLAS}" \
            --output-dir  "${CACHE_ATLAS_DIR}" \
            --n-jobs      "${PREPROC_JOBS}" \
            ${FORCE_FLAG}
        PREPROC_END=$(date +%s)
        echo -e "  → preprocess 완료: $((PREPROC_END - PREPROC_START))s"
    else
        echo -e "${YELLOW}[SKIP PREPROC]${NC} ${ATLAS} — cache 있음 (${N_CACHED} subjects)"
    fi

    # ── Step 2: duration sweep ────────────────────────────────────────────
    if [[ "${FORCE}" == "0" && -f "${AGG_CSV}" ]]; then
        echo -e "${YELLOW}[SKIP SWEEP]${NC} ${ATLAS} — 기존 결과 있음 (${AGG_CSV})"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo -e "${GREEN}[SWEEP]${NC}   ${ATLAS} ..."
    SWEEP_START=$(date +%s)

    python3 "${SWEEP_SCRIPT}" \
        --dataset "${DATASET}" \
        --atlas   "${ATLAS}" \
        --n-seeds "${N_SEEDS}" \
        --n-jobs  "${N_JOBS}" \
        --min-total-sec "${MIN_TOTAL_SEC}" \
        --target-duration-min "${TARGET_DUR_MIN}"

    SWEEP_END=$(date +%s)
    echo -e "  → sweep 완료: $((SWEEP_END - SWEEP_START))s  (${RAW_CSV})"
    DONE=$((DONE + 1))
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo -e "${CYAN}=== 배치 완료 ===${NC}"
echo "  실행: ${DONE}개  |  건너뜀: ${SKIPPED}개"
echo "  총 소요: ${TOTAL_ELAPSED}s"
