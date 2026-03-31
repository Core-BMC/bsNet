#!/usr/bin/env bash
# run_ds000243_batch.sh — ds000243 duration sweep, all atlases
#
# 사용법:
#   cd /path/to/bsNet
#   source .venv/bin/activate
#   bash src/scripts/run_ds000243_batch.sh
#
# 옵션 (환경변수로 오버라이드):
#   N_SEEDS=10        bash src/scripts/run_ds000243_batch.sh
#   N_JOBS=4          bash src/scripts/run_ds000243_batch.sh
#   ATLASES="schaefer200 schaefer400"  bash ...
#   FORCE=1           bash ...  # 기존 결과 덮어쓰기
#
# 기본 동작: 이미 집계 CSV가 있으면 skip

set -euo pipefail

# ── 설정 ─────────────────────────────────────────────────────────────────
DATASET="ds000243"
ATLASES="${ATLASES:-schaefer200 schaefer400 cc200 cc400 aal harvard_oxford}"
N_SEEDS="${N_SEEDS:-10}"
N_JOBS="${N_JOBS:-4}"
FORCE="${FORCE:-0}"
MIN_TOTAL_SEC=600
TARGET_DUR_MIN=15

RESULTS_DIR="data/${DATASET}/results"
SCRIPT="src/scripts/run_duration_sweep.py"

# ── 색상 출력 ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=== DS000243 Duration Sweep Batch ===${NC}"
echo    "  Dataset : ${DATASET} (WashU resting-state, N=50)"
echo    "  Atlases : ${ATLASES}"
echo    "  N_SEEDS : ${N_SEEDS}"
echo    "  N_JOBS  : ${N_JOBS}"
echo    "  FORCE   : ${FORCE}"
echo ""

TOTAL_START=$(date +%s)
DONE=0
SKIPPED=0

for ATLAS in ${ATLASES}; do
    AGG_CSV="${RESULTS_DIR}/${DATASET}_duration_sweep_${ATLAS}_aggregated.csv"
    RAW_CSV="${RESULTS_DIR}/${DATASET}_duration_sweep_${ATLAS}.csv"

    # ── skip-existing ────────────────────────────────────────────────────
    if [[ "${FORCE}" == "0" && -f "${AGG_CSV}" ]]; then
        echo -e "${YELLOW}[SKIP]${NC} ${ATLAS} — 기존 결과 있음 (${AGG_CSV})"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo -e "${GREEN}[RUN]${NC}  ${ATLAS} ..."
    ATLAS_START=$(date +%s)

    python3 "${SCRIPT}" \
        --dataset "${DATASET}" \
        --atlas   "${ATLAS}" \
        --n-seeds "${N_SEEDS}" \
        --n-jobs  "${N_JOBS}" \
        --min-total-sec "${MIN_TOTAL_SEC}" \
        --target-duration-min "${TARGET_DUR_MIN}"

    ATLAS_END=$(date +%s)
    ELAPSED=$((ATLAS_END - ATLAS_START))
    echo -e "  → 완료: ${ELAPSED}s  (raw: ${RAW_CSV})"
    DONE=$((DONE + 1))
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo -e "${CYAN}=== 배치 완료 ===${NC}"
echo "  실행: ${DONE}개  |  건너뜀: ${SKIPPED}개"
echo "  총 소요: ${TOTAL_ELAPSED}s"
