#!/usr/bin/env bash
# Component Necessity (Track E) — ABIDE 전체 N=468, CC200 + CC400
# Usage: bash src/scripts/run_component_necessity_batch.sh [N_JOBS]

set -euo pipefail

N_JOBS="${1:-8}"
N_BOOTSTRAPS=50
SHORT_SAMPLES=60
METHOD="fisher_z"
N_SUBJECTS=0  # 0 = all

echo "=== Component Necessity Batch ==="
echo "N_JOBS=${N_JOBS}, N_BOOTSTRAPS=${N_BOOTSTRAPS}, SHORT_SAMPLES=${SHORT_SAMPLES}"
echo ""

for ATLAS in cc200 cc400; do
    INPUT_DIR="data/abide/timeseries_cache/${ATLAS}"
    if [ ! -d "${INPUT_DIR}" ]; then
        echo "[SKIP] ${INPUT_DIR} not found"
        continue
    fi

    N_FILES=$(ls "${INPUT_DIR}"/*.npy 2>/dev/null | wc -l)
    echo "[START] ${ATLAS}: ${N_FILES} subjects × 10 seeds × 6 conditions"
    echo "        $(date '+%Y-%m-%d %H:%M:%S')"

    python3 -m src.scripts.run_component_necessity \
        --input-dir "${INPUT_DIR}" \
        --n-subjects "${N_SUBJECTS}" \
        --short-samples "${SHORT_SAMPLES}" \
        --n-bootstraps "${N_BOOTSTRAPS}" \
        --correction-method "${METHOD}" \
        --n-jobs "${N_JOBS}"

    echo "[DONE]  ${ATLAS}: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo "=== All done ==="
ls -lh artifacts/reports/component_necessity_ABIDE_*.csv 2>/dev/null
