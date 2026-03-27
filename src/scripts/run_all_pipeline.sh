#!/usr/bin/env bash
# ============================================================================
# BS-NET: Full Real-Data Pipeline Runner
# ============================================================================
# 전체 파이프라인을 단계별로 실행하는 마스터 스크립트.
# 각 단계는 독립적으로 실행 가능하며, 이미 완료된 단계는 자동 스킵.
#
# 파이프라인 단계:
#   Step 0: 환경 설정 (Python venv + packages)
#   Step 1: HC 인덱싱 (7 OpenNeuro datasets → hc_adult_index.csv)
#   Step 2: 100명 다운로드 (hc_100_selection.csv 기반)
#   Step 3: fMRIPrep 전처리 (Docker/Singularity, 워크스테이션 권장)
#   Step 4: BS-NET 실행 (fMRIPrep 출력 → rho_hat_T)
#   Step 5: 결과 요약 및 시각화
#
# 사용법:
#   # 전체 실행
#   ./src/scripts/run_all_pipeline.sh
#
#   # 특정 단계만 실행
#   ./src/scripts/run_all_pipeline.sh --step 3    # fMRIPrep만
#   ./src/scripts/run_all_pipeline.sh --step 4    # BS-NET만
#
#   # Step 3 옵션: Singularity, 병렬 처리
#   ./src/scripts/run_all_pipeline.sh --step 3 --singularity --max-parallel 4
#
#   # Dry-run (전체)
#   ./src/scripts/run_all_pipeline.sh --dry-run
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

# ---- Parse arguments ----
STEP=""
DRY_RUN=0
SINGULARITY_FLAG=""
MAX_PARALLEL_FLAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)          STEP="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --singularity)   SINGULARITY_FLAG="--singularity"; shift ;;
        --max-parallel)  MAX_PARALLEL_FLAG="--max-parallel $2"; shift 2 ;;
        -h|--help)
            head -30 "$0" | grep "^#" | sed 's/^# //'
            exit 0
            ;;
        *)               EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---- Logging ----
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pipeline_${TIMESTAMP}.log"

log() {
    local msg="[$(date +%H:%M:%S)] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

# ---- Step functions ----

step0_setup() {
    log "━━━ Step 0: Environment Setup ━━━"

    if [ -d "$PROJECT_ROOT/.venv" ] && "$PROJECT_ROOT/.venv/bin/python" -c "import nilearn" 2>/dev/null; then
        log "  venv already configured, skipping"
        return 0
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        log "  [DRY-RUN] ./src/scripts/setup_local_env.sh --venv"
        return 0
    fi

    bash "$SCRIPT_DIR/setup_local_env.sh" --venv 2>&1 | tee -a "$LOG_FILE"
}

step1_index() {
    log "━━━ Step 1: Index HC Adults ━━━"

    local index_csv="$DATA_DIR/hc_adult_index.csv"
    if [ -f "$index_csv" ]; then
        local count
        count=$(wc -l < "$index_csv")
        log "  Index exists: $index_csv ($((count - 1)) subjects)"
        return 0
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        log "  [DRY-RUN] python src/scripts/index_openneuro_hc.py"
        return 0
    fi

    log "  Indexing 7 OpenNeuro datasets..."
    python "$SCRIPT_DIR/index_openneuro_hc.py" 2>&1 | tee -a "$LOG_FILE"
}

step2_download() {
    log "━━━ Step 2: Download 100 HC Subjects ━━━"

    local selection_csv="$DATA_DIR/hc_100_selection.csv"

    # Check if selection exists
    if [ ! -f "$selection_csv" ]; then
        log "  Selection CSV not found, will generate during download"
    fi

    # Count already downloaded subjects
    local downloaded=0
    if [ -f "$selection_csv" ]; then
        while IFS=, read -r ds sub _; do
            [[ "$ds" == "dataset_id" ]] && continue
            if [ -d "$DATA_DIR/openneuro/$ds/$sub" ]; then
                downloaded=$((downloaded + 1))
            fi
        done < "$selection_csv"
        log "  Already downloaded: $downloaded / $(( $(wc -l < "$selection_csv") - 1 ))"
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        log "  [DRY-RUN] python src/scripts/download_hc_100.py --n-subjects 100 --seed 42"
        return 0
    fi

    log "  Starting download (skips existing)..."
    python "$SCRIPT_DIR/download_hc_100.py" --n-subjects 100 --seed 42 2>&1 | tee -a "$LOG_FILE"
}

step3_fmriprep() {
    log "━━━ Step 3: fMRIPrep Preprocessing ━━━"

    local selection_csv="$DATA_DIR/hc_100_selection.csv"
    if [ ! -f "$selection_csv" ]; then
        log "  ERROR: $selection_csv not found. Run Step 2 first."
        return 1
    fi

    # Count already processed
    local processed=0
    local total=0
    while IFS=, read -r ds sub _; do
        [[ "$ds" == "dataset_id" ]] && continue
        total=$((total + 1))
        if [ -f "$DATA_DIR/derivatives/fmriprep/${sub}.html" ]; then
            processed=$((processed + 1))
        fi
    done < "$selection_csv"
    log "  fMRIPrep status: $processed / $total processed"

    if [ "$processed" -eq "$total" ]; then
        log "  All subjects processed, skipping"
        return 0
    fi

    local fmriprep_cmd="bash $SCRIPT_DIR/run_fmriprep_batch.sh --csv $selection_csv"
    [ -n "$SINGULARITY_FLAG" ] && fmriprep_cmd="$fmriprep_cmd $SINGULARITY_FLAG"
    [ -n "$MAX_PARALLEL_FLAG" ] && fmriprep_cmd="$fmriprep_cmd $MAX_PARALLEL_FLAG"

    if [ "$DRY_RUN" -eq 1 ]; then
        log "  [DRY-RUN] $fmriprep_cmd"
        return 0
    fi

    log "  Running fMRIPrep batch..."
    log "  Command: $fmriprep_cmd"
    log "  ⚠ This may take hours. Consider running in tmux/screen."
    log "  ⚠ Estimated time: ~2h/subject (serial), check logs in $DERIV_DIR/"
    eval "$fmriprep_cmd" 2>&1 | tee -a "$LOG_FILE"
}

step4_bsnet() {
    log "━━━ Step 4: BS-NET Pipeline ━━━"

    local fmriprep_dir="$DATA_DIR/derivatives/fmriprep"
    local bsnet_dir="$DATA_DIR/derivatives/bsnet"
    local selection_csv="$DATA_DIR/hc_100_selection.csv"

    # Count available fMRIPrep outputs
    local available=0
    if [ -d "$fmriprep_dir" ]; then
        available=$(find "$fmriprep_dir" -name "*MNI*bold*" -type f 2>/dev/null | wc -l)
    fi
    log "  fMRIPrep outputs available: $available subjects"

    if [ "$available" -eq 0 ]; then
        log "  ERROR: No fMRIPrep outputs found. Run Step 3 first."
        return 1
    fi

    # Count already BS-NET processed
    local bsnet_done=0
    if [ -d "$bsnet_dir" ]; then
        bsnet_done=$(find "$bsnet_dir" -name "*_bsnet_results.json" -type f 2>/dev/null | wc -l)
    fi
    log "  BS-NET processed: $bsnet_done subjects"

    local bsnet_cmd="python $SCRIPT_DIR/run_fmriprep_bsnet.py --run-all --verbose"
    if [ -f "$selection_csv" ]; then
        bsnet_cmd="python $SCRIPT_DIR/run_fmriprep_bsnet.py --run-selection $selection_csv --verbose"
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        log "  [DRY-RUN] $bsnet_cmd"
        return 0
    fi

    log "  Running BS-NET..."
    eval "$bsnet_cmd" 2>&1 | tee -a "$LOG_FILE"
}

step5_summarize() {
    log "━━━ Step 5: Results Summary ━━━"

    local summary_csv="$DATA_DIR/derivatives/bsnet/bsnet_results_summary.csv"
    if [ ! -f "$summary_csv" ]; then
        log "  ERROR: $summary_csv not found. Run Step 4 first."
        return 1
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        log "  [DRY-RUN] python -c '...summarize...'"
        return 0
    fi

    python - <<'PYEOF'
import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("data/derivatives/bsnet/bsnet_results_summary.csv")
df = pd.read_csv(csv_path)

success = df[df["status"] == "success"]
n = len(success)

print(f"\n{'='*60}")
print(f" BS-NET Real Data Results (N={n})")
print(f"{'='*60}")

if n == 0:
    print("  No successful subjects.")
else:
    print(f"  r_FC (raw 2min):      {success['r_fc_raw'].mean():.4f} ± {success['r_fc_raw'].std():.4f}")
    print(f"  rho_hat_T (BS-NET):   {success['rho_hat_T'].mean():.4f} ± {success['rho_hat_T'].std():.4f}")
    delta = success['rho_hat_T'] - success['r_fc_raw']
    print(f"  Improvement (Δ):      {delta.mean():+.4f} ± {delta.std():.4f}")
    print(f"  ARI:                  {success['ari'].mean():.4f} ± {success['ari'].std():.4f}")
    print()

    # Per-status breakdown
    print(f"  Status breakdown:")
    for status, count in df["status"].value_counts().items():
        print(f"    {status}: {count}")

    # Save enhanced summary
    out = csv_path.parent / "bsnet_results_enhanced.csv"
    success_copy = success.copy()
    success_copy["delta_rho"] = delta.values
    success_copy.to_csv(out, index=False)
    print(f"\n  Enhanced results: {out}")

print(f"{'='*60}")
PYEOF
}

# ---- Execute ----
log "BS-NET Real-Data Pipeline"
log "Project: $PROJECT_ROOT"
log "Log: $LOG_FILE"
log ""

if [ -n "$STEP" ]; then
    # Single step
    case "$STEP" in
        0) step0_setup ;;
        1) step1_index ;;
        2) step2_download ;;
        3) step3_fmriprep ;;
        4) step4_bsnet ;;
        5) step5_summarize ;;
        *) log "ERROR: Unknown step $STEP (valid: 0-5)"; exit 1 ;;
    esac
else
    # Full pipeline
    step0_setup
    step1_index
    step2_download
    step3_fmriprep
    step4_bsnet
    step5_summarize
fi

log ""
log "Pipeline finished. Log: $LOG_FILE"
