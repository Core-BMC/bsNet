#!/usr/bin/env bash
# ============================================================================
# BS-NET: Per-Dataset Pipeline Runner
# ============================================================================
# 특정 ds* 데이터셋 단위로 Download → fMRIPrep → BS-NET 을 실행.
# 전체 100명 다운로드를 기다리지 않고, 준비된 데이터셋부터 점진적 처리.
#
# 사용 시나리오:
#   A. 빠른 검증 (proof-of-concept):
#      특정 데이터셋 하나로 파이프라인 전체를 빠르게 검증
#   B. 점진적 처리:
#      다운로드 완료된 데이터셋부터 순차적으로 fMRIPrep + BS-NET 실행
#
# 사용법:
#   # 단일 데이터셋 전체 파이프라인
#   ./src/scripts/run_dataset_pipeline.sh --dataset ds000030
#
#   # 여러 데이터셋 지정 (순차 처리)
#   ./src/scripts/run_dataset_pipeline.sh --dataset ds000030,ds000243
#
#   # 다운로드된 모든 데이터셋에 대해 자동 실행
#   ./src/scripts/run_dataset_pipeline.sh --auto
#
#   # N명만 (proof-of-concept)
#   ./src/scripts/run_dataset_pipeline.sh --dataset ds000030 --max-subjects 5
#
#   # fMRIPrep만 (다운로드 이미 완료된 경우)
#   ./src/scripts/run_dataset_pipeline.sh --dataset ds000030 --skip-download
#
#   # BS-NET만 (fMRIPrep 이미 완료된 경우)
#   ./src/scripts/run_dataset_pipeline.sh --dataset ds000030 --only-bsnet
#
#   # Dry-run
#   ./src/scripts/run_dataset_pipeline.sh --dataset ds000030 --dry-run
#
#   # Singularity (HPC)
#   ./src/scripts/run_dataset_pipeline.sh --dataset ds000030 --singularity
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
OPENNEURO_DIR="$DATA_DIR/openneuro"
DERIV_DIR="$DATA_DIR/derivatives"

# ---- Parse arguments ----
DATASETS=""
AUTO_MODE=0
DRY_RUN=0
SKIP_DOWNLOAD=0
ONLY_BSNET=0
MAX_SUBJECTS=0  # 0 = all
SINGULARITY_FLAG=""
NCPUS_FLAG=""
MEM_FLAG=""
MAX_PARALLEL_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASETS="$2"; shift 2 ;;
        --auto)          AUTO_MODE=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --only-bsnet)    ONLY_BSNET=1; shift ;;
        --max-subjects)  MAX_SUBJECTS="$2"; shift 2 ;;
        --singularity)   SINGULARITY_FLAG="--singularity"; shift ;;
        --ncpus)         NCPUS_FLAG="--ncpus $2"; shift 2 ;;
        --mem-mb)        MEM_FLAG="--mem-mb $2"; shift 2 ;;
        --max-parallel)  MAX_PARALLEL_FLAG="--max-parallel $2"; shift 2 ;;
        -h|--help)
            head -40 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Logging ----
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/dataset_pipeline_${TIMESTAMP}.log"

log() {
    local msg="[$(date +%H:%M:%S)] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

# ---- Resolve dataset list ----
declare -a DS_LIST=()

if [ "$AUTO_MODE" -eq 1 ]; then
    # Auto-detect: find all ds* dirs with at least one sub-* containing anat+func
    if [ -d "$OPENNEURO_DIR" ]; then
        for ds_dir in "$OPENNEURO_DIR"/ds*; do
            [ -d "$ds_dir" ] || continue
            ds_id=$(basename "$ds_dir")
            # Check at least one subject has both anat and func
            for sub_dir in "$ds_dir"/sub-*; do
                [ -d "$sub_dir" ] || continue
                if [ -d "$sub_dir/anat" ] && [ -d "$sub_dir/func" ]; then
                    DS_LIST+=("$ds_id")
                    break
                fi
            done
        done
    fi
    if [ ${#DS_LIST[@]} -eq 0 ]; then
        log "ERROR: No downloaded datasets found in $OPENNEURO_DIR"
        log "  Run download first: python src/scripts/download_hc_100.py"
        exit 1
    fi
elif [ -n "$DATASETS" ]; then
    IFS=',' read -ra DS_LIST <<< "$DATASETS"
else
    echo "ERROR: Specify --dataset ds000030[,ds000243,...] or --auto"
    echo "  Run with -h for help."
    exit 1
fi

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log " BS-NET Per-Dataset Pipeline"
log " Datasets: ${DS_LIST[*]}"
log " Max subjects/dataset: ${MAX_SUBJECTS:-all}"
log " Skip download: $SKIP_DOWNLOAD"
log " Only BS-NET: $ONLY_BSNET"
if [ "$DRY_RUN" -eq 1 ]; then log " ** DRY RUN **"; fi
log " Log: $LOG_FILE"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ---- Helper: list subjects in a dataset ----
list_subjects_downloaded() {
    local ds_id="$1"
    local ds_dir="$OPENNEURO_DIR/$ds_id"
    local subjects=()

    if [ ! -d "$ds_dir" ]; then
        echo ""
        return
    fi

    for sub_dir in "$ds_dir"/sub-*; do
        [ -d "$sub_dir" ] || continue
        local sub_id
        sub_id=$(basename "$sub_dir")
        # Verify anat + func exist
        if [ -d "$sub_dir/anat" ] && [ -d "$sub_dir/func" ]; then
            subjects+=("$sub_id")
        fi
    done

    echo "${subjects[*]}"
}

# ---- Helper: list subjects from hc_adult_index.csv for a given dataset ----
list_subjects_indexed() {
    local ds_id="$1"
    local index_csv="$DATA_DIR/hc_adult_index.csv"

    if [ ! -f "$index_csv" ]; then
        echo ""
        return
    fi

    local subjects=()
    while IFS=, read -r ds sub _rest; do
        [[ "$ds" == "dataset_id" ]] && continue
        if [[ "$ds" == "$ds_id" ]]; then
            # Ensure sub- prefix
            [[ "$sub" == sub-* ]] || sub="sub-$sub"
            subjects+=("$sub")
        fi
    done < "$index_csv"

    echo "${subjects[*]}"
}

# ---- Per-dataset processing ----
GRAND_TOTAL=0
GRAND_SUCCESS=0
GRAND_FAIL=0
GRAND_SKIP=0

for ds_id in "${DS_LIST[@]}"; do
    log ""
    log "╔══════════════════════════════════════════════╗"
    log "║  Dataset: $ds_id"
    log "╚══════════════════════════════════════════════╝"

    # === Phase 1: Download (if not skipped) ===
    if [ "$SKIP_DOWNLOAD" -eq 0 ] && [ "$ONLY_BSNET" -eq 0 ]; then
        log ""
        log "── Phase 1: Download $ds_id ──"

        # Check if index exists
        INDEX_CSV="$DATA_DIR/hc_adult_index.csv"
        if [ ! -f "$INDEX_CSV" ]; then
            log "  Index CSV not found. Running indexer first..."
            if [ "$DRY_RUN" -eq 1 ]; then
                log "  [DRY-RUN] python src/scripts/index_openneuro_hc.py"
            else
                python "$SCRIPT_DIR/index_openneuro_hc.py" 2>&1 | tee -a "$LOG_FILE"
            fi
        fi

        # Get subjects for this dataset from index
        read -ra INDEXED_SUBS <<< "$(list_subjects_indexed "$ds_id")"
        if [ ${#INDEXED_SUBS[@]} -eq 0 ]; then
            log "  WARNING: No HC subjects found for $ds_id in index. Skipping download."
        else
            log "  HC subjects in index: ${#INDEXED_SUBS[@]}"

            # Apply max-subjects cap
            if [ "$MAX_SUBJECTS" -gt 0 ] && [ ${#INDEXED_SUBS[@]} -gt "$MAX_SUBJECTS" ]; then
                INDEXED_SUBS=("${INDEXED_SUBS[@]:0:$MAX_SUBJECTS}")
                log "  Capped to $MAX_SUBJECTS subjects"
            fi

            # Download each subject
            for sub_id in "${INDEXED_SUBS[@]}"; do
                local_dir="$OPENNEURO_DIR/$ds_id/$sub_id"
                if [ -d "$local_dir/anat" ] && [ -d "$local_dir/func" ]; then
                    log "  SKIP (exists): $sub_id"
                    continue
                fi

                dl_cmd="python $SCRIPT_DIR/download_hc_100.py --index-csv $INDEX_CSV --output-dir $OPENNEURO_DIR --n-subjects 1 --dry-run"
                # Use openneuro-py directly for single subject
                dl_cmd="openneuro-py download --dataset $ds_id --include $sub_id --target-dir $OPENNEURO_DIR/$ds_id"

                if [ "$DRY_RUN" -eq 1 ]; then
                    log "  [DRY-RUN] $dl_cmd"
                else
                    log "  Downloading $sub_id..."
                    if eval "$dl_cmd" >> "$LOG_FILE" 2>&1; then
                        log "  OK: $sub_id"
                    else
                        log "  FAIL: $sub_id download"
                    fi
                fi
            done
        fi
    fi

    # === Phase 2: fMRIPrep ===
    if [ "$ONLY_BSNET" -eq 0 ]; then
        log ""
        log "── Phase 2: fMRIPrep $ds_id ──"

        # List actually downloaded subjects
        read -ra DL_SUBS <<< "$(list_subjects_downloaded "$ds_id")"

        if [ ${#DL_SUBS[@]} -eq 0 ]; then
            log "  No downloaded subjects for $ds_id. Skipping fMRIPrep."
        else
            # Apply max-subjects cap
            if [ "$MAX_SUBJECTS" -gt 0 ] && [ ${#DL_SUBS[@]} -gt "$MAX_SUBJECTS" ]; then
                DL_SUBS=("${DL_SUBS[@]:0:$MAX_SUBJECTS}")
            fi

            log "  Subjects to process: ${#DL_SUBS[@]}"

            for sub_id in "${DL_SUBS[@]}"; do
                GRAND_TOTAL=$((GRAND_TOTAL + 1))

                # Check if already processed
                if [ -f "$DERIV_DIR/fmriprep/${sub_id}.html" ]; then
                    log "  SKIP (processed): $sub_id"
                    GRAND_SKIP=$((GRAND_SKIP + 1))
                    continue
                fi

                fmriprep_cmd="bash $SCRIPT_DIR/run_fmriprep_batch.sh --subject $sub_id --dataset $ds_id"
                [ -n "$SINGULARITY_FLAG" ] && fmriprep_cmd="$fmriprep_cmd $SINGULARITY_FLAG"
                [ -n "$NCPUS_FLAG" ] && fmriprep_cmd="$fmriprep_cmd $NCPUS_FLAG"
                [ -n "$MEM_FLAG" ] && fmriprep_cmd="$fmriprep_cmd $MEM_FLAG"

                if [ "$DRY_RUN" -eq 1 ]; then
                    log "  [DRY-RUN] $fmriprep_cmd"
                    GRAND_SKIP=$((GRAND_SKIP + 1))
                else
                    log "  fMRIPrep: $sub_id ($ds_id)..."
                    if eval "$fmriprep_cmd" >> "$LOG_FILE" 2>&1; then
                        log "  OK: $sub_id"
                        GRAND_SUCCESS=$((GRAND_SUCCESS + 1))
                    else
                        log "  FAIL: $sub_id fMRIPrep"
                        GRAND_FAIL=$((GRAND_FAIL + 1))
                    fi
                fi
            done
        fi
    fi

    # === Phase 2.5: XCP-D Post-Processing ===
    if [ "$ONLY_BSNET" -eq 0 ]; then
        log ""
        log "── Phase 2.5: XCP-D $ds_id ──"

        read -ra DL_SUBS <<< "$(list_subjects_downloaded "$ds_id")"
        if [ "$MAX_SUBJECTS" -gt 0 ] && [ ${#DL_SUBS[@]} -gt "$MAX_SUBJECTS" ]; then
            DL_SUBS=("${DL_SUBS[@]:0:$MAX_SUBJECTS}")
        fi

        XCPD_TODO=0
        for sub_id in "${DL_SUBS[@]}"; do
            # Only run XCP-D if fMRIPrep done but XCP-D not yet
            if ls "$DERIV_DIR/fmriprep/$sub_id"/func/*MNI*bold* &>/dev/null 2>&1; then
                if ! ls "$DERIV_DIR/xcp_d/$sub_id"/func/*Schaefer*timeseries* &>/dev/null 2>&1; then
                    XCPD_TODO=$((XCPD_TODO + 1))
                fi
            fi
        done

        if [ "$XCPD_TODO" -eq 0 ]; then
            log "  All fMRIPrep subjects already have XCP-D outputs (or none ready)."
        else
            log "  Subjects needing XCP-D: $XCPD_TODO"
            for sub_id in "${DL_SUBS[@]}"; do
                if ! ls "$DERIV_DIR/fmriprep/$sub_id"/func/*MNI*bold* &>/dev/null 2>&1; then
                    continue
                fi
                if ls "$DERIV_DIR/xcp_d/$sub_id"/func/*Schaefer*timeseries* &>/dev/null 2>&1; then
                    log "  SKIP (XCP-D done): $sub_id"
                    continue
                fi

                xcpd_cmd="bash $SCRIPT_DIR/run_xcpd_batch.sh --subject $sub_id"
                [ -n "$SINGULARITY_FLAG" ] && xcpd_cmd="$xcpd_cmd $SINGULARITY_FLAG"
                [ -n "$NCPUS_FLAG" ] && xcpd_cmd="$xcpd_cmd $NCPUS_FLAG"

                if [ "$DRY_RUN" -eq 1 ]; then
                    log "  [DRY-RUN] $xcpd_cmd"
                else
                    log "  XCP-D: $sub_id..."
                    if eval "$xcpd_cmd" >> "$LOG_FILE" 2>&1; then
                        log "  OK: $sub_id"
                    else
                        log "  FAIL: $sub_id XCP-D"
                    fi
                fi
            done
        fi
    fi

    # === Phase 3: BS-NET ===
    log ""
    log "── Phase 3: BS-NET $ds_id ──"

    # Find subjects with XCP-D outputs (preferred) or fMRIPrep outputs (fallback)
    read -ra DL_SUBS <<< "$(list_subjects_downloaded "$ds_id")"
    if [ "$MAX_SUBJECTS" -gt 0 ] && [ ${#DL_SUBS[@]} -gt "$MAX_SUBJECTS" ]; then
        DL_SUBS=("${DL_SUBS[@]:0:$MAX_SUBJECTS}")
    fi

    BSNET_READY=0
    INPUT_MODE="xcpd"
    for sub_id in "${DL_SUBS[@]}"; do
        if ls "$DERIV_DIR/xcp_d/$sub_id"/func/*Schaefer*timeseries* &>/dev/null 2>&1; then
            BSNET_READY=$((BSNET_READY + 1))
        elif ls "$DERIV_DIR/fmriprep/$sub_id"/func/*MNI*bold* &>/dev/null 2>&1; then
            BSNET_READY=$((BSNET_READY + 1))
            INPUT_MODE="fmriprep"
        fi
    done

    if [ "$BSNET_READY" -eq 0 ]; then
        log "  No preprocessed outputs ready for $ds_id. Skipping BS-NET."
        continue
    fi

    log "  Outputs ready: $BSNET_READY subjects (input: $INPUT_MODE)"

    for sub_id in "${DL_SUBS[@]}"; do
        # Determine input mode per subject
        sub_input="xcpd"
        if ls "$DERIV_DIR/xcp_d/$sub_id"/func/*Schaefer*timeseries* &>/dev/null 2>&1; then
            sub_input="xcpd"
        elif ls "$DERIV_DIR/fmriprep/$sub_id"/func/*MNI*bold* &>/dev/null 2>&1; then
            sub_input="fmriprep"
        else
            continue
        fi

        # Check if already BS-NET processed
        if [ -f "$DERIV_DIR/bsnet/$sub_id/${sub_id}_bsnet_results.json" ]; then
            log "  SKIP (BS-NET done): $sub_id"
            continue
        fi

        bsnet_cmd="python $SCRIPT_DIR/run_fmriprep_bsnet.py --subject $sub_id --input-mode $sub_input --verbose"

        if [ "$DRY_RUN" -eq 1 ]; then
            log "  [DRY-RUN] $bsnet_cmd"
        else
            log "  BS-NET ($sub_input): $sub_id..."
            if eval "$bsnet_cmd" >> "$LOG_FILE" 2>&1; then
                log "  OK: $sub_id → rho_hat_T computed"
            else
                log "  FAIL: $sub_id BS-NET"
            fi
        fi
    done

    log ""
    log "  ✓ $ds_id pipeline complete"
done

# ---- Grand Summary ----
log ""
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log " Per-Dataset Pipeline Complete"
log "  Datasets processed: ${#DS_LIST[@]} (${DS_LIST[*]})"
log "  fMRIPrep — Total: $GRAND_TOTAL, Success: $GRAND_SUCCESS, Failed: $GRAND_FAIL, Skipped: $GRAND_SKIP"
log ""
log " Outputs:"
log "   fMRIPrep: $DERIV_DIR/fmriprep/"
log "   XCP-D:    $DERIV_DIR/xcp_d/"
log "   BS-NET:   $DERIV_DIR/bsnet/"
log "   Log:      $LOG_FILE"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Hint for next steps
if [ "$DRY_RUN" -eq 0 ]; then
    log ""
    log "Next steps:"
    log "  # Results summary"
    log "  ./src/scripts/run_all_pipeline.sh --step 5"
    log ""
    log "  # Process next dataset"
    log "  ./src/scripts/run_dataset_pipeline.sh --dataset <next_ds_id>"
fi
