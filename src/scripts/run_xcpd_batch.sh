#!/usr/bin/env bash
# ============================================================================
# BS-NET: Batch XCP-D Post-Processing
# ============================================================================
# fMRIPrep 출력에 XCP-D를 적용하여 denoised BOLD + FC를 생성.
#
# XCP-D 수행 내용:
#   - 36-parameter confound regression (24 motion + 8 WM/CSF + 4 global signal)
#   - Bandpass filtering (0.01–0.1 Hz)
#   - Motion scrubbing (FD > 0.5 mm)
#   - Spatial smoothing (6 mm FWHM)
#   - Parcellated time series (Schaefer 100 / 400)
#   - FC matrix 산출
#
# 사전 요구사항:
#   - Docker (또는 Singularity)
#   - fMRIPrep 결과: data/derivatives/fmriprep/
#
# 사용법:
#   # 단일 피험자
#   ./src/scripts/run_xcpd_batch.sh --subject sub-10159
#
#   # CSV 기반 일괄 처리
#   ./src/scripts/run_xcpd_batch.sh --csv data/hc_100_selection.csv
#
#   # 전체 (fMRIPrep 완료된 피험자만)
#   ./src/scripts/run_xcpd_batch.sh --all
#
#   # Singularity (HPC)
#   ./src/scripts/run_xcpd_batch.sh --csv data/hc_100_selection.csv --singularity
#
#   # Dry-run
#   ./src/scripts/run_xcpd_batch.sh --csv data/hc_100_selection.csv --dry-run
# ============================================================================
set -euo pipefail

# ---- Configuration ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
FMRIPREP_DIR="$DATA_DIR/derivatives/fmriprep"
XCPD_OUT_DIR="$DATA_DIR/derivatives/xcp_d"

# XCP-D settings
XCPD_VERSION="0.10.0"
N_CPUS="${BSNET_NCPUS:-4}"
MEM_GB="${BSNET_MEM_GB:-16}"
FD_THRESHOLD="0.5"
SMOOTHING_FWHM="6"
LOW_PASS="0.1"
HIGH_PASS="0.01"
# Denoising strategy: 36P = 24 motion + 8 WM/CSF tissue + 4 GSR derivatives
DENOISING_STRATEGY="36P"

# Singularity image path (HPC)
SINGULARITY_IMG="${XCPD_SIF:-$PROJECT_ROOT/xcp_d-${XCPD_VERSION}.sif}"

# Runtime
DRY_RUN=0
USE_SINGULARITY=0
SUBJECT=""
CSV_FILE=""
RUN_ALL=0
MAX_PARALLEL="${BSNET_MAX_PARALLEL:-1}"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)       SUBJECT="$2"; shift 2 ;;
        --csv)           CSV_FILE="$2"; shift 2 ;;
        --all)           RUN_ALL=1; shift ;;
        --singularity)   USE_SINGULARITY=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --ncpus)         N_CPUS="$2"; shift 2 ;;
        --mem-gb)        MEM_GB="$2"; shift 2 ;;
        --fd-threshold)  FD_THRESHOLD="$2"; shift 2 ;;
        --smoothing)     SMOOTHING_FWHM="$2"; shift 2 ;;
        --max-parallel)  MAX_PARALLEL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--subject SUB] [--csv CSV] [--all]"
            echo "           [--singularity] [--dry-run] [--ncpus N] [--mem-gb GB]"
            echo ""
            echo "Options:"
            echo "  --subject SUB      Process single subject (e.g., sub-10159)"
            echo "  --csv CSV          CSV with dataset_id,participant_id columns"
            echo "  --all              Process all fMRIPrep-completed subjects"
            echo "  --singularity      Use Singularity instead of Docker"
            echo "  --dry-run          Print commands without executing"
            echo "  --ncpus N          CPUs per subject (default: 4, env: BSNET_NCPUS)"
            echo "  --mem-gb GB        Memory limit in GB (default: 16, env: BSNET_MEM_GB)"
            echo "  --fd-threshold F   FD threshold for scrubbing (default: 0.5)"
            echo "  --smoothing F      Smoothing FWHM in mm (default: 6)"
            echo "  --max-parallel N   Parallel jobs (default: 1, env: BSNET_MAX_PARALLEL)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Validation ----
echo "============================================"
echo " BS-NET: Batch XCP-D Post-Processing"
echo " XCP-D version: $XCPD_VERSION"
echo " Denoising: $DENOISING_STRATEGY"
echo " FD threshold: $FD_THRESHOLD mm"
echo " Smoothing: $SMOOTHING_FWHM mm FWHM"
echo " Bandpass: $HIGH_PASS–$LOW_PASS Hz"
echo " CPUs/subject: $N_CPUS, Memory: ${MEM_GB}GB"
echo " Output: $XCPD_OUT_DIR"
if [ "$DRY_RUN" -eq 1 ]; then echo " ** DRY RUN **"; fi
echo "============================================"

# fMRIPrep output check
if [ ! -d "$FMRIPREP_DIR" ]; then
    echo "ERROR: fMRIPrep directory not found: $FMRIPREP_DIR"
    echo "  Run fMRIPrep first: ./src/scripts/run_fmriprep_batch.sh"
    exit 1
fi

# Docker/Singularity check (skip in dry-run)
if [ "$DRY_RUN" -eq 0 ]; then
    if [ "$USE_SINGULARITY" -eq 1 ]; then
        if ! command -v singularity &>/dev/null; then
            echo "ERROR: singularity not found in PATH"
            exit 1
        fi
        if [ ! -f "$SINGULARITY_IMG" ]; then
            echo "ERROR: Singularity image not found: $SINGULARITY_IMG"
            echo "Build: singularity build xcp_d-${XCPD_VERSION}.sif docker://pennlinc/xcp_d:${XCPD_VERSION}"
            exit 1
        fi
    else
        if ! command -v docker &>/dev/null; then
            echo "ERROR: docker not found in PATH"
            exit 1
        fi
    fi
fi

# ---- Build subject list ----
declare -a SUBJECTS=()

if [ -n "$SUBJECT" ]; then
    SUBJECTS+=("$SUBJECT")
elif [ -n "$CSV_FILE" ]; then
    if [ ! -f "$CSV_FILE" ]; then
        echo "ERROR: CSV not found: $CSV_FILE"; exit 1
    fi
    while IFS=, read -r _ds sub _ ; do
        [[ "$_ds" == "dataset_id" ]] && continue
        [[ -z "$sub" ]] && continue
        [[ "$sub" == sub-* ]] || sub="sub-$sub"
        SUBJECTS+=("$sub")
    done < "$CSV_FILE"
elif [ "$RUN_ALL" -eq 1 ]; then
    for sub_dir in "$FMRIPREP_DIR"/sub-*; do
        [ -d "$sub_dir" ] || continue
        sub_id=$(basename "$sub_dir")
        # Only include if fMRIPrep func outputs exist
        if ls "$sub_dir"/func/*MNI*bold* &>/dev/null 2>&1; then
            SUBJECTS+=("$sub_id")
        fi
    done
else
    echo "ERROR: Specify --subject, --csv, or --all"
    exit 1
fi

echo ""
echo "Subjects to process: ${#SUBJECTS[@]}"

# ---- Helper: check if already processed ----
is_xcpd_processed() {
    local sub_id="$1"
    # XCP-D creates atlas-based time series and FC matrices
    local xcpd_func="$XCPD_OUT_DIR/$sub_id/func"
    if [ -d "$xcpd_func" ] && ls "$xcpd_func"/*Schaefer*timeseries* &>/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# ---- Helper: run XCP-D for one subject ----
run_xcpd_one() {
    local sub_id="$1"
    local work_dir="$DATA_DIR/xcpd_work/${sub_id}"

    # Validate fMRIPrep output
    if ! ls "$FMRIPREP_DIR/$sub_id"/func/*MNI*bold* &>/dev/null 2>&1; then
        echo "  SKIP: $sub_id — no fMRIPrep BOLD output"
        return 1
    fi

    mkdir -p "$work_dir" "$XCPD_OUT_DIR"

    if [ "$USE_SINGULARITY" -eq 1 ]; then
        local cmd=(
            singularity run --cleanenv
            -B "$FMRIPREP_DIR":/data:ro
            -B "$XCPD_OUT_DIR":/out
            -B "$work_dir":/work
            "$SINGULARITY_IMG"
            /data /out participant
            --participant-label "${sub_id#sub-}"
            --nprocs "$N_CPUS"
            --mem-gb "$MEM_GB"
            --work-dir /work
            --despike
            --fd-thresh "$FD_THRESHOLD"
            --smoothing "$SMOOTHING_FWHM"
            --low-pass "$LOW_PASS"
            --high-pass "$HIGH_PASS"
            --atlases Schaefer100 Schaefer400
            --min-time 100
            --notrack
        )
    else
        local cmd=(
            docker run --rm
            -v "$FMRIPREP_DIR":/data:ro
            -v "$XCPD_OUT_DIR":/out
            -v "$work_dir":/work
            pennlinc/xcp_d:"$XCPD_VERSION"
            /data /out participant
            --participant-label "${sub_id#sub-}"
            --nprocs "$N_CPUS"
            --mem-gb "$MEM_GB"
            --work-dir /work
            --despike
            --fd-thresh "$FD_THRESHOLD"
            --smoothing "$SMOOTHING_FWHM"
            --low-pass "$LOW_PASS"
            --high-pass "$HIGH_PASS"
            --atlases Schaefer100 Schaefer400
            --min-time 100
            --notrack
        )
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY-RUN] ${cmd[*]}"
        return 0
    fi

    echo "  Running XCP-D for $sub_id..."
    local log_file="$XCPD_OUT_DIR/${sub_id}_xcpd.log"

    if "${cmd[@]}" > "$log_file" 2>&1; then
        echo "  OK: $sub_id → $XCPD_OUT_DIR/$sub_id/"
        rm -rf "$work_dir"
        return 0
    else
        echo "  FAIL: $sub_id (see $log_file)"
        return 1
    fi
}

# ---- Main loop ----
echo ""
TOTAL=${#SUBJECTS[@]}
SUCCESS=0
FAIL=0
SKIP=0

RUNNING_PIDS=()

for i in "${!SUBJECTS[@]}"; do
    sub_id="${SUBJECTS[$i]}"
    idx=$((i + 1))

    echo "[${idx}/${TOTAL}] ${sub_id}"

    if is_xcpd_processed "$sub_id"; then
        echo "  SKIP: already processed"
        SKIP=$((SKIP + 1))
        continue
    fi

    if [ "$MAX_PARALLEL" -le 1 ]; then
        if run_xcpd_one "$sub_id"; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAIL=$((FAIL + 1))
        fi
    else
        while [ ${#RUNNING_PIDS[@]} -ge "$MAX_PARALLEL" ]; do
            for pid_idx in "${!RUNNING_PIDS[@]}"; do
                if ! kill -0 "${RUNNING_PIDS[$pid_idx]}" 2>/dev/null; then
                    wait "${RUNNING_PIDS[$pid_idx]}" && SUCCESS=$((SUCCESS + 1)) || FAIL=$((FAIL + 1))
                    unset 'RUNNING_PIDS[$pid_idx]'
                    RUNNING_PIDS=("${RUNNING_PIDS[@]}")
                    break
                fi
            done
            sleep 2
        done

        run_xcpd_one "$sub_id" &
        RUNNING_PIDS+=($!)
    fi
done

for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid" && SUCCESS=$((SUCCESS + 1)) || FAIL=$((FAIL + 1))
done

# ---- Summary ----
echo ""
echo "============================================"
echo " XCP-D Batch Complete"
echo "  Total:    $TOTAL"
echo "  Success:  $SUCCESS"
echo "  Failed:   $FAIL"
echo "  Skipped:  $SKIP (already processed)"
echo ""
echo " Outputs: $XCPD_OUT_DIR"
echo "   Per-subject: <sub>/func/*atlas-Schaefer*_timeseries.tsv"
echo "   Per-subject: <sub>/func/*atlas-Schaefer*_relmat.tsv"
echo "============================================"

if [ "$SUCCESS" -gt 0 ] || [ "$SKIP" -gt 0 ]; then
    echo ""
    echo "Next step: Run BS-NET on XCP-D outputs"
    echo "  python src/scripts/run_fmriprep_bsnet.py --run-all --verbose"
fi
