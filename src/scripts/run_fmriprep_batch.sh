#!/usr/bin/env bash
# ============================================================================
# BS-NET: Batch fMRIPrep Preprocessing
# ============================================================================
# 워크스테이션/서버에서 OpenNeuro HC subjects 를 fMRIPrep으로 일괄 전처리.
#
# 사전 요구사항:
#   - Docker (또는 Singularity)
#   - FreeSurfer license (fs_license.txt)
#   - 데이터: data/openneuro/<dataset_id>/sub-XXXX/{anat,func}
#
# 사용법:
#   chmod +x src/scripts/run_fmriprep_batch.sh
#
#   # 단일 피험자
#   ./src/scripts/run_fmriprep_batch.sh --subject sub-10159 --dataset ds000030
#
#   # CSV 기반 일괄 처리
#   ./src/scripts/run_fmriprep_batch.sh --csv data/hc_100_selection.csv
#
#   # 전체 (fmriprep 미처리 피험자만)
#   ./src/scripts/run_fmriprep_batch.sh --all
#
#   # Singularity 사용 (HPC 환경)
#   ./src/scripts/run_fmriprep_batch.sh --csv data/hc_100_selection.csv --singularity
#
#   # Dry-run (실제 실행 없이 명령어만 출력)
#   ./src/scripts/run_fmriprep_batch.sh --csv data/hc_100_selection.csv --dry-run
# ============================================================================
set -euo pipefail

# ---- Configuration ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
OPENNEURO_DIR="$DATA_DIR/openneuro"
DERIV_DIR="$DATA_DIR/derivatives/fmriprep"

# fMRIPrep settings
FMRIPREP_VERSION="24.1.1"
N_CPUS="${BSNET_NCPUS:-4}"
MEM_MB="${BSNET_MEM_MB:-16000}"
OUTPUT_SPACES="MNI152NLin6Asym:res-2"
FS_LICENSE="${FS_LICENSE:-$PROJECT_ROOT/fs_license.txt}"

# Singularity image path (HPC)
SINGULARITY_IMG="${FMRIPREP_SIF:-$PROJECT_ROOT/fmriprep-${FMRIPREP_VERSION}.sif}"

# Runtime
DRY_RUN=0
USE_SINGULARITY=0
SUBJECT=""
DATASET=""
CSV_FILE=""
RUN_ALL=0
MAX_PARALLEL="${BSNET_MAX_PARALLEL:-1}"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)     SUBJECT="$2"; shift 2 ;;
        --dataset)     DATASET="$2"; shift 2 ;;
        --csv)         CSV_FILE="$2"; shift 2 ;;
        --all)         RUN_ALL=1; shift ;;
        --singularity) USE_SINGULARITY=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --ncpus)       N_CPUS="$2"; shift 2 ;;
        --mem-mb)      MEM_MB="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --fs-license)  FS_LICENSE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--subject SUB --dataset DS] [--csv CSV] [--all]"
            echo "           [--singularity] [--dry-run] [--ncpus N] [--mem-mb MB]"
            echo ""
            echo "Options:"
            echo "  --subject SUB    Process single subject (requires --dataset)"
            echo "  --dataset DS     Dataset ID (e.g., ds000030)"
            echo "  --csv CSV        CSV with dataset_id,participant_id columns"
            echo "  --all            Process all downloaded subjects"
            echo "  --singularity    Use Singularity instead of Docker"
            echo "  --dry-run        Print commands without executing"
            echo "  --ncpus N        CPUs per subject (default: 4, env: BSNET_NCPUS)"
            echo "  --mem-mb MB      Memory limit in MB (default: 16000, env: BSNET_MEM_MB)"
            echo "  --max-parallel N Parallel fMRIPrep jobs (default: 1, env: BSNET_MAX_PARALLEL)"
            echo "  --fs-license F   FreeSurfer license path (env: FS_LICENSE)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Validation ----
echo "============================================"
echo " BS-NET: Batch fMRIPrep"
echo " fMRIPrep version: $FMRIPREP_VERSION"
echo " CPUs/subject: $N_CPUS, Memory: ${MEM_MB}MB"
echo " Max parallel: $MAX_PARALLEL"
echo " Output: $DERIV_DIR"
if [ "$DRY_RUN" -eq 1 ]; then echo " ** DRY RUN **"; fi
echo "============================================"

# FreeSurfer license check (skip in dry-run)
if [ "$DRY_RUN" -eq 0 ] && [ ! -f "$FS_LICENSE" ]; then
    echo ""
    echo "ERROR: FreeSurfer license not found at: $FS_LICENSE"
    echo ""
    echo "Solutions:"
    echo "  1. Place fs_license.txt in project root: $PROJECT_ROOT/"
    echo "  2. Set FS_LICENSE env var: export FS_LICENSE=/path/to/license.txt"
    echo "  3. Pass --fs-license /path/to/license.txt"
    echo ""
    echo "Get license (free): https://surfer.nmr.mgh.harvard.edu/registration.html"
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
            echo "Build: singularity build fmriprep-${FMRIPREP_VERSION}.sif docker://nipreps/fmriprep:${FMRIPREP_VERSION}"
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
declare -a SUBJECTS=()   # "dataset_id:sub_id" pairs
declare -a BIDS_DIRS=()  # corresponding BIDS root dirs

if [ -n "$SUBJECT" ] && [ -n "$DATASET" ]; then
    # Single subject mode
    SUBJECTS+=("${DATASET}:${SUBJECT}")
elif [ -n "$CSV_FILE" ]; then
    # CSV mode: expects dataset_id,participant_id columns
    if [ ! -f "$CSV_FILE" ]; then
        echo "ERROR: CSV not found: $CSV_FILE"; exit 1
    fi
    while IFS=, read -r ds sub _ ; do
        # Skip header
        [[ "$ds" == "dataset_id" ]] && continue
        [[ -z "$ds" || -z "$sub" ]] && continue
        SUBJECTS+=("${ds}:${sub}")
    done < "$CSV_FILE"
elif [ "$RUN_ALL" -eq 1 ]; then
    # All mode: scan openneuro dir
    for ds_dir in "$OPENNEURO_DIR"/ds*; do
        ds_id=$(basename "$ds_dir")
        for sub_dir in "$ds_dir"/sub-*; do
            [ -d "$sub_dir" ] || continue
            sub_id=$(basename "$sub_dir")
            SUBJECTS+=("${ds_id}:${sub_id}")
        done
    done
else
    echo "ERROR: Specify --subject/--dataset, --csv, or --all"
    exit 1
fi

echo ""
echo "Subjects to process: ${#SUBJECTS[@]}"

# ---- Helper: check if already processed ----
is_processed() {
    local sub_id="$1"
    local fmriprep_html="$DERIV_DIR/${sub_id}.html"
    local fmriprep_func="$DERIV_DIR/${sub_id}/func"
    if [ -f "$fmriprep_html" ] && [ -d "$fmriprep_func" ]; then
        return 0
    fi
    return 1
}

# ---- Helper: run fMRIPrep for one subject ----
run_fmriprep_one() {
    local ds_id="$1"
    local sub_id="$2"
    local bids_dir="$OPENNEURO_DIR/$ds_id"
    local work_dir="$DATA_DIR/fmriprep_work/${ds_id}_${sub_id}"

    # Validate BIDS input
    if [ ! -d "$bids_dir/$sub_id" ]; then
        echo "  SKIP: $sub_id not found in $bids_dir"
        return 1
    fi

    # Check for required files
    local has_anat=0 has_func=0
    if ls "$bids_dir/$sub_id"/anat/*T1w*.nii* &>/dev/null 2>&1; then has_anat=1; fi
    if ls "$bids_dir/$sub_id"/func/*bold*.nii* &>/dev/null 2>&1; then has_func=1; fi

    if [ "$has_anat" -eq 0 ] || [ "$has_func" -eq 0 ]; then
        echo "  SKIP: $sub_id missing anat($has_anat) or func($has_func)"
        return 1
    fi

    mkdir -p "$work_dir" "$DERIV_DIR"

    # Ensure dataset_description.json exists (BIDS requirement)
    if [ ! -f "$bids_dir/dataset_description.json" ]; then
        echo '{"Name":"OpenNeuro","BIDSVersion":"1.6.0","DatasetType":"raw"}' \
            > "$bids_dir/dataset_description.json"
    fi

    if [ "$USE_SINGULARITY" -eq 1 ]; then
        # ---- Singularity ----
        local cmd=(
            singularity run --cleanenv
            -B "$bids_dir":/data:ro
            -B "$DERIV_DIR":/out
            -B "$work_dir":/work
            -B "$FS_LICENSE":/opt/freesurfer/license.txt:ro
            "$SINGULARITY_IMG"
            /data /out participant
            --participant-label "${sub_id#sub-}"
            --output-spaces "$OUTPUT_SPACES"
            --nprocs "$N_CPUS"
            --mem-mb "$MEM_MB"
            --work-dir /work
            --skip-bids-validation
            --notrack
            --ignore fieldmaps slicetiming
        )
    else
        # ---- Docker ----
        local cmd=(
            docker run --rm
            -v "$bids_dir":/data:ro
            -v "$DERIV_DIR":/out
            -v "$work_dir":/work
            -v "$FS_LICENSE":/opt/freesurfer/license.txt:ro
            nipreps/fmriprep:"$FMRIPREP_VERSION"
            /data /out participant
            --participant-label "${sub_id#sub-}"
            --output-spaces "$OUTPUT_SPACES"
            --nprocs "$N_CPUS"
            --mem-mb "$MEM_MB"
            --work-dir /work
            --skip-bids-validation
            --notrack
            --ignore fieldmaps slicetiming
        )
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY-RUN] ${cmd[*]}"
        return 0
    fi

    echo "  Running fMRIPrep for $sub_id ($ds_id)..."
    local log_file="$DERIV_DIR/${ds_id}_${sub_id}_fmriprep.log"

    if "${cmd[@]}" > "$log_file" 2>&1; then
        echo "  OK: $sub_id → $DERIV_DIR/$sub_id/"
        # Cleanup work dir on success
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

# Track parallel jobs
RUNNING_PIDS=()

for i in "${!SUBJECTS[@]}"; do
    IFS=: read -r ds_id sub_id <<< "${SUBJECTS[$i]}"
    idx=$((i + 1))

    echo "[${idx}/${TOTAL}] ${ds_id}/${sub_id}"

    # Skip if already processed
    if is_processed "$sub_id"; then
        echo "  SKIP: already processed"
        SKIP=$((SKIP + 1))
        continue
    fi

    if [ "$MAX_PARALLEL" -le 1 ]; then
        # Sequential execution
        if run_fmriprep_one "$ds_id" "$sub_id"; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAIL=$((FAIL + 1))
        fi
    else
        # Parallel execution: wait if at capacity
        while [ ${#RUNNING_PIDS[@]} -ge "$MAX_PARALLEL" ]; do
            # Wait for any one job to finish
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

        # Launch in background
        run_fmriprep_one "$ds_id" "$sub_id" &
        RUNNING_PIDS+=($!)
    fi
done

# Wait for remaining parallel jobs
for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid" && SUCCESS=$((SUCCESS + 1)) || FAIL=$((FAIL + 1))
done

# ---- Summary ----
echo ""
echo "============================================"
echo " fMRIPrep Batch Complete"
echo "  Total:    $TOTAL"
echo "  Success:  $SUCCESS"
echo "  Failed:   $FAIL"
echo "  Skipped:  $SKIP (already processed)"
echo ""
echo " Outputs: $DERIV_DIR"
echo "============================================"

# List processed subjects
if [ "$SUCCESS" -gt 0 ] || [ "$SKIP" -gt 0 ]; then
    echo ""
    echo "Next step: Run BS-NET on fMRIPrep outputs"
    echo "  python src/scripts/run_fmriprep_bsnet.py --run-all --verbose"
    echo "  python src/scripts/run_fmriprep_bsnet.py --run-selection data/hc_100_selection.csv"
fi
