#!/usr/bin/env bash
# ============================================================================
# BS-NET: fMRIPrep Manual Runner with Diagnostics
# ============================================================================
# 자동 파이프라인 실패 시 수동으로 fMRIPrep을 실행하는 스크립트.
# 1) 환경 진단 (Docker, FS license, BIDS 구조, 이미지)
# 2) 단일 subject docker run 명령 출력 또는 직접 실행
# 3) 배치 모드: 특정 dataset의 전체 subject를 순차 처리
#
# 사용법:
#   # 환경 진단만
#   ./src/scripts/run_fmriprep_manual.sh --check
#
#   # 단일 subject 명령어 출력 (복붙용)
#   ./src/scripts/run_fmriprep_manual.sh --subject sub-10159 --dataset ds000030 --print
#
#   # 단일 subject 실행
#   ./src/scripts/run_fmriprep_manual.sh --subject sub-10159 --dataset ds000030
#
#   # 배치: dataset 전체 (미처리 subject만)
#   ./src/scripts/run_fmriprep_manual.sh --dataset ds000030 --batch
#
#   # 배치: 최대 N명
#   ./src/scripts/run_fmriprep_manual.sh --dataset ds000030 --batch --max 5
#
#   # Singularity 사용
#   ./src/scripts/run_fmriprep_manual.sh --dataset ds000030 --batch --singularity
# ============================================================================

# ---- Configuration ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
OPENNEURO_DIR="$DATA_DIR/openneuro"
DERIV_DIR="$DATA_DIR/derivatives/fmriprep"

FMRIPREP_VERSION="24.1.1"
N_CPUS="${BSNET_NCPUS:-4}"
MEM_MB="${BSNET_MEM_MB:-16000}"
OUTPUT_SPACES="MNI152NLin6Asym:res-2"
FS_LICENSE="${FS_LICENSE:-$PROJECT_ROOT/fs_license.txt}"
SINGULARITY_IMG="${FMRIPREP_SIF:-$PROJECT_ROOT/fmriprep-${FMRIPREP_VERSION}.sif}"

# ---- Parse arguments ----
SUBJECT=""
DATASET=""
CHECK_ONLY=0
PRINT_ONLY=0
BATCH_MODE=0
MAX_SUBJECTS=0
USE_SINGULARITY=0
SKIP_ANAT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subject)       SUBJECT="$2"; shift 2 ;;
        --dataset)       DATASET="$2"; shift 2 ;;
        --check)         CHECK_ONLY=1; shift ;;
        --print)         PRINT_ONLY=1; shift ;;
        --batch)         BATCH_MODE=1; shift ;;
        --max)           MAX_SUBJECTS="$2"; shift 2 ;;
        --singularity)   USE_SINGULARITY=1; shift ;;
        --ncpus)         N_CPUS="$2"; shift 2 ;;
        --mem-mb)        MEM_MB="$2"; shift 2 ;;
        --fs-license)    FS_LICENSE="$2"; shift 2 ;;
        --skip-anat)     SKIP_ANAT=1; shift ;;
        -h|--help)
            sed -n '/^# 사용법:/,/^# ====/p' "$0" | head -n -1
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Color helpers ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }
info() { echo -e "  ${CYAN}→${NC} $*"; }

# ============================================================================
# 환경 진단
# ============================================================================
run_diagnostics() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " fMRIPrep Environment Diagnostics"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    local errors=0

    # 1. Docker / Singularity
    echo ""
    echo "[1/5] Container Runtime"
    if [ "$USE_SINGULARITY" -eq 1 ]; then
        if command -v singularity &>/dev/null; then
            ok "Singularity: $(singularity --version 2>/dev/null || echo 'found')"
        else
            fail "Singularity not found in PATH"
            errors=$((errors + 1))
        fi
        if [ -f "$SINGULARITY_IMG" ]; then
            ok "Singularity image: $SINGULARITY_IMG"
        else
            fail "Singularity image not found: $SINGULARITY_IMG"
            info "Build: singularity build fmriprep-${FMRIPREP_VERSION}.sif docker://nipreps/fmriprep:${FMRIPREP_VERSION}"
            errors=$((errors + 1))
        fi
    else
        if command -v docker &>/dev/null; then
            ok "Docker: $(docker --version 2>/dev/null | head -1)"
        else
            fail "Docker not found in PATH"
            info "Install: https://docs.docker.com/engine/install/"
            errors=$((errors + 1))
        fi

        # Docker daemon running?
        if docker info &>/dev/null 2>&1; then
            ok "Docker daemon: running"
        else
            fail "Docker daemon: not running or permission denied"
            info "Try: sudo systemctl start docker"
            info "Or:  sudo usermod -aG docker \$USER && newgrp docker"
            errors=$((errors + 1))
        fi

        # fMRIPrep image
        if docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -q "nipreps/fmriprep:${FMRIPREP_VERSION}"; then
            ok "fMRIPrep image: nipreps/fmriprep:${FMRIPREP_VERSION}"
        else
            fail "fMRIPrep image not found locally"
            info "Pull: docker pull nipreps/fmriprep:${FMRIPREP_VERSION}"
            errors=$((errors + 1))
        fi
    fi

    # 2. FreeSurfer License
    echo ""
    echo "[2/5] FreeSurfer License"
    if [ -f "$FS_LICENSE" ]; then
        ok "License: $FS_LICENSE ($(wc -l < "$FS_LICENSE") lines)"
    else
        fail "License not found: $FS_LICENSE"
        info "Get free license: https://surfer.nmr.mgh.harvard.edu/registration.html"
        info "Then: cp license.txt $PROJECT_ROOT/fs_license.txt"
        info "Or:   export FS_LICENSE=/path/to/license.txt"
        errors=$((errors + 1))
    fi

    # 3. Data directory
    echo ""
    echo "[3/5] Data Directory"
    if [ -d "$OPENNEURO_DIR" ]; then
        local n_datasets
        n_datasets=$(find "$OPENNEURO_DIR" -maxdepth 1 -name 'ds*' -type d 2>/dev/null | wc -l)
        ok "OpenNeuro dir: $OPENNEURO_DIR ($n_datasets datasets)"
        for ds_dir in "$OPENNEURO_DIR"/ds*; do
            [ -d "$ds_dir" ] || continue
            local ds_id n_subs
            ds_id=$(basename "$ds_dir")
            n_subs=$(find "$ds_dir" -maxdepth 1 -name 'sub-*' -type d 2>/dev/null | wc -l)
            info "$ds_id: $n_subs subjects"
        done
    else
        fail "OpenNeuro dir not found: $OPENNEURO_DIR"
        errors=$((errors + 1))
    fi

    # 4. Derivatives directory
    echo ""
    echo "[4/5] Derivatives"
    if [ -d "$DERIV_DIR" ]; then
        local n_done
        n_done=$(find "$DERIV_DIR" -maxdepth 1 -name 'sub-*' -type d 2>/dev/null | wc -l)
        ok "fMRIPrep output: $DERIV_DIR ($n_done subjects processed)"
    else
        warn "fMRIPrep output dir does not exist yet (will be created)"
    fi

    # 5. Disk space
    echo ""
    echo "[5/5] Disk Space"
    local avail_gb
    avail_gb=$(df -BG "$DATA_DIR" 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
    if [ -n "$avail_gb" ] && [ "$avail_gb" -gt 50 ]; then
        ok "Available: ${avail_gb}GB (recommend >50GB)"
    elif [ -n "$avail_gb" ]; then
        warn "Available: ${avail_gb}GB (low — recommend >50GB)"
    else
        warn "Could not determine disk space"
    fi

    # Summary
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ "$errors" -eq 0 ]; then
        echo -e " ${GREEN}All checks passed.${NC} Ready to run fMRIPrep."
    else
        echo -e " ${RED}${errors} issue(s) found.${NC} Fix before running."
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    return "$errors"
}

# ============================================================================
# BIDS 유효성 검사 (단일 subject)
# ============================================================================
check_bids_subject() {
    local ds_id="$1"
    local sub_id="$2"
    local bids_dir="$OPENNEURO_DIR/$ds_id"
    local sub_dir="$bids_dir/$sub_id"
    local ok=1

    if [ ! -d "$sub_dir" ]; then
        fail "Subject dir not found: $sub_dir"
        return 1
    fi

    # Check anat
    if ls "$sub_dir"/anat/*T1w*.nii* &>/dev/null 2>&1 || \
       ls "$sub_dir"/ses-*/anat/*T1w*.nii* &>/dev/null 2>&1; then
        ok "anat: T1w found"
    else
        if [ "$SKIP_ANAT" -eq 1 ]; then
            warn "anat: T1w not found (--skip-anat enabled, will use --anat-derivatives)"
        else
            fail "anat: T1w not found in $sub_dir/anat/ or $sub_dir/ses-*/anat/"
            ok=0
        fi
    fi

    # Check func
    if ls "$sub_dir"/func/*bold*.nii* &>/dev/null 2>&1 || \
       ls "$sub_dir"/ses-*/func/*bold*.nii* &>/dev/null 2>&1; then
        local n_bold
        n_bold=$(find "$sub_dir" -name '*bold*.nii*' 2>/dev/null | wc -l)
        ok "func: $n_bold BOLD file(s)"
    else
        fail "func: No BOLD files found"
        ok=0
    fi

    # dataset_description.json
    if [ ! -f "$bids_dir/dataset_description.json" ]; then
        warn "dataset_description.json missing (will auto-create)"
    fi

    return $((1 - ok))
}

# ============================================================================
# Docker/Singularity 명령 생성
# ============================================================================
build_fmriprep_cmd() {
    local ds_id="$1"
    local sub_id="$2"
    local bids_dir="$OPENNEURO_DIR/$ds_id"
    local work_dir="$DATA_DIR/fmriprep_work/${ds_id}_${sub_id}"

    # Auto-create dataset_description.json
    if [ ! -f "$bids_dir/dataset_description.json" ]; then
        echo '{"Name":"OpenNeuro","BIDSVersion":"1.6.0","DatasetType":"raw"}' \
            > "$bids_dir/dataset_description.json"
    fi

    if [ "$USE_SINGULARITY" -eq 1 ]; then
        echo "singularity run --cleanenv \\"
        echo "  -B ${bids_dir}:/data:ro \\"
        echo "  -B ${DERIV_DIR}:/out \\"
        echo "  -B ${work_dir}:/work \\"
        echo "  -B ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \\"
        echo "  ${SINGULARITY_IMG} \\"
        echo "  /data /out participant \\"
        echo "  --participant-label ${sub_id#sub-} \\"
        echo "  --output-spaces ${OUTPUT_SPACES} \\"
        echo "  --nprocs ${N_CPUS} --mem-mb ${MEM_MB} \\"
        echo "  --work-dir /work \\"
        echo "  --skip-bids-validation --notrack \\"
        echo "  --ignore fieldmaps slicetiming"
    else
        echo "docker run --rm \\"
        echo "  -v ${bids_dir}:/data:ro \\"
        echo "  -v ${DERIV_DIR}:/out \\"
        echo "  -v ${work_dir}:/work \\"
        echo "  -v ${FS_LICENSE}:/opt/freesurfer/license.txt:ro \\"
        echo "  nipreps/fmriprep:${FMRIPREP_VERSION} \\"
        echo "  /data /out participant \\"
        echo "  --participant-label ${sub_id#sub-} \\"
        echo "  --output-spaces ${OUTPUT_SPACES} \\"
        echo "  --nprocs ${N_CPUS} --mem-mb ${MEM_MB} \\"
        echo "  --work-dir /work \\"
        echo "  --skip-bids-validation --notrack \\"
        echo "  --ignore fieldmaps slicetiming"
    fi
}

# ============================================================================
# 후처리 정리: 필요한 파일만 유지, 나머지 삭제
# ============================================================================
cleanup_subject() {
    local sub_id="$1"
    local ds_id="$2"
    local work_dir="$DATA_DIR/fmriprep_work/${ds_id}_${sub_id}"
    local sub_deriv="$DERIV_DIR/$sub_id"
    local freed=0

    echo ""
    echo "── Cleanup: $sub_id ──"

    # 1) Work dir 삭제 (가장 큼, 10~30GB)
    if [ -d "$work_dir" ]; then
        local work_size
        work_size=$(du -sm "$work_dir" 2>/dev/null | awk '{print $1}')
        rm -rf "$work_dir"
        freed=$((freed + work_size))
        ok "Work dir removed (${work_size}MB)"
    fi

    # 2) FreeSurfer recon-all 출력 삭제 (~300MB/subject, BS-NET 불필요)
    local fs_dir="$DERIV_DIR/sourcedata/freesurfer/$sub_id"
    if [ -d "$fs_dir" ]; then
        local fs_size
        fs_size=$(du -sm "$fs_dir" 2>/dev/null | awk '{print $1}')
        rm -rf "$fs_dir"
        freed=$((freed + fs_size))
        ok "FreeSurfer outputs removed (${fs_size}MB)"
    fi

    # 3) derivatives 내 불필요 파일 정리 (필요한 것만 유지)
    #    유지 목록:
    #      func/*MNI152NLin6Asym*_desc-preproc_bold.nii.gz  (전처리된 BOLD)
    #      func/*MNI152NLin6Asym*_boldref.nii.gz            (BOLD reference)
    #      func/*MNI152NLin6Asym*_brain_mask.nii.gz         (brain mask)
    #      func/*_desc-confounds_timeseries.tsv              (XCP-D 입력)
    #      func/*_desc-confounds_timeseries.json             (confounds 메타)
    #      *.html (QC report)
    #      figures/ (QC figures, 작음)
    if [ -d "$sub_deriv" ]; then
        local before_size
        before_size=$(du -sm "$sub_deriv" 2>/dev/null | awk '{print $1}')

        # func/ 내 불필요 파일 삭제
        if [ -d "$sub_deriv/func" ]; then
            find "$sub_deriv/func" -type f \
                ! -name '*MNI152NLin6Asym*_desc-preproc_bold.nii.gz' \
                ! -name '*MNI152NLin6Asym*_boldref.nii.gz' \
                ! -name '*MNI152NLin6Asym*_desc-brain_mask.nii.gz' \
                ! -name '*_desc-confounds_timeseries.tsv' \
                ! -name '*_desc-confounds_timeseries.json' \
                -delete 2>/dev/null
        fi

        # anat/ 내 불필요 파일 삭제 (MNI space T1w만 유지)
        if [ -d "$sub_deriv/anat" ]; then
            find "$sub_deriv/anat" -type f \
                ! -name '*MNI152NLin6Asym*' \
                -delete 2>/dev/null
        fi

        # ses-*/func, ses-*/anat 도 동일 처리
        for ses_dir in "$sub_deriv"/ses-*; do
            [ -d "$ses_dir" ] || continue
            if [ -d "$ses_dir/func" ]; then
                find "$ses_dir/func" -type f \
                    ! -name '*MNI152NLin6Asym*_desc-preproc_bold.nii.gz' \
                    ! -name '*MNI152NLin6Asym*_boldref.nii.gz' \
                    ! -name '*MNI152NLin6Asym*_desc-brain_mask.nii.gz' \
                    ! -name '*_desc-confounds_timeseries.tsv' \
                    ! -name '*_desc-confounds_timeseries.json' \
                    -delete 2>/dev/null
            fi
            if [ -d "$ses_dir/anat" ]; then
                find "$ses_dir/anat" -type f \
                    ! -name '*MNI152NLin6Asym*' \
                    -delete 2>/dev/null
            fi
        done

        # 빈 디렉토리 정리
        find "$sub_deriv" -type d -empty -delete 2>/dev/null

        local after_size
        after_size=$(du -sm "$sub_deriv" 2>/dev/null | awk '{print $1}')
        local deriv_freed=$((before_size - after_size))
        freed=$((freed + deriv_freed))
        ok "Derivatives trimmed: ${before_size}MB → ${after_size}MB (saved ${deriv_freed}MB)"
    fi

    info "Total freed: ${freed}MB"
}

# ============================================================================
# 단일 subject 실행
# ============================================================================
run_one_subject() {
    local ds_id="$1"
    local sub_id="$2"
    local bids_dir="$OPENNEURO_DIR/$ds_id"
    local work_dir="$DATA_DIR/fmriprep_work/${ds_id}_${sub_id}"
    local log_file="$DATA_DIR/logs/fmriprep_${ds_id}_${sub_id}.log"

    # Already processed?
    if [ -f "$DERIV_DIR/${sub_id}.html" ] && [ -d "$DERIV_DIR/${sub_id}/func" ]; then
        warn "$sub_id already processed. Skipping."
        return 0
    fi

    # BIDS check
    echo ""
    echo "── BIDS Check: $sub_id ──"
    if ! check_bids_subject "$ds_id" "$sub_id"; then
        fail "BIDS validation failed for $sub_id"
        return 1
    fi

    # Auto-create dirs
    mkdir -p "$work_dir" "$DERIV_DIR" "$(dirname "$log_file")"

    # Auto-create dataset_description.json
    if [ ! -f "$bids_dir/dataset_description.json" ]; then
        echo '{"Name":"OpenNeuro","BIDSVersion":"1.6.0","DatasetType":"raw"}' \
            > "$bids_dir/dataset_description.json"
    fi

    # Build command array (not string)
    local -a cmd
    if [ "$USE_SINGULARITY" -eq 1 ]; then
        cmd=(
            singularity run --cleanenv
            -B "$bids_dir":/data:ro
            -B "$DERIV_DIR":/out
            -B "$work_dir":/work
            -B "$FS_LICENSE":/opt/freesurfer/license.txt:ro
            "$SINGULARITY_IMG"
            /data /out participant
            --participant-label "${sub_id#sub-}"
            --output-spaces "$OUTPUT_SPACES"
            --nprocs "$N_CPUS" --mem-mb "$MEM_MB"
            --work-dir /work
            --skip-bids-validation --notrack
            --ignore fieldmaps slicetiming
        )
    else
        cmd=(
            docker run --rm
            -v "$bids_dir":/data:ro
            -v "$DERIV_DIR":/out
            -v "$work_dir":/work
            -v "$FS_LICENSE":/opt/freesurfer/license.txt:ro
            nipreps/fmriprep:"$FMRIPREP_VERSION"
            /data /out participant
            --participant-label "${sub_id#sub-}"
            --output-spaces "$OUTPUT_SPACES"
            --nprocs "$N_CPUS" --mem-mb "$MEM_MB"
            --work-dir /work
            --skip-bids-validation --notrack
            --ignore fieldmaps slicetiming
        )
    fi

    echo ""
    info "Running: ${cmd[*]}"
    info "Log: $log_file"
    echo ""

    local start_time
    start_time=$(date +%s)

    if "${cmd[@]}" > "$log_file" 2>&1; then
        local elapsed=$(( $(date +%s) - start_time ))
        ok "$sub_id completed (${elapsed}s)"
        cleanup_subject "$sub_id" "$ds_id"
        return 0
    else
        local elapsed=$(( $(date +%s) - start_time ))
        fail "$sub_id failed after ${elapsed}s"
        echo ""
        echo "── Last 20 lines of log ──"
        tail -20 "$log_file" 2>/dev/null
        echo ""
        info "Full log: $log_file"
        # 실패해도 work dir은 정리 (디스크 회수)
        if [ -d "$work_dir" ]; then
            local work_size
            work_size=$(du -sm "$work_dir" 2>/dev/null | awk '{print $1}')
            rm -rf "$work_dir"
            info "Failed work dir removed (${work_size}MB freed)"
        fi
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================

# --check: 진단만
if [ "$CHECK_ONLY" -eq 1 ]; then
    run_diagnostics
    exit $?
fi

# 항상 진단 먼저 (에러 있으면 중단)
run_diagnostics
DIAG_RESULT=$?

if [ "$DIAG_RESULT" -ne 0 ] && [ "$PRINT_ONLY" -eq 0 ]; then
    echo "Fix the above issues before running fMRIPrep."
    echo "Or use --print to just see the commands."
    exit 1
fi

# --print: 명령어만 출력
if [ "$PRINT_ONLY" -eq 1 ]; then
    if [ -z "$SUBJECT" ] || [ -z "$DATASET" ]; then
        echo "ERROR: --print requires --subject and --dataset"
        exit 1
    fi
    echo ""
    echo "── fMRIPrep command for $SUBJECT ($DATASET) ──"
    echo ""
    build_fmriprep_cmd "$DATASET" "$SUBJECT"
    echo ""
    echo "# Copy-paste the above command to run manually."
    echo "# Pre-create dirs:"
    echo "mkdir -p $DERIV_DIR $DATA_DIR/fmriprep_work/${DATASET}_${SUBJECT}"
    exit 0
fi

# --batch: dataset 전체 순차 처리
if [ "$BATCH_MODE" -eq 1 ]; then
    if [ -z "$DATASET" ]; then
        echo "ERROR: --batch requires --dataset"
        exit 1
    fi

    ds_dir="$OPENNEURO_DIR/$DATASET"
    if [ ! -d "$ds_dir" ]; then
        echo "ERROR: Dataset dir not found: $ds_dir"
        exit 1
    fi

    # Collect unprocessed subjects
    declare -a PENDING=()
    for sub_dir in "$ds_dir"/sub-*; do
        [ -d "$sub_dir" ] || continue
        sub_id=$(basename "$sub_dir")
        if [ -f "$DERIV_DIR/${sub_id}.html" ] && [ -d "$DERIV_DIR/${sub_id}/func" ]; then
            continue  # already done
        fi
        PENDING+=("$sub_id")
    done

    total=${#PENDING[@]}
    if [ "$total" -eq 0 ]; then
        echo "All subjects in $DATASET are already processed."
        exit 0
    fi

    # Apply --max limit
    if [ "$MAX_SUBJECTS" -gt 0 ] && [ "$MAX_SUBJECTS" -lt "$total" ]; then
        PENDING=("${PENDING[@]:0:$MAX_SUBJECTS}")
        echo "Processing ${MAX_SUBJECTS}/${total} pending subjects (--max $MAX_SUBJECTS)"
    else
        echo "Processing $total pending subjects"
    fi

    echo ""
    SUCCESS=0
    FAIL=0
    for i in "${!PENDING[@]}"; do
        sub_id="${PENDING[$i]}"
        idx=$((i + 1))
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo " [${idx}/${#PENDING[@]}] $DATASET / $sub_id"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        if run_one_subject "$DATASET" "$sub_id"; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAIL=$((FAIL + 1))
            echo ""
            warn "Continue with next subject? (Ctrl+C to abort)"
            sleep 3
        fi
    done

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Batch Complete: $DATASET"
    echo "  Success: $SUCCESS / ${#PENDING[@]}"
    echo "  Failed:  $FAIL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
fi

# 단일 subject 실행
if [ -n "$SUBJECT" ] && [ -n "$DATASET" ]; then
    run_one_subject "$DATASET" "$SUBJECT"
    exit $?
fi

echo "ERROR: Specify --check, --print, --batch, or --subject/--dataset"
echo "Run with -h for help"
exit 1
