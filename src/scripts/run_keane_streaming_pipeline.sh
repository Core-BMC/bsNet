#!/usr/bin/env bash
# ============================================================================
# BS-NET: Keane Streaming Pipeline (subject-by-subject)
# ============================================================================
# 목적:
#   subject 단위로 아래를 순차 실행하여 디스크 사용량을 최소화한다.
#     1) (옵션) datalad get 으로 subject 원본 다운로드
#     2) fMRIPrep (REST task only)
#     3) BS-NET(fMRIPrep-direct) 실행 -> *_ts.npy 포함 산출물 생성
#     4) cleanup 정책에 따라 raw/fmriprep 중간산출 삭제
#
# 사용 예:
#   # sub-B06 1명 처리, 최소 보존 정책
#   bash src/scripts/run_keane_streaming_pipeline.sh \
#     --dataset ds005073 \
#     --subject sub-B06 \
#     --cleanup-level minimal
#
#   # subject 자동 감지 (dataset의 sub-* 또는 participants.tsv 기반)
#   bash src/scripts/run_keane_streaming_pipeline.sh \
#     --dataset ds005073 \
#     --cleanup-level minimal
#
#   # 여러 subject 처리 + dry-run
#   bash src/scripts/run_keane_streaming_pipeline.sh \
#     --dataset ds005073 \
#     --subject sub-B06 sub-S06 \
#     --cleanup-level minimal \
#     --dry-run
#
#   # subject 디렉토리 없으면 datalad get 시도
#   bash src/scripts/run_keane_streaming_pipeline.sh \
#     --dataset ds003404 \
#     --subject sub-C05 \
#     --auto-datalad-get \
#     --cleanup-level minimal
#
#   # datalad clone 자동 설치 + subject on-demand get
#   bash src/scripts/run_keane_streaming_pipeline.sh \
#     --dataset ds005073 \
#     --auto-datalad-install \
#     --auto-datalad-get \
#     --cleanup-level minimal
#
# cleanup-level:
#   - minimal: bsnet 출력만 보존 (raw subject + fmriprep subject/work 삭제)
#   - debug  : raw subject는 삭제, fmriprep 결과는 보존
#   - full   : 삭제 없음
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="$PROJECT_ROOT/data"
FMRIPREP_ROOT="$DATA_ROOT/derivatives/fmriprep_keane"
WORK_ROOT="$DATA_ROOT/derivatives/fmriprep_keane_work"
BSNET_OUT="$DATA_ROOT/derivatives/bsnet"

DATASET="ds005073"
SUBJECTS=()
PARCELS=100
N_CPUS="${BSNET_NCPUS:-8}"
MEM_MB="${BSNET_MEM_MB:-12000}"
VERBOSE=0
DRY_RUN=0
AUTO_DATALAD_GET=0
AUTO_DATALAD_INSTALL=0
CLEANUP_LEVEL="minimal"   # minimal|debug|full
HAS_DATALAD=0

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] $*"
        return 0
    fi
    "$@"
}

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
err() { echo "[ERR ] $*" >&2; }

discover_subjects_from_participants() {
    local ds_dir="$1"
    local tsv="$ds_dir/participants.tsv"
    [[ -f "$tsv" ]] || return 1

    # Find participant_id column index from header.
    local header
    header="$(head -n 1 "$tsv")"
    local idx=0
    local pid_col=0
    IFS=$'\t' read -r -a cols <<< "$header"
    for c in "${cols[@]}"; do
        idx=$((idx + 1))
        if [[ "$c" == "participant_id" ]]; then
            pid_col="$idx"
            break
        fi
    done
    [[ "$pid_col" -gt 0 ]] || return 1

    # Emit unique non-empty participant IDs.
    awk -F'\t' -v col="$pid_col" '
        NR > 1 {
            gsub(/^[ \t]+|[ \t]+$/, "", $col);
            if ($col != "") print $col
        }
    ' "$tsv" | sort -u
    return 0
}

usage() {
    sed -n '3,56p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --subject)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SUBJECTS+=("$1")
                shift
            done
            ;;
        --parcels) PARCELS="$2"; shift 2 ;;
        --ncpus) N_CPUS="$2"; shift 2 ;;
        --mem-mb) MEM_MB="$2"; shift 2 ;;
        --cleanup-level) CLEANUP_LEVEL="$2"; shift 2 ;;
        --auto-datalad-get) AUTO_DATALAD_GET=1; shift ;;
        --auto-datalad-install) AUTO_DATALAD_INSTALL=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --verbose) VERBOSE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

case "$DATASET" in
    ds003404|ds005073) ;;
    *) err "--dataset must be ds003404 or ds005073"; exit 1 ;;
esac

case "$CLEANUP_LEVEL" in
    minimal|debug|full) ;;
    *) err "--cleanup-level must be minimal|debug|full"; exit 1 ;;
esac

if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    ds_dir="$DATA_ROOT/$DATASET"
    if [[ ! -d "$ds_dir" ]]; then
        err "Dataset directory not found for auto-discovery: $ds_dir"
        exit 1
    fi
    # 1) Prefer physically present subject directories.
    while IFS= read -r sub; do
        [[ -n "$sub" ]] && SUBJECTS+=("$sub")
    done < <(find "$ds_dir" -maxdepth 1 -type d -name 'sub-*' -print | xargs -I{} basename "{}" | sort || true)

    # 2) Fallback to participants.tsv (for not-yet-downloaded subjects).
    if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
        while IFS= read -r sub; do
            [[ -n "$sub" ]] && SUBJECTS+=("$sub")
        done < <(discover_subjects_from_participants "$ds_dir" || true)
    fi

    if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
        err "No subjects found in $ds_dir (sub-* or participants.tsv participant_id)"
        err "Provide explicit --subject or ensure participants.tsv exists."
        exit 1
    fi
    log "Auto-discovered subjects: ${#SUBJECTS[@]}"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    command -v python3 >/dev/null 2>&1 || { err "python3 not found"; exit 1; }
fi
if command -v datalad >/dev/null 2>&1; then
    HAS_DATALAD=1
fi

ensure_subject_present() {
    local ds="$1"
    local sub="$2"
    local sub_dir="$DATA_ROOT/$ds/$sub"

    if [[ -d "$sub_dir" ]]; then
        # Even when directory exists, ensure annexed payload is materialized.
        if [[ "$AUTO_DATALAD_GET" -eq 1 && "$HAS_DATALAD" -eq 1 ]]; then
            log "datalad get (refresh): $ds/$sub"
            if ! run_cmd datalad get -r "$sub_dir"; then
                warn "datalad get refresh failed for $sub_dir (continue with local files)"
            fi
        fi
        return 0
    fi
    if [[ "$AUTO_DATALAD_GET" -eq 0 ]]; then
        warn "Missing subject directory: $sub_dir (skip)"
        warn "Use --auto-datalad-get to fetch on demand."
        return 2
    fi
    if [[ "$HAS_DATALAD" -eq 0 ]]; then
        warn "datalad not found; cannot auto-download $sub (skip)"
        return 2
    fi
    log "datalad get: $ds/$sub"
    if ! run_cmd datalad get -r "$sub_dir"; then
        warn "datalad get failed for $sub_dir (skip)"
        warn "Current dataset tree may not be a datalad-install with retrievable subject paths."
        return 2
    fi
    if [[ ! -d "$sub_dir" ]]; then
        warn "Subject still missing after datalad get: $sub_dir (skip)"
        return 2
    fi
}

run_fmriprep_one() {
    local ds="$1"
    local sub="$2"
    local extra=()
    if [[ "$DRY_RUN" -eq 1 ]]; then
        extra+=(--dry-run)
    fi
    log "fMRIPrep: $ds/$sub"
    run_cmd bash "$SCRIPT_DIR/run_fmriprep_keane.sh" \
        --dataset "$ds" \
        --subject "$sub" \
        --ncpus "$N_CPUS" \
        --mem-mb "$MEM_MB" \
        "${extra[@]}"
}

run_bsnet_one() {
    local ds="$1"
    local sub="$2"
    local fmriprep_dir="$FMRIPREP_ROOT/$ds"
    local extra=()
    if [[ "$VERBOSE" -eq 1 ]]; then
        extra+=(--verbose)
    fi

    log "BS-NET (fmriprep-direct): $ds/$sub"
    run_cmd python3 -m src.scripts.run_fmriprep_bsnet \
        --subject "$sub" \
        --input-mode fmriprep \
        --fmriprep-dir "$fmriprep_dir" \
        --parcels "$PARCELS" \
        "${extra[@]}"

    # sanity check for timeseries npy
    local ts_npy="$BSNET_OUT/$sub/${sub}_ts.npy"
    if [[ "$DRY_RUN" -eq 0 ]] && [[ ! -f "$ts_npy" ]]; then
        err "Expected timeseries output missing: $ts_npy"
        return 1
    fi
    if [[ "$DRY_RUN" -eq 0 ]]; then
        log "Generated: $ts_npy"
    fi
    return 0
}

cleanup_one() {
    local ds="$1"
    local sub="$2"
    local raw_sub="$DATA_ROOT/$ds/$sub"
    local fmri_sub="$FMRIPREP_ROOT/$ds/$sub"
    local fmri_html="$FMRIPREP_ROOT/$ds/${sub}.html"
    local fmri_log="$FMRIPREP_ROOT/$ds/${sub}_fmriprep.log"
    local fmri_figs="$FMRIPREP_ROOT/$ds/figures/${sub}"  # may not exist
    local work_sub="$WORK_ROOT/${ds}_${sub}"
    local work_bids="$WORK_ROOT/${ds}_${sub}_bids"

    case "$CLEANUP_LEVEL" in
        full)
            log "Cleanup(full): skip deletion for $ds/$sub"
            ;;
        debug)
            log "Cleanup(debug): remove raw + work only for $ds/$sub"
            if [[ -d "$raw_sub" ]]; then
                run_cmd rm -rf "$raw_sub"
            fi
            if [[ -d "$work_sub" ]]; then
                run_cmd rm -rf "$work_sub"
            fi
            if [[ -d "$work_bids" ]]; then
                run_cmd rm -rf "$work_bids"
            fi
            ;;
        minimal)
            log "Cleanup(minimal): keep bsnet only, remove raw/fmriprep/work for $ds/$sub"
            if [[ -d "$raw_sub" ]]; then
                run_cmd rm -rf "$raw_sub"
            fi
            if [[ -d "$fmri_sub" ]]; then
                run_cmd rm -rf "$fmri_sub"
            fi
            if [[ -f "$fmri_html" ]]; then
                run_cmd rm -f "$fmri_html"
            fi
            if [[ -f "$fmri_log" ]]; then
                run_cmd rm -f "$fmri_log"
            fi
            if [[ -d "$fmri_figs" ]]; then
                run_cmd rm -rf "$fmri_figs"
            fi
            if [[ -d "$work_sub" ]]; then
                run_cmd rm -rf "$work_sub"
            fi
            if [[ -d "$work_bids" ]]; then
                run_cmd rm -rf "$work_bids"
            fi
            ;;
    esac
}

echo "============================================================"
echo " Keane Streaming Pipeline"
echo " Dataset      : $DATASET"
echo " Subjects     : ${SUBJECTS[*]}"
echo " Parcels      : $PARCELS"
echo " NCPUS/MEM_MB : $N_CPUS / $MEM_MB"
echo " Cleanup      : $CLEANUP_LEVEL"
echo " AutoDownload : $AUTO_DATALAD_GET"
[[ "$DRY_RUN" -eq 1 ]] && echo " ** DRY RUN **"
echo "============================================================"

N_OK=0
N_FAIL=0
N_SKIP=0

for sub in "${SUBJECTS[@]}"; do
    echo ""
    echo "---- [$sub] ----"
    ensure_subject_present "$DATASET" "$sub" || rc=$?
    rc=${rc:-0}
    if [[ "$rc" -eq 2 ]]; then
        N_SKIP=$((N_SKIP + 1))
        rc=0
        continue
    elif [[ "$rc" -ne 0 ]]; then
        N_FAIL=$((N_FAIL + 1))
        rc=0
        continue
    fi
    if ! run_fmriprep_one "$DATASET" "$sub"; then
        err "fMRIPrep failed: $sub"
        N_FAIL=$((N_FAIL + 1))
        continue
    fi
    if ! run_bsnet_one "$DATASET" "$sub"; then
        err "BS-NET failed: $sub"
        N_FAIL=$((N_FAIL + 1))
        continue
    fi
    cleanup_one "$DATASET" "$sub"
    N_OK=$((N_OK + 1))
done

echo ""
echo "============================================================"
echo " Streaming Summary"
echo "   Success: $N_OK"
echo "   Skipped: $N_SKIP"
echo "   Failed : $N_FAIL"
echo "============================================================"

if [[ "$N_FAIL" -gt 0 ]]; then
    exit 1
fi
