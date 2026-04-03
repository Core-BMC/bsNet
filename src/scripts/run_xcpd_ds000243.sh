#!/usr/bin/env bash
# ============================================================================
# BS-NET: XCP-D Post-Processing for ds000243
# ============================================================================
# XCP-D v0.9.0, NIfTI mode, 36P, 4S256/4S456 parcellation
#
# 수행 내용:
#   1. native-space T1w symlink 생성 (fMRIPrep MNI-only 출력 대응)
#   2. XCP-D 실행 (36P denoising + bandpass + parcellation)
#   3. 이미 처리된 subject 자동 skip
#
# 사용법:
#   # 전체 실행
#   bash src/scripts/run_xcpd_ds000243.sh
#
#   # 단일 subject
#   bash src/scripts/run_xcpd_ds000243.sh --subject 001
#
#   # Dry-run (명령어만 출력)
#   bash src/scripts/run_xcpd_ds000243.sh --dry-run
#
#   # subject 범위 지정
#   bash src/scripts/run_xcpd_ds000243.sh --subject 001 002 003
# ============================================================================
set -euo pipefail

# ── 경로 설정 ────────────────────────────────────────────────────────────────
FMRIPREP_DIR="/home/hwon/dev/code/bsNet/data/derivatives/fmriprep"
XCPD_OUT_DIR="/home/hwon/dev/code/bsNet/data/derivatives/xcpd"
WORK_DIR="/home/hwon/dev/code/bsNet/data/ds000243/xcpd_work"
FS_LICENSE="/home/hwon/dev/code/bsNet/fs_license.txt"

# ── XCP-D 설정 ───────────────────────────────────────────────────────────────
XCPD_VERSION="0.9.0"
NPROCS=16
MEM_GB=24

# ── 인자 파싱 ────────────────────────────────────────────────────────────────
DRY_RUN=0
SUBJECTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --subject)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SUBJECTS+=("$1"); shift
            done
            ;;
        -h|--help)
            sed -n '3,20p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── 검증 ─────────────────────────────────────────────────────────────────────
if [ ! -d "$FMRIPREP_DIR" ]; then
    echo "ERROR: fMRIPrep directory not found: $FMRIPREP_DIR"; exit 1
fi
if [ ! -f "$FS_LICENSE" ]; then
    echo "ERROR: FreeSurfer license not found: $FS_LICENSE"; exit 1
fi
if [ "$DRY_RUN" -eq 0 ] && ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found in PATH"; exit 1
fi

# ── Subject 목록 구성 ────────────────────────────────────────────────────────
if [ ${#SUBJECTS[@]} -eq 0 ]; then
    for sub_dir in "$FMRIPREP_DIR"/sub-*; do
        [ -d "$sub_dir" ] || continue
        sub_id=$(basename "$sub_dir")
        [[ "$sub_id" == sub-*.html ]] && continue
        SUBJECTS+=("${sub_id#sub-}")
    done
fi

echo "============================================================"
echo " BS-NET XCP-D Pipeline (v${XCPD_VERSION})"
echo " Input  : $FMRIPREP_DIR"
echo " Output : $XCPD_OUT_DIR"
echo " Atlases: 4S256Parcels 4S456Parcels"
echo " Subjects: ${#SUBJECTS[@]}"
[ "$DRY_RUN" -eq 1 ] && echo " ** DRY RUN **"
echo "============================================================"
echo ""

# ── 처리 완료 확인 ───────────────────────────────────────────────────────────
is_done() {
    local label="$1"
    # 두 아틀라스 모두 timeseries TSV가 있으면 완료로 판단
    local out_func="$XCPD_OUT_DIR/sub-${label}/func"
    if [ -d "$out_func" ] && \
       ls "$out_func"/*4S256Parcels*timeseries*.tsv &>/dev/null 2>&1 && \
       ls "$out_func"/*4S456Parcels*timeseries*.tsv &>/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# ── Symlink 생성 (native-space T1w) ─────────────────────────────────────────
make_symlinks() {
    local label="$1"
    local anat_dir="$FMRIPREP_DIR/sub-${label}/anat"

    if [ ! -d "$anat_dir" ]; then
        echo "  WARN: anat dir not found: $anat_dir"
        return 1
    fi

    # T1w symlink
    local t1w_mni
    t1w_mni=$(ls "$anat_dir"/sub-"${label}"_space-MNI152NLin6Asym_*_desc-preproc_T1w.nii.gz \
               2>/dev/null | head -1)
    if [ -n "$t1w_mni" ] && [ ! -e "$anat_dir/sub-${label}_desc-preproc_T1w.nii.gz" ]; then
        ln -s "$(basename "$t1w_mni")" \
              "$anat_dir/sub-${label}_desc-preproc_T1w.nii.gz"
        echo "  symlink: sub-${label}_desc-preproc_T1w.nii.gz"
    fi

    # brain_mask symlink
    local mask_mni
    mask_mni=$(ls "$anat_dir"/sub-"${label}"_space-MNI152NLin6Asym_*_desc-brain_mask.nii.gz \
               2>/dev/null | head -1)
    if [ -n "$mask_mni" ] && [ ! -e "$anat_dir/sub-${label}_desc-brain_mask.nii.gz" ]; then
        ln -s "$(basename "$mask_mni")" \
              "$anat_dir/sub-${label}_desc-brain_mask.nii.gz"
        echo "  symlink: sub-${label}_desc-brain_mask.nii.gz"
    fi
}

# ── XCP-D 실행 ───────────────────────────────────────────────────────────────
run_xcpd() {
    local label="$1"
    local sub_work="$WORK_DIR/sub-${label}"
    mkdir -p "$sub_work" "$XCPD_OUT_DIR"

    local cmd=(
        docker run --rm
        --user "$(id -u):$(id -g)"
        -v "${FMRIPREP_DIR}:/data:ro"
        -v "${XCPD_OUT_DIR}:/out"
        -v "${sub_work}:/work"
        -v "${FS_LICENSE}:/opt/freesurfer/license.txt:ro"
        "pennlinc/xcp_d:${XCPD_VERSION}"
        /data /out participant
        --participant-label "${label}"
        --mode linc
        --input-type fmriprep
        --file-format nifti
        -p 36P
        --fd-thresh 0.5
        --lower-bpf 0.01
        --upper-bpf 0.1
        --smoothing 0
        --combine-runs y
        --atlases 4S256Parcels 4S456Parcels
        --min-time 120
        --nprocs "${NPROCS}"
        --mem-gb "${MEM_GB}"
        -w /work
        --notrack
        --fs-license-file /opt/freesurfer/license.txt
    )

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [DRY-RUN] ${cmd[*]}"
        return 0
    fi

    local log_file="$XCPD_OUT_DIR/sub-${label}_xcpd.log"
    echo "  Running XCP-D for sub-${label} (log: $log_file) ..."

    if "${cmd[@]}" > "$log_file" 2>&1; then
        echo "  OK: sub-${label} → $XCPD_OUT_DIR/sub-${label}/"
        # work 디렉토리 정리 (디스크 절약)
        rm -rf "$sub_work"
        return 0
    else
        echo "  FAIL: sub-${label} (see $log_file)"
        return 1
    fi
}

# ── 메인 루프 ────────────────────────────────────────────────────────────────
TOTAL=${#SUBJECTS[@]}
SUCCESS=0; FAIL=0; SKIP=0

for i in "${!SUBJECTS[@]}"; do
    label="${SUBJECTS[$i]}"
    idx=$((i + 1))
    echo "[${idx}/${TOTAL}] sub-${label}"

    # 완료 확인
    if is_done "$label"; then
        echo "  SKIP: already processed"
        SKIP=$((SKIP + 1))
        continue
    fi

    # fMRIPrep 출력 존재 확인
    if ! ls "$FMRIPREP_DIR/sub-${label}/func/"*desc-preproc_bold.nii.gz \
             &>/dev/null 2>&1; then
        echo "  SKIP: no fMRIPrep BOLD output"
        SKIP=$((SKIP + 1))
        continue
    fi

    # Symlink 생성
    make_symlinks "$label"

    # XCP-D 실행
    if run_xcpd "$label"; then
        SUCCESS=$((SUCCESS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
done

# ── 요약 ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " XCP-D Complete"
echo "  Total  : $TOTAL"
echo "  Success: $SUCCESS"
echo "  Failed : $FAIL"
echo "  Skipped: $SKIP (already done or no BOLD)"
echo ""
echo " Outputs: $XCPD_OUT_DIR"
echo "   Timeseries: sub-XXX/func/*4S256Parcels*timeseries.tsv"
echo "               sub-XXX/func/*4S456Parcels*timeseries.tsv"
echo ""
echo " Next: convert TSV → NPY"
echo "   python src/scripts/convert_xcpd_to_npy.py \\"
echo "     --xcpd-dir $XCPD_OUT_DIR \\"
echo "     --atlas 4S256Parcels 4S456Parcels"
echo "============================================================"
