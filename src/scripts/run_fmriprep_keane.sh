#!/usr/bin/env bash
# ============================================================================
# BS-NET: fMRIPrep Runner for Keane Datasets (ds003404, ds005073)
# ============================================================================
# 대상 데이터셋:
#   - data/ds003404 (HC)
#   - data/ds005073 (BP/SZ)
#
# 특징:
#   - REST task만 처리 (--task-id rest)
#   - 최소 파일 구성(anat + rest bold/json)에서도 실행 가능
#   - 미완료 subject 자동 skip
#
# 사용법:
#   # ds005073 전체 대상 (sub-*/anat + sub-*/func/*task-rest* 존재 대상)
#   bash src/scripts/run_fmriprep_keane.sh --dataset ds005073
#
#   # ds003404 + ds005073 모두 처리
#   bash src/scripts/run_fmriprep_keane.sh --dataset all
#
#   # 특정 subject만 처리
#   bash src/scripts/run_fmriprep_keane.sh --dataset ds005073 --subject sub-B06 sub-S06
#
#   # Dry-run
#   bash src/scripts/run_fmriprep_keane.sh --dataset ds005073 --dry-run
#
# 환경변수(선택):
#   BSNET_NCPUS, BSNET_MEM_MB, FS_LICENSE, FMRIPREP_VERSION
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_ROOT="$PROJECT_ROOT/data"
DERIV_ROOT="$DATA_ROOT/derivatives/fmriprep_keane"
WORK_ROOT="$DATA_ROOT/derivatives/fmriprep_keane_work"

FMRIPREP_VERSION="${FMRIPREP_VERSION:-25.2.5}"
N_CPUS="${BSNET_NCPUS:-8}"
MEM_MB="${BSNET_MEM_MB:-12000}"
OUTPUT_SPACES="${BSNET_OUTPUT_SPACES:-MNI152NLin6Asym:res-2}"
FS_LICENSE="${FS_LICENSE:-$PROJECT_ROOT/fs_license.txt}"

DATASET="ds005073"
DRY_RUN=0
SUBJECTS=()

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
        --dry-run) DRY_RUN=1; shift ;;
        --ncpus) N_CPUS="$2"; shift 2 ;;
        --mem-mb) MEM_MB="$2"; shift 2 ;;
        --fs-license) FS_LICENSE="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,45p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$DRY_RUN" -eq 0 ]]; then
    command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
    [[ -f "$FS_LICENSE" ]] || { echo "ERROR: FreeSurfer license not found: $FS_LICENSE"; exit 1; }
fi

case "$DATASET" in
    ds003404|ds005073|all) ;;
    *)
        echo "ERROR: --dataset must be one of: ds003404, ds005073, all"
        exit 1
        ;;
esac

declare -a DATASETS
if [[ "$DATASET" == "all" ]]; then
    DATASETS=("ds003404" "ds005073")
else
    DATASETS=("$DATASET")
fi

is_done() {
    local ds_id="$1"
    local sub_id="$2"
    local out_dir="$DERIV_ROOT/$ds_id/$sub_id/func"
    local html="$DERIV_ROOT/$ds_id/${sub_id}.html"
    if [[ -d "$out_dir" ]] && ls "$out_dir"/*task-rest*desc-preproc_bold.nii.gz >/dev/null 2>&1 && [[ -f "$html" ]]; then
        return 0
    fi
    return 1
}

has_min_inputs() {
    local bids_dir="$1"
    local sub_id="$2"
    local sub_dir="$bids_dir/$sub_id"

    ls "$sub_dir"/anat/*T1w*.nii* >/dev/null 2>&1 || return 1
    ls "$sub_dir"/func/*task-rest*bold.nii* >/dev/null 2>&1 || return 1
    return 0
}

run_one() {
    local ds_id="$1"
    local sub_id="$2"
    local bids_dir="$DATA_ROOT/$ds_id"
    local staging_dir="$WORK_ROOT/${ds_id}_${sub_id}_bids"
    local out_dir="$DERIV_ROOT/$ds_id"
    local work_dir="$WORK_ROOT/${ds_id}_${sub_id}"

    if [[ ! -d "$bids_dir/$sub_id" ]]; then
        echo "  SKIP: missing subject dir: $bids_dir/$sub_id"
        return 1
    fi
    if ! has_min_inputs "$bids_dir" "$sub_id"; then
        echo "  SKIP: missing required inputs (anat T1w or rest bold) for $sub_id"
        return 1
    fi
    if is_done "$ds_id" "$sub_id"; then
        echo "  SKIP: already processed ($ds_id/$sub_id)"
        return 0
    fi

    mkdir -p "$out_dir" "$work_dir"
    rm -rf "$staging_dir"
    mkdir -p "$staging_dir"

    # Build a minimal clean BIDS input per subject.
    # This avoids parsing unrelated derivatives JSON files in dataset root.
    if [[ -f "$bids_dir/dataset_description.json" ]]; then
        cp "$bids_dir/dataset_description.json" "$staging_dir/dataset_description.json"
    else
        echo '{"Name":"Keane Dataset","BIDSVersion":"1.3.0","DatasetType":"raw"}' > "$staging_dir/dataset_description.json"
    fi
    [[ -f "$bids_dir/participants.tsv" ]] && cp "$bids_dir/participants.tsv" "$staging_dir/participants.tsv"
    [[ -f "$bids_dir/participants.json" ]] && cp "$bids_dir/participants.json" "$staging_dir/participants.json"

    # Build minimal subject tree for REST-only fMRIPrep.
    # Avoid copying unrelated task files to reduce I/O/storage.
    local stage_sub="$staging_dir/$sub_id"
    mkdir -p "$stage_sub/anat" "$stage_sub/func"
    [[ -d "$bids_dir/$sub_id/fmap" ]] && mkdir -p "$stage_sub/fmap"

    local anat_files=()
    local func_files=()
    local fmap_files=()
    shopt -s nullglob
    anat_files+=("$bids_dir/$sub_id"/anat/*T1w*.nii* "$bids_dir/$sub_id"/anat/*T1w*.json)
    anat_files+=("$bids_dir/$sub_id"/anat/*T2w*.nii* "$bids_dir/$sub_id"/anat/*T2w*.json)
    func_files+=("$bids_dir/$sub_id"/func/*task-rest*bold.nii* "$bids_dir/$sub_id"/func/*task-rest*bold.json)
    func_files+=("$bids_dir/$sub_id"/func/*task-rest*sbref.nii* "$bids_dir/$sub_id"/func/*task-rest*sbref.json)
    fmap_files+=("$bids_dir/$sub_id"/fmap/*.nii* "$bids_dir/$sub_id"/fmap/*.json)
    shopt -u nullglob

    for f in "${anat_files[@]}"; do
        cp -L "$f" "$stage_sub/anat/"
    done
    for f in "${func_files[@]}"; do
        cp -L "$f" "$stage_sub/func/"
    done
    for f in "${fmap_files[@]}"; do
        cp -L "$f" "$stage_sub/fmap/"
    done

    local cmd=(
        docker run --rm
        --user "$(id -u):$(id -g)"
        -v "$staging_dir":/data:ro
        -v "$out_dir":/out
        -v "$work_dir":/work
        -v "$FS_LICENSE":/opt/freesurfer/license.txt:ro
        "nipreps/fmriprep:${FMRIPREP_VERSION}"
        /data /out participant
        --participant-label "${sub_id#sub-}"
        --task-id rest
        --output-spaces "$OUTPUT_SPACES"
        --nprocs "$N_CPUS"
        --mem-mb "$MEM_MB"
        --work-dir /work
        --skip-bids-validation
        --notrack
        --fs-no-reconall
        --ignore slicetiming
    )

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  [DRY-RUN] ${cmd[*]}"
        return 0
    fi

    local log_file="$out_dir/${sub_id}_fmriprep.log"
    echo "  RUN: $ds_id/$sub_id (log: $log_file)"
    if "${cmd[@]}" >"$log_file" 2>&1; then
        echo "  OK : $ds_id/$sub_id"
        return 0
    else
        echo "  FAIL: $ds_id/$sub_id (see $log_file)"
        echo "  ----- log tail (last 40) -----"
        tail -n 40 "$log_file" | sed 's/^/    | /' || true
        return 1
    fi
}

collect_subjects() {
    local bids_dir="$1"
    if [[ ${#SUBJECTS[@]} -gt 0 ]]; then
        printf "%s\n" "${SUBJECTS[@]}"
        return 0
    fi
    for sub_dir in "$bids_dir"/sub-*; do
        [[ -d "$sub_dir" ]] || continue
        basename "$sub_dir"
    done
}

echo "============================================================"
echo " fMRIPrep Keane Runner"
echo " Dataset(s): ${DATASETS[*]}"
echo " fMRIPrep : ${FMRIPREP_VERSION}"
echo " CPUs/MEM : ${N_CPUS} / ${MEM_MB}MB"
echo " Out root : ${DERIV_ROOT}"
[[ "$DRY_RUN" -eq 1 ]] && echo " ** DRY RUN **"
echo "============================================================"

TOTAL=0
OK=0
FAIL=0
SKIP=0

for ds_id in "${DATASETS[@]}"; do
    bids_dir="$DATA_ROOT/$ds_id"
    if [[ ! -d "$bids_dir" ]]; then
        echo "WARN: dataset directory missing: $bids_dir"
        continue
    fi
    echo ""
    echo "[Dataset: $ds_id]"
    mapfile -t ds_subjects < <(collect_subjects "$bids_dir")
    echo "Candidates: ${#ds_subjects[@]}"
    for sub_id in "${ds_subjects[@]}"; do
        TOTAL=$((TOTAL + 1))
        echo "- $sub_id"
        if output="$(run_one "$ds_id" "$sub_id" 2>&1)"; then
            echo "$output"
            if echo "$output" | grep -q "SKIP"; then
                SKIP=$((SKIP + 1))
            else
                OK=$((OK + 1))
            fi
        else
            echo "$output"
            FAIL=$((FAIL + 1))
        fi
    done
done

echo ""
echo "============================================================"
echo " Summary"
echo "   Total: $TOTAL"
echo "   OK   : $OK"
echo "   SKIP : $SKIP"
echo "   FAIL : $FAIL"
echo "============================================================"

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
