#!/usr/bin/env bash
# ============================================================================
# Setup Keane datasets via DataLad clone (ds003404, ds005073)
# ============================================================================
# 기능:
#   - OpenNeuro DataLad 리포지토리를 data/ 하위에 clone
#   - 기존 폴더를 inactive로 이동(옵션)
#
# 사용 예:
#   # ds005073만 설정, 기존 폴더는 inactive로 이동
#   bash src/scripts/setup_keane_datalad.sh --dataset ds005073 --inactive-existing
#
#   # 둘 다 설정
#   bash src/scripts/setup_keane_datalad.sh --dataset all --inactive-existing
#
#   # dry-run
#   bash src/scripts/setup_keane_datalad.sh --dataset all --inactive-existing --dry-run
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="$PROJECT_ROOT/data"

DATASET="all"   # ds003404|ds005073|all
INACTIVE_EXISTING=0
DRY_RUN=0

log()  { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
err()  { echo "[ERR ] $*" >&2; }

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] $*"
        return 0
    fi
    "$@"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --inactive-existing) INACTIVE_EXISTING=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

case "$DATASET" in
    ds003404|ds005073|all) ;;
    *) err "--dataset must be ds003404|ds005073|all"; exit 1 ;;
esac

command -v datalad >/dev/null 2>&1 || { err "datalad not found"; exit 1; }
mkdir -p "$DATA_ROOT"

declare -a TARGETS
if [[ "$DATASET" == "all" ]]; then
    TARGETS=("ds003404" "ds005073")
else
    TARGETS=("$DATASET")
fi

clone_one() {
    local ds="$1"
    local target="$DATA_ROOT/$ds"
    local url="https://github.com/OpenNeuroDatasets/${ds}.git"

    if [[ -d "$target/.datalad" ]]; then
        log "$ds already looks like a datalad dataset: $target (skip)"
        return 0
    fi

    if [[ -e "$target" ]]; then
        if [[ "$INACTIVE_EXISTING" -eq 1 ]]; then
            local ts
            ts="$(date +%Y%m%d_%H%M%S)"
            local backup="${target}_inactive_${ts}"
            warn "$ds exists but is not a datalad clone. Moving to: $backup"
            run_cmd mv "$target" "$backup"
        else
            err "$ds exists and is not a datalad clone: $target"
            err "Use --inactive-existing to move it aside automatically."
            return 1
        fi
    fi

    log "Cloning $ds from $url -> $target"
    run_cmd datalad clone "$url" "$target"
}

echo "============================================================"
echo " Setup Keane DataLad Datasets"
echo " Dataset(s)        : ${TARGETS[*]}"
echo " Data root         : $DATA_ROOT"
echo " Inactive existing : $INACTIVE_EXISTING"
[[ "$DRY_RUN" -eq 1 ]] && echo " ** DRY RUN **"
echo "============================================================"

FAIL=0
for ds in "${TARGETS[@]}"; do
    echo ""
    echo "---- [$ds] ----"
    if ! clone_one "$ds"; then
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "============================================================"
if [[ "$FAIL" -eq 0 ]]; then
    echo " Setup completed successfully."
else
    echo " Setup finished with failures: $FAIL"
fi
echo "============================================================"

[[ "$FAIL" -eq 0 ]]
