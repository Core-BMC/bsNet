#!/usr/bin/env bash
# ============================================================================
# Install DataLad (+ git-annex) on Ubuntu/WSL
# ============================================================================
# 기본 전략:
#   1) apt 설치 (권장): datalad + git-annex
#   2) 실패 시 pip 설치 (대체): datalad
#
# 사용법:
#   bash src/scripts/install_datalad.sh
#   bash src/scripts/install_datalad.sh --yes
#   bash src/scripts/install_datalad.sh --method apt
#   bash src/scripts/install_datalad.sh --method pip
# ============================================================================
set -euo pipefail

METHOD="auto"  # auto|apt|pip
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method) METHOD="$2"; shift 2 ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        -h|--help)
            sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$METHOD" in
    auto|apt|pip) ;;
    *) echo "ERROR: --method must be one of: auto, apt, pip"; exit 1 ;;
esac

log()  { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
err()  { echo "[ERR ] $*" >&2; }

if [[ "$(uname -s)" != "Linux" ]]; then
    err "This script targets Linux/WSL (detected: $(uname -s))."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    err "python3 not found. Install python3 first."
    exit 1
fi

SUDO=""
if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

if [[ "$ASSUME_YES" -eq 1 ]]; then
    APT_YES="-y"
else
    APT_YES=""
fi

verify_install() {
    local ok=0
    if command -v datalad >/dev/null 2>&1; then
        log "datalad: $(datalad --version 2>/dev/null || echo 'installed')"
        ok=1
    fi
    if command -v git-annex >/dev/null 2>&1; then
        log "git-annex: $(git-annex version | head -n 1)"
    else
        warn "git-annex not found (some DataLad dataset operations may be limited)."
    fi
    return "$ok"
}

install_via_apt() {
    if ! command -v apt-get >/dev/null 2>&1; then
        err "apt-get not available."
        return 1
    fi

    log "Installing via apt: datalad git-annex"
    $SUDO apt-get update
    $SUDO apt-get install $APT_YES datalad git-annex ca-certificates
}

install_via_pip() {
    log "Installing via pip (user site): datalad"
    python3 -m pip install --user --upgrade pip
    python3 -m pip install --user datalad

    local user_bin
    user_bin="$(python3 -m site --user-base)/bin"
    if [[ ":$PATH:" != *":$user_bin:"* ]]; then
        warn "Add this to your shell profile (~/.bashrc or ~/.zshrc):"
        echo "export PATH=\"$user_bin:\$PATH\""
    fi
}

log "Install method: $METHOD"

if [[ "$METHOD" == "apt" ]]; then
    install_via_apt
elif [[ "$METHOD" == "pip" ]]; then
    install_via_pip
else
    # auto
    if install_via_apt; then
        :
    else
        warn "apt installation failed. Falling back to pip."
        install_via_pip
    fi
fi

if verify_install; then
    log "DataLad installation completed."
    exit 0
fi

err "datalad command not found after installation."
exit 1
