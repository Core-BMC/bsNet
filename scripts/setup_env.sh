#!/usr/bin/env bash
# =============================================================================
# BS-NET 전처리 환경 설치 스크립트
# 실행: bash scripts/setup_env.sh
# =============================================================================
set -euo pipefail

echo "=== BS-NET Preprocessing Environment Setup ==="

# 1. Conda 환경 생성 (권장)
ENV_NAME="bsnet"

if command -v conda &>/dev/null; then
    echo "[1/4] Creating conda environment '${ENV_NAME}' ..."
    conda create -n "${ENV_NAME}" python=3.10 -y
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
else
    echo "[1/4] conda not found — using system Python (pip only)"
fi

# 2. Core dependencies
echo "[2/4] Installing core packages ..."
pip install --upgrade pip
pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    nibabel \
    nilearn \
    antspyx \
    tqdm

# 3. BS-NET project dependencies
echo "[3/4] Installing BS-NET project dependencies ..."
pip install \
    ruff \
    pytest

# 4. Schaefer atlas 사전 다운로드
echo "[4/4] Pre-fetching Schaefer 2018 atlas (100 parcels, 7 networks) ..."
python -c "
from nilearn.datasets import fetch_atlas_schaefer_2018
atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
print(f'Atlas downloaded: {atlas.maps}')
print(f'Labels: {len(atlas.labels)} regions')
"

echo ""
echo "=== Setup complete ==="
echo "Usage:"
echo "  conda activate ${ENV_NAME}"
echo "  python scripts/preprocess_bold.py --test        # 1명 테스트"
echo "  python scripts/preprocess_bold.py --all         # 100명 전체"
echo "  python scripts/run_real_figure4.py              # 분석 + Figure 생성"
