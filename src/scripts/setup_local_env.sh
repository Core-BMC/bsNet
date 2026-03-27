#!/usr/bin/env bash
# ============================================================================
# BS-NET Local Environment Setup
# ============================================================================
# 이 스크립트를 로컬 머신(또는 연구실 서버)에서 실행하세요.
#
# 사전 요구사항:
#   - Python 3.9+
#   - conda 또는 python3 venv
#   - AWS CLI (optional, for faster OpenNeuro download)
#
# 사용법:
#   chmod +x src/scripts/setup_local_env.sh
#   ./src/scripts/setup_local_env.sh          # conda 우선, 없으면 venv
#   ./src/scripts/setup_local_env.sh --venv   # venv 강제 사용
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
ATLAS_DIR="$DATA_DIR/atlas"
DERIV_DIR="$DATA_DIR/derivatives"
VENV_DIR="$PROJECT_ROOT/.venv"
USE_VENV="${1:-}"

echo "============================================"
echo " BS-NET Local Environment Setup"
echo " Project root: $PROJECT_ROOT"
echo "============================================"

# ---- Step 1: Create Python environment (conda or venv) ----
echo ""
echo "[Step 1/5] Setting up Python environment..."

if [ "$USE_VENV" = "--venv" ]; then
    # Force venv mode
    echo "  Using venv (--venv flag)..."
    if [ -d "$VENV_DIR" ]; then
        echo "  venv already exists: $VENV_DIR"
    else
        echo "  Creating venv at $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
elif command -v conda &>/dev/null; then
    # conda available
    if conda env list | grep -q "bsnet"; then
        echo "  conda env 'bsnet' already exists. Activating..."
    else
        echo "  Creating conda env 'bsnet' (Python 3.10)..."
        conda create -n bsnet python=3.10 -y
    fi
    eval "$(conda shell.bash hook)"
    conda activate bsnet
else
    # Fallback to venv
    echo "  conda not found. Using venv instead..."
    if [ -d "$VENV_DIR" ]; then
        echo "  venv already exists: $VENV_DIR"
    else
        echo "  Creating venv at $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

echo "  Python: $(python3 --version) at $(which python3)"

# ---- Step 2: Install Python packages ----
echo ""
echo "[Step 2/5] Installing Python packages..."
pip install -q \
    numpy scipy pandas matplotlib seaborn \
    nibabel nilearn dipy scikit-learn \
    templateflow openneuro-py \
    ruff pytest

echo "  Installed packages:"
python -c "
import nilearn; print(f'  nilearn    {nilearn.__version__}')
import dipy;    print(f'  dipy       {dipy.__version__}')
import nibabel; print(f'  nibabel    {nibabel.__version__}')
"

# ---- Step 3: Download Schaefer 100 Atlas ----
echo ""
echo "[Step 3/5] Downloading Schaefer 2018 Atlas (100 parcels, 7 networks)..."
mkdir -p "$ATLAS_DIR"

ATLAS_FILE="$ATLAS_DIR/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
if [ -f "$ATLAS_FILE" ]; then
    echo "  Atlas already exists: $ATLAS_FILE"
else
    # Method 1: templateflow (preferred)
    python -c "
from templateflow import api as tflow
import shutil
path = tflow.get('MNI152NLin6Asym', atlas='Schaefer2018',
                 desc='100Parcels7Networks', resolution=2,
                 suffix='dseg', extension='.nii.gz')
shutil.copy2(str(path), '$ATLAS_FILE')
print(f'  Downloaded via templateflow: $ATLAS_FILE')
" 2>/dev/null || {
        # Method 2: direct GitHub download
        echo "  templateflow failed, trying GitHub..."
        curl -L -o "$ATLAS_FILE" \
            "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
        echo "  Downloaded from GitHub: $ATLAS_FILE"
    }
fi

# Verify atlas
python -c "
import nibabel as nib, numpy as np
img = nib.load('$ATLAS_FILE')
n = len(np.unique(img.get_fdata())) - 1
print(f'  Atlas verified: {img.shape}, {n} parcels')
"

# ---- Step 4: Download OpenNeuro ds000030 (anat + func) ----
echo ""
echo "[Step 4/5] Downloading OpenNeuro ds000030 (healthy controls, anat + rest)..."
mkdir -p "$DATA_DIR/openneuro"

# Get participant list for healthy controls with rest + T1w
# Auto-detect: openneuro-py may nest under 1.0.0/uncompressed/ or place directly
if [ -f "$DATA_DIR/openneuro/ds000030/1.0.0/uncompressed/participants.tsv" ]; then
    NEUR_ROOT="$DATA_DIR/openneuro/ds000030/1.0.0/uncompressed"
elif [ -f "$DATA_DIR/openneuro/ds000030/participants.tsv" ]; then
    NEUR_ROOT="$DATA_DIR/openneuro/ds000030"
else
    NEUR_ROOT="$DATA_DIR/openneuro/ds000030"
fi
PARTICIPANTS_FILE="$NEUR_ROOT/participants.tsv"
DS_DIR="$DATA_DIR/openneuro/ds000030"
mkdir -p "$DS_DIR"

if [ ! -f "$PARTICIPANTS_FILE" ]; then
    echo "  Downloading dataset metadata first..."
    openneuro-py download --dataset ds000030 \
        --target-dir "$DS_DIR" \
        --include "participants.tsv" \
        --include "dataset_description.json"
fi

# Download anat + rest func for first HC subject (test)
echo ""
echo "  Downloading 1 HC subject (sub-10159) for testing..."
SUB="sub-10159"
SUBDIR="$DS_DIR/$SUB"

# Check if anat already exists
if [ -f "$SUBDIR/anat/${SUB}_T1w.nii.gz" ] && [ -f "$SUBDIR/func/${SUB}_task-rest_bold.nii.gz" ]; then
    echo "  $SUB already has anat + func"
else
    openneuro-py download --dataset ds000030 \
        --target-dir "$DS_DIR" \
        --include "$SUB"
    echo "  Downloaded: $SUB"
fi

# Verify
echo ""
echo "  Verifying $SUB data:"
python -c "
import nibabel as nib
import os

sub_dir = '$SUBDIR'
# T1w
t1_path = os.path.join(sub_dir, 'anat', '${SUB}_T1w.nii.gz')
if os.path.exists(t1_path):
    t1 = nib.load(t1_path)
    print(f'  T1w: {t1.shape}, voxel={list(t1.header.get_zooms()[:3])}')
else:
    print(f'  T1w: NOT FOUND at {t1_path}')

# BOLD
bold_path = os.path.join(sub_dir, 'func', '${SUB}_task-rest_bold.nii.gz')
if os.path.exists(bold_path):
    bold = nib.load(bold_path)
    print(f'  BOLD: {bold.shape}, TR={bold.header.get_zooms()[-1]}s, {bold.shape[-1]} vols')
else:
    print(f'  BOLD: NOT FOUND at {bold_path}')
"

# ---- Step 5: Run 1-subject test ----
echo ""
echo "[Step 5/5] Running 1-subject end-to-end test..."
cd "$PROJECT_ROOT"
python src/scripts/preprocess_real_data.py --subject sub-10159 --verbose

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   # Download all HC subjects:"
echo "   python -m src.scripts.download_openneuro --all-hc"
echo ""
echo "   # Process all subjects:"
echo "   python src/scripts/preprocess_real_data.py --run-all"
echo "============================================"
