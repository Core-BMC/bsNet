#!/usr/bin/env bash
# Setup Diffusion-TS for BS-NET Signal Recovery Experiment
#
# Prerequisites:
#   - NVIDIA GPU (RTX 4090 tested)
#   - CUDA 11.8+ / 12.x
#   - conda or python3.10+
#
# Usage:
#   cd /path/to/bsNet
#   bash src/scripts/setup_diffusion_ts.sh
#
# After setup:
#   1. Prepare data:   PYTHONPATH=. python3 src/scripts/prepare_signal_recovery_data.py --cache-dir data/ds000243/timeseries_cache --atlas harvard_oxford
#   2. Compute weights: PYTHONPATH=. python3 src/scripts/compute_reliability_weights.py --data-dir data/ds000243/signal_recovery/harvard_oxford
#   3. Train model:     cd external/Diffusion-TS && python main.py --config ../../configs/diffusion_ts_fmri_phase1.yaml --mode train
#   4. Run imputation:  python run_signal_recovery.py (see below)
#   5. Evaluate:        cd ../.. && PYTHONPATH=. python3 src/scripts/eval_signal_recovery.py --data-dir data/ds000243/signal_recovery/harvard_oxford

set -euo pipefail

BSNET_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXTERNAL_DIR="${BSNET_ROOT}/external"
DTS_DIR="${EXTERNAL_DIR}/Diffusion-TS"

echo "=== BS-NET Signal Recovery: Diffusion-TS Setup ==="
echo "BS-NET root: ${BSNET_ROOT}"

# ---------------------------------------------------------------
# 1. Clone Diffusion-TS
# ---------------------------------------------------------------
if [ -d "${DTS_DIR}" ]; then
    echo "[SKIP] Diffusion-TS already exists at ${DTS_DIR}"
    cd "${DTS_DIR}" && git pull --ff-only || true
else
    echo "[1/4] Cloning Diffusion-TS (ICLR 2024)..."
    mkdir -p "${EXTERNAL_DIR}"
    git clone https://github.com/Y-debug-sys/Diffusion-TS.git "${DTS_DIR}"
fi

cd "${DTS_DIR}"

# ---------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------
echo "[2/4] Installing Diffusion-TS dependencies..."

# Core dependencies (Diffusion-TS requirements)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
    echo "  PyTorch already installed or install failed — check CUDA version"

pip install -r requirements.txt 2>/dev/null || \
    echo "  Some requirements may need manual resolution"

# Additional dependencies for BS-NET integration
pip install scikit-learn scipy 2>/dev/null || true

echo "  PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# ---------------------------------------------------------------
# 3. Symlink BS-NET data into Diffusion-TS
# ---------------------------------------------------------------
echo "[3/4] Creating data symlinks..."

SR_DATA="${BSNET_ROOT}/data/ds000243/signal_recovery"
if [ -d "${SR_DATA}" ]; then
    ln -sfn "${SR_DATA}" "${DTS_DIR}/data_bsnet"
    echo "  Linked: ${DTS_DIR}/data_bsnet -> ${SR_DATA}"
else
    echo "  [WARN] Signal recovery data not found at ${SR_DATA}"
    echo "         Run prepare_signal_recovery_data.py first."
fi

# ---------------------------------------------------------------
# 4. Copy config
# ---------------------------------------------------------------
echo "[4/4] Copying BS-NET config..."
mkdir -p "${DTS_DIR}/Config"
cp "${BSNET_ROOT}/configs/diffusion_ts_fmri_phase1.yaml" "${DTS_DIR}/Config/" 2>/dev/null || true

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Prepare data (from bsNet root):"
echo "     PYTHONPATH=. python3 src/scripts/prepare_signal_recovery_data.py \\"
echo "         --cache-dir data/ds000243/timeseries_cache \\"
echo "         --atlas harvard_oxford \\"
echo "         --seq-len 180 --short-len 48 --n-windows 20"
echo ""
echo "  2. Compute reliability weights:"
echo "     PYTHONPATH=. python3 src/scripts/compute_reliability_weights.py \\"
echo "         --data-dir data/ds000243/signal_recovery/harvard_oxford"
echo ""
echo "  3. Train Diffusion-TS:"
echo "     cd external/Diffusion-TS"
echo "     python main.py --config Config/diffusion_ts_fmri_phase1.yaml --mode train"
echo ""
echo "  4. Run imputation (Condition A + B):"
echo "     python run_signal_recovery.py  (see src/scripts/run_signal_recovery.py)"
echo ""
echo "  5. Evaluate:"
echo "     cd ${BSNET_ROOT}"
echo "     PYTHONPATH=. python3 src/scripts/eval_signal_recovery.py \\"
echo "         --data-dir data/ds000243/signal_recovery/harvard_oxford \\"
echo "         --imputed-a results/signal_recovery/imputed_naive.npy \\"
echo "         --imputed-b results/signal_recovery/imputed_guided.npy \\"
echo "         --output results/signal_recovery/eval_results.csv"
