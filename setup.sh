#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Constants
# -----------------------
WORKSPACE_ROOT="/workspace"
REPO_URL="https://github.com/loveiscomplicated/CTMP_GIN.git"
REPO_DIR="${WORKSPACE_ROOT}/CTMP_GIN"
BRANCH="runpod"

CONDA_DIR="$HOME/miniconda3"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
ENV_NAME="pyg_2"

RUNS_DIR="${REPO_DIR}/runs"
DATA_DIR="${REPO_DIR}/src/data/raw"
GDOWN_FILE_ID="1T1oYAsdYDcdqUckd7CBzBWj9RnwGrEZg"

# rclone upload
RCLONE_REMOTE="gdrive"
RCLONE_DEST_DIR="CTMP_GIN_runs"
UPLOAD_RETRIES=3

# notifier
SEND_MESSAGE_PY="${REPO_DIR}/src/utils/send_message.py"
BOT_NAME="runpod_setup"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

notify() {
  local msg="$1"
  if [[ -f "$SEND_MESSAGE_PY" ]]; then
    python "$SEND_MESSAGE_PY" "$msg" "$BOT_NAME" || true
  else
    echo "[$(ts)] send_message.py not found: $SEND_MESSAGE_PY"
  fi
}

hold_forever() {
  echo "[$(ts)] holding forever..."
  while true; do sleep 3600; done
}

# -----------------------
# Environment Info
# -----------------------
echo "[$(ts)] RUNPOD_POD_ID='${RUNPOD_POD_ID:-}'"
if command -v runpodctl >/dev/null 2>&1; then
  echo "[$(ts)] runpodctl: $(command -v runpodctl)"
  runpodctl --version || true
else
  echo "[$(ts)] runpodctl not found"
fi

# -----------------------
# System deps
# -----------------------
apt update
apt install -y tmux rclone git wget

# tmux mouse
echo "set -g mouse on" >> ~/.tmux.conf || true
# Note: tmux source-file might fail if not inside a session, ignore error
tmux source-file ~/.tmux.conf || true

# -----------------------
# Repo setup
# -----------------------
mkdir -p "$WORKSPACE_ROOT"
cd "$WORKSPACE_ROOT"
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[$(ts)] repo exists -> update"
  cd "$REPO_DIR"
  git fetch --all
else
  echo "[$(ts)] cloning repo"
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

git checkout "$BRANCH"
git pull origin "$BRANCH"

# -----------------------
# Miniconda + env
# -----------------------
cd "$WORKSPACE_ROOT"
if [[ ! -d "$CONDA_DIR" ]]; then
  echo "[$(ts)] installing miniconda"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR"
else
  echo "[$(ts)] miniconda exists -> skip"
fi

# Load conda
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
else
    echo "[$(ts)] Error: $CONDA_SH not found"
    exit 1
fi

# ----------------------------------
# Accept Anaconda ToS (non-interactive fix)
# ----------------------------------
conda activate base || true

echo "[$(ts)] conda: $(command -v conda)"
conda --version

# ToS accept
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

conda config --set channels defaults || true
conda config --set channel_priority flexible || true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[$(ts)] conda env $ENV_NAME exists -> skip create"
else
  echo "[$(ts)] creating conda env $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.12
fi

# -----------------------
# Python deps (표준 안정화 버전)
# -----------------------
conda activate "$ENV_NAME"

# 1. pip 업그레이드
python -m pip install -U pip

# CUDA 버전 자동 감지
CUDA_RAW=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/' || echo "")
if [[ -z "$CUDA_RAW" ]]; then
  CUDA_RAW=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/' || echo "")
fi

case "$CUDA_RAW" in
  128|129) CUDA_TAG="cu128" ;;
  126|127) CUDA_TAG="cu126" ;;
  *)       CUDA_TAG="cu124" ;;
esac

echo "[$(ts)] Detected CUDA tag: $CUDA_TAG (raw: ${CUDA_RAW:-unknown})"
pip3 install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"


# 4. PyG 본체 및 sparse kernel 패키지 설치
pip install torch-geometric
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install torch-scatter torch-sparse torch-cluster \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"

cd "$REPO_DIR"
pip install -r requirements.txt
pip install requests gdown

echo "[$(ts)] Environment setup complete with Stable PyTorch."
# -----------------------
# Data download
# -----------------------
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
gdown "$GDOWN_FILE_ID"

echo "[$(ts)] Setup complete."
