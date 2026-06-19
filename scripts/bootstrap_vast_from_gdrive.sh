#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: bash scripts/bootstrap_vast_from_gdrive.sh <teds_file_id> <ce_run_file_id> <focal_run_file_id> [missing_corrected_file_id]"
  echo "Example: bash scripts/bootstrap_vast_from_gdrive.sh <TEDS_ID> <CE_RUN_ID> <FOCAL_RUN_ID>"
  exit 1
fi

TEDS_FILE_ID="$1"
CE_RUN_FILE_ID="$2"
FOCAL_RUN_FILE_ID="$3"
MISSING_CORRECTED_FILE_ID="${4:-}"

WORKSPACE_ROOT="/workspace"
REPO_URL="https://github.com/loveiscomplicated/CTMP_GIN.git"
REPO_DIR="${WORKSPACE_ROOT}/CTMP_GIN"
BRANCH="vastai"

CONDA_DIR="$HOME/miniconda3"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
ENV_NAME="pyg_2"

RAW_DATA_DIR="${REPO_DIR}/src/data/raw"
RUNS_DIR="${REPO_DIR}/runs"
DOWNLOAD_DIR="${WORKSPACE_ROOT}/downloads"

CE_RUN_ID="20260508-100129__los_ce_predictor__bs=1024__lr=1.00e-03__seed=1"
FOCAL_RUN_ID="20260508-152249__los_coarse_focal_sqrt_alpha_g1_predictor__bs=1024__lr=1.00e-03__seed=1"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

download_drive_file() {
  local file_id="$1"
  local output_path="$2"
  if [[ -f "$output_path" ]]; then
    echo "[$(ts)] already exists: $output_path"
    return 0
  fi
  echo "[$(ts)] downloading Google Drive file -> $output_path"
  gdown "https://drive.google.com/uc?id=${file_id}" -O "$output_path"
}

extract_run_archive() {
  local archive_path="$1"
  local expected_run_id="$2"
  if [[ -d "${RUNS_DIR}/${expected_run_id}" ]]; then
    echo "[$(ts)] run already extracted: ${RUNS_DIR}/${expected_run_id}"
    return 0
  fi
  echo "[$(ts)] extracting ${archive_path} -> ${RUNS_DIR}"
  tar -xzf "$archive_path" -C "$RUNS_DIR"
}

mkdir -p "$WORKSPACE_ROOT" "$DOWNLOAD_DIR"
cd "$WORKSPACE_ROOT"

apt update
apt install -y git wget

if [[ -d "${REPO_DIR}/.git" ]]; then
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

if [[ ! -d "$CONDA_DIR" ]]; then
  echo "[$(ts)] installing miniconda"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR"
fi

source "$CONDA_SH"
conda activate base || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.12 pip
fi

conda activate "$ENV_NAME"
python -m pip install -U pip

CUDA_RAW=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/' || echo "")
if [[ -z "$CUDA_RAW" ]]; then
  CUDA_RAW=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/' || echo "")
fi

case "$CUDA_RAW" in
  128|129) CUDA_TAG="cu128" ;;
  126|127) CUDA_TAG="cu126" ;;
  *) CUDA_TAG="cu124" ;;
esac

echo "[$(ts)] detected CUDA tag: $CUDA_TAG"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
pip install torch-geometric
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install torch-scatter torch-sparse torch-cluster \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"

cd "$REPO_DIR"
pip install -r requirements.txt
pip install requests gdown

mkdir -p "$RAW_DATA_DIR" "$RUNS_DIR"

download_drive_file "$TEDS_FILE_ID" "${DOWNLOAD_DIR}/TEDS_Discharge.csv"
cp "${DOWNLOAD_DIR}/TEDS_Discharge.csv" "${RAW_DATA_DIR}/TEDS_Discharge.csv"

if [[ -n "$MISSING_CORRECTED_FILE_ID" ]]; then
  download_drive_file "$MISSING_CORRECTED_FILE_ID" "${DOWNLOAD_DIR}/missing_corrected.csv"
  cp "${DOWNLOAD_DIR}/missing_corrected.csv" "${RAW_DATA_DIR}/missing_corrected.csv"
fi

download_drive_file "$CE_RUN_FILE_ID" "${DOWNLOAD_DIR}/${CE_RUN_ID}.tgz"
download_drive_file "$FOCAL_RUN_FILE_ID" "${DOWNLOAD_DIR}/${FOCAL_RUN_ID}.tgz"

extract_run_archive "${DOWNLOAD_DIR}/${CE_RUN_ID}.tgz" "$CE_RUN_ID"
extract_run_archive "${DOWNLOAD_DIR}/${FOCAL_RUN_ID}.tgz" "$FOCAL_RUN_ID"

echo "[$(ts)] bootstrap complete"
echo "[$(ts)] repo    : $REPO_DIR"
echo "[$(ts)] data    : ${RAW_DATA_DIR}/TEDS_Discharge.csv"
echo "[$(ts)] CE ckpt : ${RUNS_DIR}/${CE_RUN_ID}/checkpoints/best.pt"
echo "[$(ts)] focal   : ${RUNS_DIR}/${FOCAL_RUN_ID}/checkpoints/best.pt"
echo "[$(ts)] next    : source ${CONDA_SH} && conda activate ${ENV_NAME}"
