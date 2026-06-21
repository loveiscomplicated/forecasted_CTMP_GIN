#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Args
# -----------------------
if [[ $# -lt 3 ]]; then
  echo "Usage: bash run_vast.sh <model_name> <config_path> <seed>"
  echo "Example: bash run_vast.sh gin configs/gin.yaml 1"
  exit 1
fi

MODEL_NAME="$1"
CONFIG_PATH="$2"
SEED="$3"

echo "model_name: ${MODEL_NAME}"
echo "config    : ${CONFIG_PATH}"
echo "seed      : ${SEED}"

# -----------------------
# Constants
# -----------------------
WORKSPACE_ROOT="/workspace"
REPO_URL="https://github.com/loveiscomplicated/forecasted_CTMP_GIN.git"
REPO_DIR="${WORKSPACE_ROOT}/forecasted_CTMP_GIN"
BRANCH="main"

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
BOT_NAME="Vast_main_py_$MODEL_NAME"

# vast termination helper
VAST_TERMINATE_SH="${REPO_DIR}/scripts/vast_terminate.sh"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# -----------------------
# Build pipeline to run INSIDE tmux
# -----------------------
PIPELINE="$(cat <<'BASH'
set -euo pipefail
ts() { date '+%Y-%m-%d %H:%M:%S'; }

MODEL_NAME="__MODEL_NAME__"
CONFIG_PATH="__CONFIG_PATH__"
SEED="__SEED__"

WORKSPACE_ROOT="__WORKSPACE_ROOT__"
REPO_URL="__REPO_URL__"
REPO_DIR="__REPO_DIR__"
BRANCH="__BRANCH__"

CONDA_DIR="__CONDA_DIR__"
CONDA_SH="__CONDA_SH__"
ENV_NAME="__ENV_NAME__"

RUNS_DIR="__RUNS_DIR__"
DATA_DIR="__DATA_DIR__"
GDOWN_FILE_ID="__GDOWN_FILE_ID__"

RCLONE_REMOTE="__RCLONE_REMOTE__"
RCLONE_DEST_DIR="__RCLONE_DEST_DIR__"
UPLOAD_RETRIES="__UPLOAD_RETRIES__"

SEND_MESSAGE_PY="__SEND_MESSAGE_PY__"
BOT_NAME="__BOT_NAME__"
VAST_TERMINATE_SH="__VAST_TERMINATE_SH__"

export DISCORD_WEBHOOK_URL="__DISCORD_WEBHOOK_URL__"
export RCLONE_CONF_B64="__RCLONE_CONF_B64__"
export CONTAINER_API_KEY="__CONTAINER_API_KEY__"
export VAST_INSTANCE_ID="__VAST_INSTANCE_ID__"

# -----------------------
# Log file (stdout + stderr 모두 파일로 저장)
# -----------------------
LOG_FILE="${WORKSPACE_ROOT}/train_${MODEL_NAME}_seed${SEED}.log"
mkdir -p "$WORKSPACE_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

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

echo "[$(ts)] ===== pipeline start ====="
echo "[$(ts)] model_name: $MODEL_NAME"
echo "[$(ts)] config    : $CONFIG_PATH"
echo "[$(ts)] seed      : $SEED"

# Vast.ai environment diagnostics
echo "[$(ts)] VAST_CONTAINERLABEL='${VAST_CONTAINERLABEL:-}'"
echo "[$(ts)] VAST_INSTANCE_ID='${VAST_INSTANCE_ID:-}'"
if command -v vastai >/dev/null 2>&1; then
  echo "[$(ts)] vastai: $(command -v vastai)"
  vastai --version || true
else
  echo "[$(ts)] vastai CLI not found (will be installed at termination time)"
fi

# -----------------------
# System deps
# -----------------------
apt update
apt install -y tmux rclone git wget python3-pip

# tmux mouse
echo "set -g mouse on" >> ~/.tmux.conf || true
tmux source-file ~/.tmux.conf || true

# -----------------------
# Repo setup
# -----------------------
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

source "$CONDA_SH"

# ----------------------------------
# Accept Anaconda ToS (non-interactive fix)
# ----------------------------------
conda activate base || true

echo "[$(ts)] conda: $(command -v conda)"
conda --version

conda activate base || true

echo "[$(ts)] conda: $(type -a conda | head -n 2)"
conda --version

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

conda config --set channels defaults || true
conda config --set channel_priority flexible || true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[$(ts)] conda env $ENV_NAME exists -> skip create"
else
  echo "[$(ts)] creating conda env $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.12 pip
fi

conda activate "$ENV_NAME"
python -m pip --version >/dev/null 2>&1 || conda install -y -n "$ENV_NAME" pip

# -----------------------
# Python deps (your order)
# -----------------------
python -m pip install -U pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
cd "$REPO_DIR"
pip install -r requirements.txt
pip install requests

# -----------------------
# Data download
# -----------------------
cd "$DATA_DIR"
gdown "$GDOWN_FILE_ID"

# -----------------------
# Training
# -----------------------
cd "$REPO_DIR"
echo "[$(ts)] training start"
set +e
python -m src.main --config "$CONFIG_PATH" --seed "$SEED"
TRAIN_RC=$?
set -e

if [[ $TRAIN_RC -eq 0 ]]; then
  notify "[SUCCESS] Training completed. model=$MODEL_NAME config=$CONFIG_PATH seed=$SEED"
else
  notify "[FAIL] Training failed (rc=$TRAIN_RC). model=$MODEL_NAME config=$CONFIG_PATH seed=$SEED"
fi

# -----------------------
# Upload runs (always try) + retry policy C
#   - upload fails => notify + HOLD (no shutdown)
# -----------------------
mkdir -p /root/.config/rclone
if [[ -z "${RCLONE_CONF_B64:-}" ]]; then
  notify "[UPLOAD_FAIL] RCLONE_CONF_B64 not set. Holding without shutdown."
  hold_forever
fi

echo "$RCLONE_CONF_B64" | base64 -d > /root/.config/rclone/rclone.conf

attempt=1
ok=0
while [[ $attempt -le $UPLOAD_RETRIES ]]; do
  echo "[$(ts)] upload attempt $attempt/$UPLOAD_RETRIES ..."
  if rclone copy "$RUNS_DIR" "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}" \
      --create-empty-src-dirs \
      --transfers 8 \
      --checkers 16 \
      --retries 3 \
      --low-level-retries 10 \
      --stats 10s
  then
    ok=1
    break
  fi
  attempt=$((attempt+1))
  sleep 5
done

if [[ $ok -eq 1 ]]; then
  rclone copy "$LOG_FILE" "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/logs/" \
      --retries 3 --low-level-retries 5 || true
  notify "[SUCCESS] Upload completed: ${RCLONE_REMOTE}:${RCLONE_DEST_DIR}"
  echo "[$(ts)] shutting down..."
  # -----------------------
  # Stop instance (Vast.ai-native)
  # -----------------------
  export SEND_MESSAGE_PY BOT_NAME
  if [[ -f "$VAST_TERMINATE_SH" ]]; then
    bash "$VAST_TERMINATE_SH"
  else
    notify "[TERMINATE_SKIP] vast_terminate.sh not found at $VAST_TERMINATE_SH. Holding."
    hold_forever
  fi
  # If vast_terminate.sh returns here, instance did not stop; hold as a safety net.
  hold_forever
else
  rclone copy "$LOG_FILE" "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/logs/" \
      --retries 3 --low-level-retries 5 || true
  notify "[UPLOAD_FAIL] Upload failed after ${UPLOAD_RETRIES} attempts. Holding without shutdown."
  hold_forever
fi
BASH
)"

# Fill placeholders
PIPELINE="${PIPELINE//__MODEL_NAME__/${MODEL_NAME}}"
PIPELINE="${PIPELINE//__CONFIG_PATH__/${CONFIG_PATH}}"
PIPELINE="${PIPELINE//__SEED__/${SEED}}"
PIPELINE="${PIPELINE//__WORKSPACE_ROOT__/${WORKSPACE_ROOT}}"
PIPELINE="${PIPELINE//__REPO_URL__/${REPO_URL}}"
PIPELINE="${PIPELINE//__REPO_DIR__/${REPO_DIR}}"
PIPELINE="${PIPELINE//__BRANCH__/${BRANCH}}"
PIPELINE="${PIPELINE//__CONDA_DIR__/${CONDA_DIR}}"
PIPELINE="${PIPELINE//__CONDA_SH__/${CONDA_SH}}"
PIPELINE="${PIPELINE//__ENV_NAME__/${ENV_NAME}}"
PIPELINE="${PIPELINE//__RUNS_DIR__/${RUNS_DIR}}"
PIPELINE="${PIPELINE//__DATA_DIR__/${DATA_DIR}}"
PIPELINE="${PIPELINE//__GDOWN_FILE_ID__/${GDOWN_FILE_ID}}"
PIPELINE="${PIPELINE//__RCLONE_REMOTE__/${RCLONE_REMOTE}}"
PIPELINE="${PIPELINE//__RCLONE_DEST_DIR__/${RCLONE_DEST_DIR}}"
PIPELINE="${PIPELINE//__UPLOAD_RETRIES__/${UPLOAD_RETRIES}}"
PIPELINE="${PIPELINE//__SEND_MESSAGE_PY__/${SEND_MESSAGE_PY}}"
PIPELINE="${PIPELINE//__BOT_NAME__/${BOT_NAME}}"
PIPELINE="${PIPELINE//__VAST_TERMINATE_SH__/${VAST_TERMINATE_SH}}"
PIPELINE="${PIPELINE//__DISCORD_WEBHOOK_URL__/${DISCORD_WEBHOOK_URL:-}}"
PIPELINE="${PIPELINE//__RCLONE_CONF_B64__/${RCLONE_CONF_B64:-}}"
PIPELINE="${PIPELINE//__CONTAINER_API_KEY__/${CONTAINER_API_KEY:-}}"
PIPELINE="${PIPELINE//__VAST_INSTANCE_ID__/${VAST_INSTANCE_ID:-}}"

# -----------------------
# tmux session: create and start
# -----------------------
apt update
apt install -y tmux

SESSION_NAME="${MODEL_NAME}_seed${SEED}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[$(ts)] tmux session exists: ${SESSION_NAME}"
else
  echo "[$(ts)] creating tmux session: ${SESSION_NAME}"
  tmux new-session -d -s "${SESSION_NAME}"
fi

PIPE_PATH="/tmp/${MODEL_NAME}__pipeline.sh"
printf "%s" "$PIPELINE" > "$PIPE_PATH"
chmod +x "$PIPE_PATH"

tmux set-environment -t "${SESSION_NAME}" RCLONE_CONF_B64 "${RCLONE_CONF_B64:-}"
tmux set-environment -t "${SESSION_NAME}" DISCORD_WEBHOOK_URL "${DISCORD_WEBHOOK_URL:-}"
tmux set-environment -t "${SESSION_NAME}" CONTAINER_API_KEY "${CONTAINER_API_KEY:-}"
tmux set-environment -t "${SESSION_NAME}" VAST_INSTANCE_ID "${VAST_INSTANCE_ID:-}"
tmux send-keys -t "${SESSION_NAME}" "bash $PIPE_PATH" C-m

echo "[$(ts)] started in tmux session '${SESSION_NAME}'."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
