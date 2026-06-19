#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash run_vast_forecast_gdrive.sh <model_name> <config_path> <seed> <archive_specs>"
  echo "archive_specs format: <run_id>::<file_id>[;<run_id>::<file_id>...]"
  exit 1
fi

MODEL_NAME="$1"
CONFIG_PATH="$2"
SEED="$3"
ARCHIVE_SPECS="$4"

echo "model_name   : ${MODEL_NAME}"
echo "config       : ${CONFIG_PATH}"
echo "seed         : ${SEED}"
echo "archive_specs: ${ARCHIVE_SPECS}"

WORKSPACE_ROOT="/workspace"
REPO_DIR="${WORKSPACE_ROOT}/CTMP_GIN"
RUNS_DIR="${REPO_DIR}/runs"
DOWNLOAD_DIR="${WORKSPACE_ROOT}/downloads"
CONDA_DIR="$HOME/miniconda3"
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
ENV_NAME="pyg_2"
RCLONE_REMOTE="gdrive"
RCLONE_DEST_DIR="CTMP_GIN_runs"
UPLOAD_RETRIES=3
SEND_MESSAGE_PY="${REPO_DIR}/src/utils/send_message.py"
BOT_NAME="vast_${MODEL_NAME}"
VAST_TERMINATE_SH="${REPO_DIR}/scripts/vast_terminate.sh"
RCLONE_B64_FILE="/tmp/rclone_conf.b64"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

for k in {1..120}; do
  if [[ -n "${RCLONE_CONF_B64:-}" ]]; then
    printf "%s" "$RCLONE_CONF_B64" > "$RCLONE_B64_FILE"
    break
  fi
  sleep 1
done

if [[ ! -s "$RCLONE_B64_FILE" ]]; then
  echo "[$(ts)] RCLONE_CONF_B64 still empty after 120s. Exiting."
  exit 1
fi

PIPELINE="$(cat <<'BASH'
set -euo pipefail
ts() { date '+%Y-%m-%d %H:%M:%S'; }

MODEL_NAME="__MODEL_NAME__"
CONFIG_PATH="__CONFIG_PATH__"
SEED="__SEED__"
ARCHIVE_SPECS="__ARCHIVE_SPECS__"
WORKSPACE_ROOT="__WORKSPACE_ROOT__"
REPO_DIR="__REPO_DIR__"
RUNS_DIR="__RUNS_DIR__"
DOWNLOAD_DIR="__DOWNLOAD_DIR__"
CONDA_SH="__CONDA_SH__"
ENV_NAME="__ENV_NAME__"
RCLONE_REMOTE="__RCLONE_REMOTE__"
RCLONE_DEST_DIR="__RCLONE_DEST_DIR__"
UPLOAD_RETRIES="__UPLOAD_RETRIES__"
SEND_MESSAGE_PY="__SEND_MESSAGE_PY__"
BOT_NAME="__BOT_NAME__"
VAST_TERMINATE_SH="__VAST_TERMINATE_SH__"
RCLONE_B64_FILE="__RCLONE_B64_FILE__"

export CONTAINER_API_KEY="__CONTAINER_API_KEY__"
export VAST_INSTANCE_ID="__VAST_INSTANCE_ID__"
export DISCORD_WEBHOOK_URL="__DISCORD_WEBHOOK_URL__"

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

download_and_extract_archive() {
  local run_id="$1"
  local file_id="$2"
  local archive_path="${DOWNLOAD_DIR}/${run_id}.tgz"

  if [[ "$file_id" == __SET_* || -z "$file_id" ]]; then
    notify "[FAIL] Missing Google Drive file id for run=${run_id}. Update launcher constants first."
    hold_forever
  fi

  if [[ ! -d "${RUNS_DIR}/${run_id}" ]]; then
    mkdir -p "$DOWNLOAD_DIR" "$RUNS_DIR"
    if [[ ! -f "$archive_path" ]]; then
      echo "[$(ts)] downloading archive for ${run_id}"
      if ! gdown "https://drive.google.com/uc?id=${file_id}" -O "$archive_path"; then
        notify "[FAIL] gdown download failed for run=${run_id}"
        hold_forever
      fi
    else
      echo "[$(ts)] archive already exists: $archive_path"
    fi

    echo "[$(ts)] extracting archive for ${run_id}"
    if ! tar -xzf "$archive_path" -C "$REPO_DIR"; then
      notify "[FAIL] archive extract failed for run=${run_id}"
      hold_forever
    fi
  else
    echo "[$(ts)] run already present: ${RUNS_DIR}/${run_id}"
  fi
}

bootstrap_forecast_archives() {
  IFS=';' read -r -a entries <<< "$ARCHIVE_SPECS"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" ]] && continue
    run_id="${entry%%::*}"
    file_id="${entry#*::}"
    if [[ -z "$run_id" || "$file_id" == "$entry" ]]; then
      notify "[FAIL] Invalid archive spec entry: $entry"
      hold_forever
    fi
    download_and_extract_archive "$run_id" "$file_id"
  done
}

validate_forecast_artifacts() {
  echo "[$(ts)] validating forecast artifacts from config"
  mapfile -t REQUIRED_PATHS < <(python - "$CONFIG_PATH" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

paths = []
for section in ("forecasted_los", "forecasted_discharge"):
    section_cfg = cfg.get(section) or {}
    if not section_cfg.get("enabled", False):
        continue
    for key in ("checkpoint_path", "calibration_path"):
        value = section_cfg.get(key)
        if value:
            paths.append(str(value))

joint_cfg = cfg.get("joint_forecast_pipeline") or {}
if joint_cfg.get("enabled", False):
    input_cfg = joint_cfg.get("joint_forecast_input") or {}
    source_run_dir = input_cfg.get("source_run_dir")
    if source_run_dir:
        paths.append(str(source_run_dir))
        paths.append(str(source_run_dir).rstrip("/") + "/joint_cache/cache_manifest.json")
    for key in (
        "train_cache_path",
        "val_cache_path",
        "test_cache_path",
        "gnn_val_cache_path",
        "outer_test_cache_path",
    ):
        value = input_cfg.get(key)
        if value:
            paths.append(str(value))

for path in dict.fromkeys(paths):
    print(path)
PY
  )

  missing=0
  for rel_path in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "$rel_path" ]]; then
      echo "[$(ts)] missing required forecast artifact: $rel_path"
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    notify "[FAIL] Missing forecast artifact(s) after gdown bootstrap. config=$CONFIG_PATH"
    hold_forever
  fi
}

cd "$REPO_DIR"
bash setup_vast.sh

mkdir -p /root/.config/rclone
if [[ -f "$RCLONE_B64_FILE" ]]; then
  base64 -d "$RCLONE_B64_FILE" > /root/.config/rclone/rclone.conf
else
  notify "[UPLOAD_FAIL] Missing RCLONE_B64_FILE at $RCLONE_B64_FILE. Holding."
  hold_forever
fi

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$ENV_NAME"
fi

cd "$REPO_DIR"
bootstrap_forecast_archives
validate_forecast_artifacts

if command -v nvidia-smi >/dev/null 2>&1; then
  WORKER_DEVICE="cuda:0"
else
  WORKER_DEVICE="cpu"
fi

echo "[$(ts)] starting training on device=$WORKER_DEVICE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

set +e
python -m src.main --config "$CONFIG_PATH" --seed "$SEED" --device "$WORKER_DEVICE"
TRAIN_RC=$?
set -e

FAIL_RC="$TRAIN_RC"
if [[ $TRAIN_RC -eq 0 ]]; then
  notify "[SUCCESS] Training completed. model=$MODEL_NAME config=$CONFIG_PATH seed=$SEED"
else
  notify "[FAIL] Training failed (rc=$TRAIN_RC). model=$MODEL_NAME config=$CONFIG_PATH seed=$SEED"
fi

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
  attempt=$((attempt + 1))
  sleep 5
done

if [[ $ok -ne 1 ]]; then
  rclone copy "$LOG_FILE" "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/logs/" \
      --retries 3 --low-level-retries 5 || true
  notify "[UPLOAD_FAIL] Upload failed after ${UPLOAD_RETRIES} attempts. Holding without shutdown."
  hold_forever
fi

rclone copy "$LOG_FILE" "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/logs/" \
    --retries 3 --low-level-retries 5 || true

if [[ "$FAIL_RC" -eq 0 ]]; then
  notify "[SUCCESS] Upload completed: ${RCLONE_REMOTE}:${RCLONE_DEST_DIR}"
  export SEND_MESSAGE_PY BOT_NAME
  if [[ -f "$VAST_TERMINATE_SH" ]]; then
    bash "$VAST_TERMINATE_SH"
  else
    notify "[TERMINATE_SKIP] vast_terminate.sh not found at $VAST_TERMINATE_SH. Holding."
    hold_forever
  fi
  hold_forever
fi

notify "[FAIL] Upload completed, but training had failures. Holding without shutdown."
hold_forever
BASH
)"

PIPELINE="${PIPELINE//__MODEL_NAME__/${MODEL_NAME}}"
PIPELINE="${PIPELINE//__CONFIG_PATH__/${CONFIG_PATH}}"
PIPELINE="${PIPELINE//__SEED__/${SEED}}"
PIPELINE="${PIPELINE//__ARCHIVE_SPECS__/${ARCHIVE_SPECS}}"
PIPELINE="${PIPELINE//__WORKSPACE_ROOT__/${WORKSPACE_ROOT}}"
PIPELINE="${PIPELINE//__REPO_DIR__/${REPO_DIR}}"
PIPELINE="${PIPELINE//__RUNS_DIR__/${RUNS_DIR}}"
PIPELINE="${PIPELINE//__DOWNLOAD_DIR__/${DOWNLOAD_DIR}}"
PIPELINE="${PIPELINE//__CONDA_SH__/${CONDA_SH}}"
PIPELINE="${PIPELINE//__ENV_NAME__/${ENV_NAME}}"
PIPELINE="${PIPELINE//__RCLONE_REMOTE__/${RCLONE_REMOTE}}"
PIPELINE="${PIPELINE//__RCLONE_DEST_DIR__/${RCLONE_DEST_DIR}}"
PIPELINE="${PIPELINE//__UPLOAD_RETRIES__/${UPLOAD_RETRIES}}"
PIPELINE="${PIPELINE//__SEND_MESSAGE_PY__/${SEND_MESSAGE_PY}}"
PIPELINE="${PIPELINE//__BOT_NAME__/${BOT_NAME}}"
PIPELINE="${PIPELINE//__VAST_TERMINATE_SH__/${VAST_TERMINATE_SH}}"
PIPELINE="${PIPELINE//__RCLONE_B64_FILE__/${RCLONE_B64_FILE}}"
PIPELINE="${PIPELINE//__CONTAINER_API_KEY__/${CONTAINER_API_KEY:-}}"
PIPELINE="${PIPELINE//__VAST_INSTANCE_ID__/${VAST_INSTANCE_ID:-}}"
PIPELINE="${PIPELINE//__DISCORD_WEBHOOK_URL__/${DISCORD_WEBHOOK_URL:-}}"

apt update
apt install -y tmux rclone

SESSION_NAME="${MODEL_NAME}_seed${SEED}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[$(ts)] tmux session exists: ${SESSION_NAME}"
else
  echo "[$(ts)] creating tmux session: ${SESSION_NAME}"
  tmux new-session -d -s "${SESSION_NAME}"
fi

PIPE_PATH="/tmp/${MODEL_NAME}__forecast_pipeline.sh"
printf "%s" "$PIPELINE" > "$PIPE_PATH"
chmod +x "$PIPE_PATH"

tmux set-environment -t "${SESSION_NAME}" RCLONE_CONF_B64 "${RCLONE_CONF_B64:-}"
tmux set-environment -t "${SESSION_NAME}" DISCORD_WEBHOOK_URL "${DISCORD_WEBHOOK_URL:-}"
tmux set-environment -t "${SESSION_NAME}" CONTAINER_API_KEY "${CONTAINER_API_KEY:-}"
tmux set-environment -t "${SESSION_NAME}" VAST_INSTANCE_ID "${VAST_INSTANCE_ID:-}"
tmux send-keys -t "${SESSION_NAME}" "bash $PIPE_PATH" C-m

echo "[$(ts)] started in tmux session '${SESSION_NAME}'."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
