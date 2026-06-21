#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash run_vast_forecast_diagnostics_4way.sh --seed <seed> --mode <mode_name> [--config <config>]"
  echo "Default config: configs/ctmp_gin_forecast_discharge_los_ce_baseline_leakage_free.yaml"
  echo "Modes:"
  echo "  oracle_D_predicted_LOS_hard"
  echo "  oracle_D_predicted_LOS_distribution"
  echo "  predicted_D_oracle_LOS"
  echo "  predicted_D_predicted_LOS"
}

SEED=""
CONFIG_PATH="configs/ctmp_gin_forecast_discharge_los_ce_baseline_leakage_free.yaml"
REQUESTED_MODE=""
CONFIG_EXPLICIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:-}"
      CONFIG_EXPLICIT=1
      shift 2
      ;;
    --mode)
      REQUESTED_MODE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$SEED" || -z "$REQUESTED_MODE" ]]; then
  usage
  exit 1
fi

case "$REQUESTED_MODE" in
  oracle_D_predicted_LOS_hard|oracle_D_predicted_LOS_distribution|predicted_D_oracle_LOS|predicted_D_predicted_LOS)
    ;;
  *)
    echo "Unsupported mode: $REQUESTED_MODE"
    usage
    exit 1
    ;;
esac

if [[ "$CONFIG_EXPLICIT" -eq 0 && "$REQUESTED_MODE" == "oracle_D_predicted_LOS_distribution" ]]; then
  CONFIG_PATH="configs/ctmp_gin_forecast_discharge_los_ce_distribution_leakage_free.yaml"
fi

MODEL_NAME="forecast_diag_${REQUESTED_MODE}"
WORKSPACE_ROOT="/workspace"
REPO_DIR="${WORKSPACE_ROOT}/forecasted_CTMP_GIN"
RUNS_DIR="${REPO_DIR}/runs"
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

for _ in {1..120}; do
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
REQUESTED_MODE="__REQUESTED_MODE__"
WORKSPACE_ROOT="__WORKSPACE_ROOT__"
REPO_DIR="__REPO_DIR__"
RUNS_DIR="__RUNS_DIR__"
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

LOG_FILE="${WORKSPACE_ROOT}/diagnose_${MODEL_NAME}_seed${SEED}.log"
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

if command -v nvidia-smi >/dev/null 2>&1; then
  WORKER_DEVICE="cuda"
else
  WORKER_DEVICE="cpu"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

notify "[START] Forecast diagnostic. mode=$REQUESTED_MODE config=$CONFIG_PATH seed=$SEED device=$WORKER_DEVICE"
echo "[$(ts)] config=$CONFIG_PATH"
echo "[$(ts)] seed=$SEED"
echo "[$(ts)] device=$WORKER_DEVICE"
echo "[$(ts)] mode=$REQUESTED_MODE"

FAIL_RC=0
echo "[$(ts)] ===== start mode=${REQUESTED_MODE} ====="
set +e
python scripts/diagnose_forecast_cache_alignment.py \
  --config "$CONFIG_PATH" \
  --mode "$REQUESTED_MODE" \
  --fold 0 \
  --seed "$SEED" \
  --device "$WORKER_DEVICE"
FAIL_RC=$?
set -e

if [[ "$FAIL_RC" -ne 0 ]]; then
  notify "[FAIL] Forecast diagnostic failed. mode=$REQUESTED_MODE rc=$FAIL_RC config=$CONFIG_PATH seed=$SEED"
else
  notify "[SUCCESS] Forecast diagnostic completed. mode=$REQUESTED_MODE config=$CONFIG_PATH seed=$SEED"
  echo "[$(ts)] ===== completed mode=${REQUESTED_MODE} ====="
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
  notify "[SUCCESS] Forecast diagnostic and upload completed. mode=$REQUESTED_MODE config=$CONFIG_PATH seed=$SEED remote=${RCLONE_REMOTE}:${RCLONE_DEST_DIR}"
  export SEND_MESSAGE_PY BOT_NAME
  if [[ -f "$VAST_TERMINATE_SH" ]]; then
    bash "$VAST_TERMINATE_SH"
  else
    notify "[TERMINATE_SKIP] vast_terminate.sh not found at $VAST_TERMINATE_SH. Holding."
    hold_forever
  fi
  hold_forever
fi

notify "[FAIL] Upload completed, but diagnostic failed. mode=$REQUESTED_MODE rc=$FAIL_RC. Holding without shutdown."
hold_forever
BASH
)"

PIPELINE="${PIPELINE//__MODEL_NAME__/${MODEL_NAME}}"
PIPELINE="${PIPELINE//__CONFIG_PATH__/${CONFIG_PATH}}"
PIPELINE="${PIPELINE//__SEED__/${SEED}}"
PIPELINE="${PIPELINE//__REQUESTED_MODE__/${REQUESTED_MODE}}"
PIPELINE="${PIPELINE//__WORKSPACE_ROOT__/${WORKSPACE_ROOT}}"
PIPELINE="${PIPELINE//__REPO_DIR__/${REPO_DIR}}"
PIPELINE="${PIPELINE//__RUNS_DIR__/${RUNS_DIR}}"
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
