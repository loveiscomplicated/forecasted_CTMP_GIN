#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash run_vast_cv.sh <model_name> <config_path> <seed>"
  echo "Example: bash run_vast_cv.sh ctmp_gin configs/ctmp_gin.yaml 1"
  exit 1
fi

MODEL_NAME="$1"
CONFIG_PATH="$2"
SEED="$3"

echo "model_name: ${MODEL_NAME}"
echo "config    : ${CONFIG_PATH}"
echo "seed      : ${SEED}"

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
BOT_NAME="vast_cv_${MODEL_NAME}"
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
echo "[$(ts)] preparing k-fold CV run"

check_torch_cuda_compat() {
  python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("TORCH_CUDA_COMPAT=cpu")
    sys.exit(0)

major, minor = torch.cuda.get_device_capability(0)
device_name = torch.cuda.get_device_name(0)
arch = f"sm_{major}{minor}"
supported = set(torch.cuda.get_arch_list())

print(f"TORCH_CUDA_DEVICE={device_name}")
print(f"TORCH_CUDA_DEVICE_ARCH={arch}")
print(f"TORCH_CUDA_SUPPORTED_ARCHES={','.join(sorted(supported))}")

if arch not in supported:
    print(f"TORCH_CUDA_COMPAT=unsupported:{arch}")
    sys.exit(42)

print("TORCH_CUDA_COMPAT=ok")
PY
}

set +e
CUDA_COMPAT_LOG="$(check_torch_cuda_compat 2>&1)"
CUDA_COMPAT_RC=$?
set -e
echo "$CUDA_COMPAT_LOG"
if [[ "$CUDA_COMPAT_RC" -eq 42 ]]; then
  notify "[FAIL] PyTorch CUDA build does not support this GPU architecture. model=$MODEL_NAME seed=$SEED. Choose a supported GPU or install a newer PyTorch CUDA build."
  hold_forever
elif [[ "$CUDA_COMPAT_RC" -ne 0 ]]; then
  notify "[FAIL] CUDA compatibility preflight failed rc=${CUDA_COMPAT_RC}. model=$MODEL_NAME seed=$SEED"
  hold_forever
fi

bootstrap_forecast_artifacts() {
  echo "[$(ts)] checking forecast artifacts from config"
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

for path in dict.fromkeys(paths):
    print(path)
PY
  )

  if [[ "${#REQUIRED_PATHS[@]}" -eq 0 ]]; then
    echo "[$(ts)] no forecast artifacts required"
    return 0
  fi

  mkdir -p "$RUNS_DIR"
  declare -A FETCHED_RUNS=()
  for rel_path in "${REQUIRED_PATHS[@]}"; do
    if [[ "$rel_path" != runs/* ]]; then
      echo "[$(ts)] non-runs artifact path configured, expecting it to exist locally: $rel_path"
      continue
    fi
    run_id="$(printf '%s\n' "$rel_path" | cut -d/ -f2)"
    if [[ -z "$run_id" ]]; then
      continue
    fi
    if [[ ! -e "$rel_path" && -z "${FETCHED_RUNS[$run_id]:-}" ]]; then
      echo "[$(ts)] downloading forecast run: ${run_id}"
      if rclone copy "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/${run_id}" "${RUNS_DIR}/${run_id}" \
        --create-empty-src-dirs \
        --transfers 8 \
        --checkers 16 \
        --retries 3 \
        --low-level-retries 10
      then
        echo "[$(ts)] downloaded forecast run: ${run_id}"
      else
        echo "[$(ts)] failed to download forecast run: ${run_id}"
      fi
      FETCHED_RUNS[$run_id]=1
    fi
  done

  missing=0
  for rel_path in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "$rel_path" ]]; then
      echo "[$(ts)] missing required forecast artifact after bootstrap: $rel_path"
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    notify "[FAIL] Missing forecast artifact(s) after rclone bootstrap. config=$CONFIG_PATH"
    hold_forever
  fi
}

bootstrap_forecast_artifacts

PREP_LOG="$(python -m src.main --config "$CONFIG_PATH" --seed "$SEED" --prepare_cv_only)"
echo "$PREP_LOG"

CV_RUN_DIR="$(printf '%s\n' "$PREP_LOG" | sed -n 's/^CV_RUN_DIR=//p' | tail -n 1)"
if [[ -z "$CV_RUN_DIR" ]]; then
  notify "[FAIL] Could not parse CV_RUN_DIR from prepare step."
  hold_forever
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  GPU_COUNT=0
fi
if [[ "$GPU_COUNT" -lt 1 ]]; then
  GPU_COUNT=1
  WORKER_DEVICE="cpu"
else
  WORKER_DEVICE="cuda:0"
fi

K_FOLDS="$(python -c "import json; print(json.load(open('${CV_RUN_DIR}/kfold_splits.json'))['n_folds'])")"
LOG_DIR="${CV_RUN_DIR}/launcher_logs"
mkdir -p "$LOG_DIR"

echo "[$(ts)] CV_RUN_DIR=$CV_RUN_DIR"
echo "[$(ts)] detected_gpus=$GPU_COUNT n_folds=$K_FOLDS worker_device=$WORKER_DEVICE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_FOLDS=()
declare -a ACTIVE_GPUS=()

NEXT_FOLD=0
FAIL_RC=0
FAIL_FOLD=""

start_fold() {
  local fold="$1"
  local gpu="$2"
  echo "[$(ts)] start fold=$fold gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python -m src.main \
      --config "$CONFIG_PATH" \
      --seed "$SEED" \
      --device "$WORKER_DEVICE" \
      --fold "$fold" \
      --cv_run_dir "$CV_RUN_DIR" \
      > "${LOG_DIR}/fold_${fold}.log" 2>&1 &

  ACTIVE_PIDS+=("$!")
  ACTIVE_FOLDS+=("$fold")
  ACTIVE_GPUS+=("$gpu")
}

INITIAL_WORKERS="$GPU_COUNT"
if [[ "$K_FOLDS" -lt "$INITIAL_WORKERS" ]]; then
  INITIAL_WORKERS="$K_FOLDS"
fi

for ((slot=0; slot<INITIAL_WORKERS; slot++)); do
  start_fold "$NEXT_FOLD" "$slot"
  NEXT_FOLD=$((NEXT_FOLD + 1))
done

while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
  UPDATED_PIDS=()
  UPDATED_FOLDS=()
  UPDATED_GPUS=()

  for i in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$i]}"
    fold="${ACTIVE_FOLDS[$i]}"
    gpu="${ACTIVE_GPUS[$i]}"

    if kill -0 "$pid" 2>/dev/null; then
      UPDATED_PIDS+=("$pid")
      UPDATED_FOLDS+=("$fold")
      UPDATED_GPUS+=("$gpu")
      continue
    fi

    wait "$pid" || rc=$?
    rc="${rc:-0}"

    if [[ "$rc" -ne 0 ]]; then
      echo "[$(ts)] fold=$fold failed rc=$rc"
      FAIL_RC="$rc"
      FAIL_FOLD="$fold"
      notify "[FAIL] fold=$fold failed on gpu=$gpu rc=$rc. See ${LOG_DIR}/fold_${fold}.log"
    else
      echo "[$(ts)] fold=$fold completed on gpu=$gpu"
    fi

    if [[ "$FAIL_RC" -eq 0 && "$NEXT_FOLD" -lt "$K_FOLDS" ]]; then
      start_fold "$NEXT_FOLD" "$gpu"
      NEXT_FOLD=$((NEXT_FOLD + 1))
    fi
    rc=0
  done

  ACTIVE_PIDS=("${UPDATED_PIDS[@]}")
  ACTIVE_FOLDS=("${UPDATED_FOLDS[@]}")
  ACTIVE_GPUS=("${UPDATED_GPUS[@]}")

  if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
    sleep 5
  fi
done

echo "[$(ts)] finalizing CV summary"
python -m src.main --config "$CONFIG_PATH" --seed "$SEED" --cv_run_dir "$CV_RUN_DIR" --finalize_cv \
  | tee "${LOG_DIR}/finalize.log"

SUMMARY_STATUS="$(python -c "import json; print(json.load(open('${CV_RUN_DIR}/cv_summary.json'))['status'])")"

if [[ "$FAIL_RC" -eq 0 && "$SUMMARY_STATUS" == "completed" ]]; then
  notify "[SUCCESS] K-fold CV completed. model=$MODEL_NAME config=$CONFIG_PATH seed=$SEED run_dir=$CV_RUN_DIR"
else
  notify "[FAIL] K-fold CV finished with errors. failed_fold=${FAIL_FOLD:-none} run_dir=$CV_RUN_DIR"
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
  notify "[UPLOAD_FAIL] Upload failed after ${UPLOAD_RETRIES} attempts. Holding without shutdown."
  hold_forever
fi

if [[ "$FAIL_RC" -eq 0 && "$SUMMARY_STATUS" == "completed" ]]; then
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

notify "[FAIL] Upload completed, but CV had failures. Holding without shutdown."
hold_forever
BASH
)"

PIPELINE="${PIPELINE//__MODEL_NAME__/${MODEL_NAME}}"
PIPELINE="${PIPELINE//__CONFIG_PATH__/${CONFIG_PATH}}"
PIPELINE="${PIPELINE//__SEED__/${SEED}}"
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

SESSION_NAME="${MODEL_NAME}_cv_seed${SEED}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[$(ts)] tmux session exists: ${SESSION_NAME}"
else
  echo "[$(ts)] creating tmux session: ${SESSION_NAME}"
  tmux new-session -d -s "${SESSION_NAME}"
fi

PIPE_PATH="/tmp/${MODEL_NAME}__cv_pipeline.sh"
printf "%s" "$PIPELINE" > "$PIPE_PATH"
chmod +x "$PIPE_PATH"

tmux set-environment -t "${SESSION_NAME}" RCLONE_CONF_B64 "${RCLONE_CONF_B64:-}"
tmux set-environment -t "${SESSION_NAME}" DISCORD_WEBHOOK_URL "${DISCORD_WEBHOOK_URL:-}"
tmux set-environment -t "${SESSION_NAME}" CONTAINER_API_KEY "${CONTAINER_API_KEY:-}"
tmux set-environment -t "${SESSION_NAME}" VAST_INSTANCE_ID "${VAST_INSTANCE_ID:-}"
tmux send-keys -t "${SESSION_NAME}" "bash $PIPE_PATH" C-m

echo "[$(ts)] started in tmux session '${SESSION_NAME}'."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
