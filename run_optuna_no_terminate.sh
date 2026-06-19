#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Args
# -----------------------
if [[ $# -lt 2 ]]; then
  echo "Usage: bash run.sh <model_name> <config_path>"
  echo "Example: bash run.sh gin configs/gin.yaml"
  exit 1
fi

MODEL_NAME="$1"
CONFIG_PATH="$2"

echo "model_name: ${MODEL_NAME}"
echo "config    : ${CONFIG_PATH}"

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

ts() { date '+%Y-%m-%d %H:%M:%S'; }

RCLONE_B64_FILE="/tmp/rclone_conf.b64"

# wait for env injection (up to 120s)
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

# notifier
SEND_MESSAGE_PY="${REPO_DIR}/src/utils/send_message.py"
BOT_NAME="runpod_optuna_${MODEL_NAME}"

EPOCHS="${EPOCHS:-20}"
TOTAL_TRIALS="${TOTAL_TRIALS:-50}"



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
# Build pipeline to run INSIDE tmux
# -----------------------
mkdir -p /root/.config/rclone

PIPELINE="$(cat <<'BASH'
set -euo pipefail
ts() { date '+%Y-%m-%d %H:%M:%S'; }

MODEL_NAME="__MODEL_NAME__"
CONFIG_PATH="__CONFIG_PATH__"
EPOCHS="__EPOCHS__"
TOTAL_TRIALS="__TOTAL_TRIALS__"
RCLONE_B64_FILE="__RCLONE_B64_FILE__"

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
BOT_NAME="runpod_optuna_${MODEL_NAME}"

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
bash setup.sh

echo "[$(ts)] Setting up rclone config before training..."
mkdir -p /root/.config/rclone
if [[ -f "$RCLONE_B64_FILE" ]]; then
  base64 -d "$RCLONE_B64_FILE" > /root/.config/rclone/rclone.conf
  echo "[$(ts)] rclone.conf created."
else
  echo "[$(ts)] Warning: RCLONE_B64_FILE not found at this stage!"
fi

# -----------------------
# Training (Parallel Optuna workers: 1 worker per GPU)
# -----------------------
cd "$REPO_DIR"
echo "[$(ts)] training start (parallel optuna workers)"

# (A) Ensure env (conda example)
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$ENV_NAME"
fi

bash postgres.sh

# (B) Wait for Postgres to be ready (if you run postgres locally)
# change port/user/db as your postgres.sh uses
PG_HOST="${PG_HOST:-127.0.0.1}"
PG_PORT="${PG_PORT:-5432}"

for k in {1..120}; do
  if command -v pg_isready >/dev/null 2>&1; then
    if pg_isready -h "$PG_HOST" -p "$PG_PORT" >/dev/null 2>&1; then
      echo "[$(ts)] postgres is ready"
      break
    fi
  else
    (echo >/dev/tcp/"$PG_HOST"/"$PG_PORT") >/dev/null 2>&1 && { echo "[$(ts)] postgres reachable"; break; }
  fi
  sleep 1
  if [[ "$k" -eq 120 ]]; then
    notify "[FAIL] Postgres not ready after 120s. Holding."
    hold_forever
  fi
done

# (C) Detect GPUs
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  GPU_COUNT=1
fi
if [[ "$GPU_COUNT" -lt 1 ]]; then GPU_COUNT=1; fi

GPU_IDS=()
for ((g=0; g<GPU_COUNT; g++)); do GPU_IDS+=("$g"); done
echo "[$(ts)] detected GPUs: ${GPU_IDS[*]}"

# (D) Avoid CPU oversubscription
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# (E) Unique study name per (model, config) to avoid mixing
CFG_BASENAME="$(basename "$CONFIG_PATH")"
STUDY_NAME="${MODEL_NAME}__${CFG_BASENAME%.*}"

# (F) Total trials control (set TOTAL_TRIALS via env var; default 50)
WORKERS="${#GPU_IDS[@]}"
# ceil division: per_worker = (TOTAL_TRIALS + WORKERS - 1) / WORKERS
PER_WORKER=$(( (TOTAL_TRIALS + WORKERS - 1) / WORKERS ))
echo "[$(ts)] total_trials=$TOTAL_TRIALS workers=$WORKERS per_worker=$PER_WORKER study_name=$STUDY_NAME"

# (G) Logs into runs for upload/debug
LOG_DIR="${RUNS_DIR}/optuna_logs/${STUDY_NAME}"
mkdir -p "$LOG_DIR"

pids=()
rc=0

echo "[$(ts)] initializing optuna study schema..."
python -m src.trainers.run_parameter_search_optuna \
    --config "$CONFIG_PATH" \
    --study-name "$STUDY_NAME" \
    --init-only

for i in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$i]}"
  echo "[$(ts)] start worker $i on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  python -m src.trainers.run_parameter_search_optuna \
    --config "$CONFIG_PATH" \
    --study-name "$STUDY_NAME" \
    --n-trials "$PER_WORKER" \
    --epochs "$EPOCHS" \
    > "${LOG_DIR}/worker_${i}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

TRAIN_RC="$rc"
if [[ "$TRAIN_RC" -eq 0 ]]; then
  notify "[SUCCESS] Training completed. workers=$WORKERS study=$STUDY_NAME model=$MODEL_NAME config=$CONFIG_PATH total_trials=$TOTAL_TRIALS"
else
  notify "[FAIL] Training failed (some workers failed). workers=$WORKERS study=$STUDY_NAME model=$MODEL_NAME config=$CONFIG_PATH"
fi

# -----------------------
# Upload runs (always try) + retry policy C
#   - upload fails => notify + HOLD (no shutdown)
# -----------------------

attempt=1
ok=0

ARCHIVE="/tmp/${STUDY_NAME}.tar.gz"

while [[ $attempt -le $UPLOAD_RETRIES ]]; do
  echo "[$(ts)] upload attempt $attempt/$UPLOAD_RETRIES ..."

  if [[ ! -d "$RUNS_DIR/optuna_logs/$STUDY_NAME" ]]; then
    notify "[UPLOAD_FAIL] log dir missing: $RUNS_DIR/optuna_logs/$STUDY_NAME"
    hold_forever
  fi

  # 1) archive 만들기 (매번 새로)
  rm -f "$ARCHIVE"
  tar -czf "$ARCHIVE" -C "$RUNS_DIR" "optuna_logs/$STUDY_NAME"

  # 2) 업로드 (rclone 옵션은 여기!)
  if rclone copyto "$ARCHIVE" "${RCLONE_REMOTE}:${RCLONE_DEST_DIR}/${STUDY_NAME}.tar.gz" \
      --transfers 2 \
      --checkers 4 \
      --tpslimit 5 \
      --tpslimit-burst 5 \
      --drive-chunk-size 32M \
      --retries 3 \
      --low-level-retries 5 \
      --stats 10s
  then
    ok=1
    break
  fi

  attempt=$((attempt+1))
  sleep 10
done

# --- [데이터 정리 및 백업] ---
echo "[$(ts)] Stopping PostgreSQL and backing up data..."
# DB 서비스 중지 (오류 무시)
service postgresql stop || true

# rsync 설치 확인
if ! command -v rsync >/dev/null 2>&1; then
  apt-get update && apt-get install -y rsync
fi

mkdir -p /workspace/pgdata

# 실제 데이터 경로 확인 (psql 실행 실패 시 기본 경로 사용)
PG_REAL_DATA=$(psql -t -A -c "SHOW data_directory;" 2>/dev/null || echo "/var/lib/postgresql/14/main")

# 경로가 존재하는지 확인 후 rsync 실행
if [[ -d "$PG_REAL_DATA" ]]; then
  echo "[$(ts)] Syncing from $PG_REAL_DATA to /workspace/pgdata..."
  rsync -av --no-owner --no-group "$PG_REAL_DATA/" /workspace/pgdata/
  tar -czf /workspace/pgdata_backup.tar.gz -C /workspace pgdata
else
  echo "[$(ts)] Warning: PG_REAL_DATA ($PG_REAL_DATA) not found. Skipping DB backup."
fi

if [[ $ok -eq 0 ]]; then
  notify "[UPLOAD_FAIL] Upload failed after ${UPLOAD_RETRIES} attempts. Holding without shutdown."
else
  notify "[UPLOAD_OK] Upload succeeded. archive=${STUDY_NAME}.tar.gz. Holding without shutdown. [no terminate]"
fi

hold_forever

BASH
)"

# Fill placeholders
PIPELINE="${PIPELINE//__MODEL_NAME__/${MODEL_NAME}}"
PIPELINE="${PIPELINE//__CONFIG_PATH__/${CONFIG_PATH}}"
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
PIPELINE="${PIPELINE//__RCLONE_B64_FILE__/${RCLONE_B64_FILE}}"
PIPELINE="${PIPELINE//__UPLOAD_RETRIES__/${UPLOAD_RETRIES}}"
PIPELINE="${PIPELINE//__SEND_MESSAGE_PY__/${SEND_MESSAGE_PY}}"
PIPELINE="${PIPELINE//__EPOCHS__/${EPOCHS}}"
PIPELINE="${PIPELINE//__TOTAL_TRIALS__/${TOTAL_TRIALS}}"

# -----------------------
# tmux session: create and start
# -----------------------
if ! command -v tmux >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y tmux
fi

if tmux has-session -t "${MODEL_NAME}" 2>/dev/null; then
  echo "[$(ts)] tmux session exists: ${MODEL_NAME}"
else
  echo "[$(ts)] creating tmux session: ${MODEL_NAME}"
  tmux new-session -d -s "${MODEL_NAME}"
fi

PIPE_PATH="/tmp/${MODEL_NAME}__pipeline.sh"
printf "%s" "$PIPELINE" > "$PIPE_PATH"
chmod +x "$PIPE_PATH"

tmux set-environment -g RCLONE_B64_FILE "$RCLONE_B64_FILE"
tmux send-keys -t "${MODEL_NAME}" "bash $PIPE_PATH" C-m

echo "[$(ts)] started in tmux session '${MODEL_NAME}'."
echo "Attach with: tmux attach -t ${MODEL_NAME}"