#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash run_vast_forecast_los_hybrid_distribution.sh <seed>"
  echo "Config: configs/ctmp_gin_forecast_discharge_los_hybrid_distribution_leakage_free.yaml"
  echo "Attach: tmux attach -t ctmp_gin_forecast_discharge_los_hybrid_distribution_leakage_free_cv_seed<seed>"
  exit 1
fi

SEED="$1"
MODEL_NAME="ctmp_gin_forecast_discharge_los_hybrid_distribution_leakage_free"
CONFIG_PATH="configs/ctmp_gin_forecast_discharge_los_hybrid_distribution_leakage_free.yaml"
SESSION_NAME="${MODEL_NAME}_cv_seed${SEED}"

echo "config : ${CONFIG_PATH}"
echo "seed   : ${SEED}"
echo "tmux   : ${SESSION_NAME}"

bash run_vast_cv.sh "${MODEL_NAME}" "${CONFIG_PATH}" "${SEED}"
