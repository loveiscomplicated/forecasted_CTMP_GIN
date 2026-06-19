#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash run_vast_forecast_los_focal_sqrt_alpha_g1_distribution.sh --seed <seed>"
  echo "Legacy: bash run_vast_forecast_los_focal_sqrt_alpha_g1_distribution.sh <seed>"
  echo "Config: configs/ctmp_gin_forecast_discharge_los_focal_sqrt_alpha_g1_distribution_leakage_free.yaml"
  echo "Attach: tmux attach -t ctmp_gin_forecast_discharge_los_focal_sqrt_alpha_g1_distribution_leakage_free_cv_seed<seed>"
}

SEED=""

if [[ $# -eq 1 ]]; then
  SEED="$1"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --seed)
        SEED="${2:-}"
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
fi

if [[ -z "$SEED" ]]; then
  usage
  exit 1
fi

MODEL_NAME="ctmp_gin_forecast_discharge_los_focal_sqrt_alpha_g1_distribution_leakage_free"
CONFIG_PATH="configs/ctmp_gin_forecast_discharge_los_focal_sqrt_alpha_g1_distribution_leakage_free.yaml"
SESSION_NAME="${MODEL_NAME}_cv_seed${SEED}"

echo "config : ${CONFIG_PATH}"
echo "seed   : ${SEED}"
echo "tmux   : ${SESSION_NAME}"

bash run_vast_cv.sh "${MODEL_NAME}" "${CONFIG_PATH}" "${SEED}"
