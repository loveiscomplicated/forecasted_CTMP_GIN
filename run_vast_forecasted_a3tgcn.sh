#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash run_vast_forecasted_a3tgcn.sh --seed <seed>"
  echo "Legacy: bash run_vast_forecasted_a3tgcn.sh <seed>"
  echo "Config: configs/a3tgcn_forecast_discharge_los_ce_baseline_leakage_free.yaml"
  echo "Attach: tmux attach -t a3tgcn_forecast_discharge_los_ce_baseline_leakage_free_cv_seed<seed>"
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

MODEL_NAME="a3tgcn_forecast_discharge_los_ce_baseline_leakage_free"
CONFIG_PATH="configs/a3tgcn_forecast_discharge_los_ce_baseline_leakage_free.yaml"
SESSION_NAME="${MODEL_NAME}_cv_seed${SEED}"

echo "config : ${CONFIG_PATH}"
echo "seed   : ${SEED}"
echo "tmux   : ${SESSION_NAME}"

bash run_vast_cv.sh "${MODEL_NAME}" "${CONFIG_PATH}" "${SEED}"
