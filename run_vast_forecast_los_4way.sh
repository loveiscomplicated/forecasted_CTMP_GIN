#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash run_vast_forecast_los_4way.sh <seed>"
  exit 1
fi

SEED="$1"

CONFIGS=(
  "configs/ctmp_gin_forecast_discharge_los_ce_baseline_leakage_free.yaml"
  "configs/ctmp_gin_forecast_discharge_los_ce_distribution_leakage_free.yaml"
  "configs/ctmp_gin_forecast_discharge_los_focal_sqrt_alpha_g1_baseline_leakage_free.yaml"
  "configs/ctmp_gin_forecast_discharge_los_focal_sqrt_alpha_g1_distribution_leakage_free.yaml"
)

CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "Conda init script not found: $CONDA_SH"
  exit 1
fi

source "$CONDA_SH"
conda activate pyg_2

for config_path in "${CONFIGS[@]}"; do
  echo "============================================================"
  echo "Running: $config_path  seed=$SEED"
  python -m src.main --config "$config_path" --seed "$SEED"
done

echo "All 4 forecast LOS runs completed."
