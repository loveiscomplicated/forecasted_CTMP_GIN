# Split Uncertain Files

This repository was split with preservation priority. Files that may support forecasted retrospective comparison, oracle upper-bound, baseline, or downstream ablation work were retained unless clearly large/generated.

## Retained For Manual Review

- `configs/ctmp_gin*.yaml`, `configs/gin*.yaml`, `configs/a3tgcn*.yaml`, `configs/xgboost*.yaml`: some original configs may be needed for oracle/baseline comparisons.
- `src/trainers/run_parameter_search_optuna*.py`: original large sweep utilities are not the forecasted main path but may be useful for comparison baselines.
- `run_optuna*.sh`, `run_vast.sh`, `run_vast_cv.sh`: original launchers retained for comparison/oracle reproduction review.
- `src/analysis/*.ipynb`, `src/analysis/*.csv`, `src/analysis/*.png`: mixed exploratory and generated analysis files retained for cleanup phase.
- `src/explainers/results/**`: generated explanation outputs retained if copied because they are small, but should be reviewed later.
- `reports/**`: forecasted/drift/risk-head summaries retained as small reproducibility references.
- `ㅁㄴㅇㄹ.md`: scratch note with unclear final role.

## Excluded As Large Or Raw Artifacts

- `runs/`
- `wandb/`
- `src/wandb/`
- `checkpoints/`
- `artifacts/`
- raw CSV/parquet/feather data under `src/data/raw/`
- processed data directories
- tensor/checkpoint/cache files such as `*.pt`, `*.pth`, `*.ckpt`
