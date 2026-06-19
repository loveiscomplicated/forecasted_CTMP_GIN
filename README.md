# Forecasted CTMP-GIN

Prospective Forecasted CTMP-GIN extension for TEDS-D episode-level classification.

This repository approximates CTMP-GIN-style cross-temporal message passing when discharge-side `_D` variables and LOS are not observed at admission time. It predicts future discharge variables and LOS from admission-only information, then builds synthetic/forecasted temporal graph inputs for downstream classification.

## Research Framing

```text
Oracle CTMP-GIN = upper bound
Admission-only = baseline
Forecasted CTMP-GIN = main prospective model
```

## Scope

Included:

- Discharge `_D` predictor.
- LOS predictor and LOS binning utilities.
- Forecast cache writer/reader/dataset support.
- Predicted discharge and predicted LOS downstream wrappers.
- Joint-consistent and joint-generative predictors.
- Outcome-aware training.
- Risk-head registry/resolver/override analysis.
- Drift diagnostics including P(D | LOS), JS divergence, dV_D, and risk-head analysis.
- 4-way downstream comparison paths:
  - predicted_D + oracle_LOS
  - oracle_D + predicted_LOS
  - predicted_D + predicted_LOS
  - oracle_D + oracle_LOS
- Self-contained CTMP-GIN core needed for oracle upper-bound and downstream comparison; this repo does not import an external CTMP-GIN repository.

Preservation note: this split intentionally keeps retrospective comparison/oracle/baseline/ablation code that may support Forecasted research reproduction. A narrower cleanup can happen in a later phase.

## Data

Place the raw CSV at:

```bash
src/data/raw/TEDS_Discharge.csv
```

Large/raw data, run directories, checkpoints, and tensor caches are intentionally not tracked.

## Environment

```bash
conda env create -f environment.yml
conda activate pyg_2
pip install -r requirements.txt
```

Quick dependency check:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

## Common Entrypoints

Original/oracle CTMP-GIN:

```bash
python src/main.py --config configs/ctmp_gin.yaml
```

Forecasted downstream CTMP-GIN examples:

```bash
python src/main.py --config configs/ctmp_gin_forecast_discharge_los_ce_distribution_leakage_free.yaml
python src/main.py --config configs/ctmp_gin_joint_fresh_id26.yaml
```

Predictor training:

```bash
python src/main_discharge.py --config configs/discharge_predictor.yaml
python src/main_los_ordinal.py --config configs/los_ordinal_predictor.yaml
```

Vast.ai launchers are retained for forecasted experiment reproduction and should be reviewed before new large sweeps.

## Outputs

Runs write to `runs/`, which is ignored:

- predictor checkpoints
- forecast caches
- downstream CTMP-GIN checkpoints
- diagnostics and report outputs
- k-fold summaries

## Layout

- `configs/`: original, baseline, forecasted, joint, LOS, and downstream experiment configs.
- `scripts/`: diagnostics, cache checks, risk-head, and report helper scripts.
- `src/data_processing/`: canonical preprocessing and predictor/downstream datasets.
- `src/models/ctmp_gin`: self-contained CTMP-GIN core.
- `src/models/discharge_predictor`: discharge/LOS/joint/risk-head predictor code.
- `src/models/forecasted_ctmp_gin`: forecasted contracts and outcome-aware helpers.
- `src/trainers/`: predictor, forecast cache, downstream, CV, and outcome-aware training.
- `src/analysis`: forecasted diagnostics and comparison analysis.
- `tests/`: forecasted and shared-core tests retained from the source split.
