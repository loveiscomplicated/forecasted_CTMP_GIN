# Repository Split Plan

## Goals

Split this source repository into two independent, self-contained sibling repositories:

- `../CTMP-GIN`: original retrospective CTMP-GIN research code.
- `../Forecasted-CTMP-GIN`: forecasted/prospective CTMP-GIN extension code.

The source repository must remain intact. The only permitted source modification for this split is this plan document. Existing uncommitted source changes are copied into the split repositories but are not committed or otherwise modified in the source repository.

## Source Snapshot

- Source path: `/Users/jeong-yunseong/Documents/programming/Phase_2_public`
- Source branch: `vastai`
- Source HEAD: `6d7a0e505248d9fe9a3a8dcd4794af1e446cf305`
- Source dirty files at audit time: `run_vast_cv.sh`

## Repository Scope

### CTMP-GIN

Keep the original retrospective CTMP-GIN setup:

- CTMP-GIN model using admission graph, discharge graph, and actual LOS cross-temporal edge features.
- Original graph construction, MI edge construction, GIN/A3TGCN/GIN-GRU/XGBoost/MLP baselines.
- Original configs, ablations, Optuna/CV/evaluation/explainer utilities needed for retrospective reproduction.

Remove forecasted/prospective implementation code:

- Discharge/LOS predictors.
- Forecast cache dataset/writer/reader.
- Predicted discharge and predicted LOS downstream wrappers.
- Joint-consistent, joint-generative, outcome-aware, risk-head, drift diagnostics.
- Forecasted configs, tests, scripts, and implementation imports.

Documentation may mention that the forecasted extension was split into a separate repository.

### Forecasted-CTMP-GIN

Keep the forecasted/prospective extension with preservation priority:

- Forecasted CTMP-GIN predictors and downstream wrappers.
- Discharge `_D` predictor, LOS predictor, forecast cache, joint-consistent predictor, joint-generative predictor, outcome-aware training.
- Risk-head registry/resolver/override suite, LOS coarse/ordinal/binning code, drift diagnostics.
- Oracle/comparison paths required for Forecasted research: oracle CTMP-GIN upper bound, admission-only baselines, predicted_D + oracle_LOS, oracle_D + predicted_LOS, predicted_D + predicted_LOS, oracle_D + oracle_LOS, 4-way downstream comparison.
- Shared CTMP-GIN core copied locally; no dependency on external CTMP-GIN repo.

Ambiguous retrospective comparison/oracle/baseline/ablation code must be retained and recorded in `SPLIT_UNCERTAIN_FILES.md` instead of deleted.

## Shared Core

The following code is copied into both repositories as self-contained core:

- `src/data_processing` canonical dataset, splits, MI, edge, tensor utilities.
- `src/models/ctmp_gin`, `src/models/gin`, `src/models/a3tgcn`, `src/models/gingru`, `src/models/xgboost`, `src/models/mlp`, `src/models/entity_embedding`, `src/models/utils`.
- `src/trainers/base.py`, `src/trainers/run_single_experiment.py`, `src/trainers/run_kfold_cv.py`, trainer utilities.
- `src/utils`, dependency manifests, small configs, tests/fixtures needed for smoke tests.

## Mixed File Patch Strategy

Patch mixed files one at a time in `../CTMP-GIN`, with an import smoke test after each file:

1. `src/main.py`: remove forecast/joint/outcome-aware CLI branches.
2. `src/trainers/base.py`: remove forecast metadata, soft-discharge, and provider override paths.
3. `src/trainers/run_kfold_cv.py`: remove forecast cache, joint pipeline, and outcome-aware stage2 paths.
4. `src/models/ctmp_gin/*`: remove forecast distribution input, soft discharge, and predicted LOS distribution paths while preserving retrospective CTMP-GIN backbone and actual LOS edge embedding.
5. `src/models/gin/model.py`: remove forecast distribution/soft-discharge paths.

`../Forecasted-CTMP-GIN` keeps these mixed branches during this split phase.

## Exclusions

Do not copy large/generated artifacts:

- `.git`
- `runs/`
- `checkpoints/`
- `artifacts/`
- `wandb/`
- `src/wandb/`
- raw and processed data files
- cache tensors/checkpoints
- `__pycache__/`, `.pytest_cache/`, local editor/system files

Small JSON/CSV/Markdown summaries, config files, registry documents, and small test fixtures may be retained. Ambiguous files are documented in `SPLIT_UNCERTAIN_FILES.md`.

## Verification

For each split repository:

```bash
python -m compileall src
pytest -q
python - <<'PY'
import src
print("import ok")
PY
python src/main.py --help
```

For `../CTMP-GIN`, additionally run:

```bash
rg -i "forecast|predicted_D|predicted_LOS|joint_consistent|outcome_aware|joint_generative|risk_head|drift"
```

Report matches separately as implementation files and documentation/report files. Implementation matches must not include forecasted imports, classes, config branches, or CLI paths.

## Commit Policy

Before initial commits, report each split repository's:

- `git status --short`
- large-file scan results
- top-level tree summary
- `SOURCE_REPO_COMMIT.txt`
- `SPLIT_UNCERTAIN_FILES.md`

Initial commit messages:

- `CTMP-GIN`: `Initial split: original retrospective CTMP-GIN`
- `Forecasted-CTMP-GIN`: `Initial split: forecasted prospective CTMP-GIN`

Each repository must include `REPO_SPLIT_REPORT.md` with generated path, tree summary, removed files, retained shared core, mixed-file patch summary, unresolved uncertain files, verification results, failures, and follow-up TODOs.
