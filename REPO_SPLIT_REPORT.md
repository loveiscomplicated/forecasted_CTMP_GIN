# Repository Split Report: Forecasted-CTMP-GIN

## Generated Path

`/Users/jeong-yunseong/Documents/programming/Forecasted-CTMP-GIN`

## Final Tree Summary

Top-level contents retained with preservation priority:

- `src/`: forecasted predictors, forecast cache/downstream paths, CTMP-GIN core, baselines, trainers, preprocessing, diagnostics, metrics, and explainers.
- `configs/`: forecasted experiment configs plus original/oracle/baseline comparison configs.
- `scripts/`: forecast diagnostics, risk-head override, joint plausibility, MI, and infrastructure helpers.
- `tests/`: forecasted predictor, diagnostics, outcome-aware, risk-head, soft-discharge, and CTMP-GIN LOS tests.
- `docs/`, `reports/`, forecasted registry, README, split metadata, and split report.

## Removed Main Files And Directories

Only clearly large/generated artifacts were removed during this split phase:

- `.git/` from the source copy.
- `runs/`, `wandb/`, `checkpoints/`, `artifacts/`, raw data payloads, processed data directories, tensor/checkpoint/cache files.
- Large generated analysis artifact: `src/analysis/gated_fusion_exports/gated_fusion_samples.csv`.
- Python bytecode and test caches generated during verification.

Retrospective comparison/oracle/baseline/4-way ablation code was intentionally retained for a later cleanup phase.

## Shared Core Retained

- CTMP-GIN model and LOS edge embedding.
- Admission/discharge graph construction and preprocessing utilities.
- GIN, A3TGCN, GIN-GRU, MLP admission-only, and XGBoost baselines.
- Metrics, trainer utilities, CV helpers, save/load helpers, and device/seed utilities.
- Oracle CTMP-GIN upper-bound and admission-only comparison paths.

## Forecasted Scope Retained

- Discharge `_D` predictor and LOS predictor.
- Forecast cache contracts, writers/readers, and downstream forecasted CTMP-GIN wrappers.
- `predicted_D + oracle_LOS`, `oracle_D + predicted_LOS`, `predicted_D + predicted_LOS`, and `oracle_D + oracle_LOS` comparison paths.
- Joint-consistent and joint-generative predictors.
- Outcome-aware single-run and stage2 training paths.
- Risk-head registry/resolver/override suite.
- LOS coarse, focal, ordinal, hybrid, 6-bin/9-bin related code.
- Drift diagnostics: `P(D | LOS)`, JS divergence, `dV_D`, risk-head analysis, and downstream diagnostics.

## Mixed File Patch Details

No Forecasted mixed implementation file was reduced in this split phase. The repository was kept preservation-first as requested, including retrospective/oracle/baseline branches that may support forecasted research reproduction.

## Unresolved Uncertain Files

See `SPLIT_UNCERTAIN_FILES.md`.

- Original configs and Optuna launchers retained for oracle/baseline comparison review.
- Mixed exploratory analysis notebooks, CSVs, PNGs, and small reports.
- Generated explanation outputs retained only when small.
- `ㅁㄴㅇㄹ.md` scratch note.

## Large Artifact Verification

Commands:

```bash
find . -type f -size +50M -print
find . -type f -size +10M -print
```

Result: no files reported.

Excluded artifact directories (`runs/`, `checkpoints/`, `artifacts/`, `wandb/`) are absent. Raw/processed data payloads are absent; only `src/data/raw/README_data.md` remains as a data reference note.

## Validation Commands And Results

```bash
python -m compileall src
```

Result: passed.

```bash
pytest -q
```

Result: failed, `2 failed, 205 passed, 1 warning`.

Failed tests:

- `tests/test_outcome_aware_stage2.py::test_resolve_joint_forecast_contract_matches_current_canonical_12_head_order`
- `tests/test_top3_fallback_ablation.py::test_resolve_variant_head_pairs_top5_uses_diagnostics_ranking`

Failure summary:

- `FileNotFoundError: src/data/raw/TEDS_Discharge.csv`
- `ValueError: Could not resolve top5 drift heads from diagnostics files under runs/diagnostics/forecast_cache_alignment/distribution_diagnosis`

Cause estimate:

- Both failures depend on raw data or generated diagnostics artifacts that were intentionally excluded from the split repository by policy.

Fix status:

- Not patched or skipped in this split. The failures are recorded without workaround.

```bash
python -c "import src; print('import ok')"
```

Result: passed, printed `import ok`.

```bash
python src/main.py --help
```

Result: passed, forecasted/outcome-aware CLI help displayed.

## Failed Tests And Causes

The two pytest failures above remain unresolved because fixing them requires either including excluded raw/generated artifacts or redesigning the tests to use small fixtures. That should be handled in a later cleanup phase with an explicit test-data policy.

## Follow-Up TODO

- Decide whether raw-data-dependent and diagnostics-artifact-dependent tests should be converted to fixture-backed tests or marked as integration tests.
- Cleanup Forecasted repo scope in a separate phase after confirming which retrospective/oracle/baseline paths are still needed.
- Review uncertain exploratory outputs and reports before publication.
