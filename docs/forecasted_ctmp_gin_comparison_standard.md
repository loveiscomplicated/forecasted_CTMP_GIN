# Forecasted CTMP-GIN Comparison Standard

## 1. Purpose

This document defines fixed comparison criteria for future Forecasted CTMP-GIN experiments. It is intended to keep future runs comparable on downstream quality, LOS routing behavior, long-stay recovery, and structured drift rather than judging them on AUC alone.

Future runs should be compared against the following references:

| Reference | Meaning |
| --- | --- |
| `id26_coarse6` | Existing joint-consistent coarse6 baseline |
| `id26_9bin_breakdown` | LOS breakdown diagnostic reference |
| `future joint-generative / outcome-aware` | Candidate solution class |

This document is a comparison standard, not a training recipe.

## 2. Fixed reference runs

| run_name | run_type | fold | best_epoch | valid_auc | test_auc | test_acc | test_f1 | test_precision | test_recall | test_loss | run_dir | role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `id26_coarse6` | `coarse6_baseline` | 0 | 44 | 0.8841430523841003 | `nan` | `nan` | `nan` | `nan` | `nan` | `nan` | `runs/20260519-041725__ctmp_gin_joint_fresh_id26__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3/folds/fold_0` | coarse6 baseline reference |
| `id26_9bin_breakdown` | `breakdown9_target` | 0 | 24 | 0.8855004628310033 | 0.8877072061816057 | 0.7989466050091911 | 0.7638836769027731 | 0.7984979240675935 | 0.7321457437522227 | 0.40867464399967784 | `runs/20260614-141133__ctmp_gin_joint_fresh_id26_breakdown__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0` | diagnostic reference, not main solution |
| `future joint-generative / outcome-aware` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | candidate solution |

## 3. Required comparison metrics

Every future candidate must be compared using at least the following groups.

### A. Overall downstream metrics

Required:

- `valid_auc`
- `test_auc`
- `test_acc`
- `test_f1`
- `test_precision`
- `test_recall`
- `test_loss`
- `best_epoch`
- `fold / CV status`

Interpretation:

- AUC alone is insufficient.
- A fold-0-only gain below about 0.002 should not be treated as meaningful unless supported by diagnostic improvements.
- Full 5-fold CV is required before claiming a stable improvement.

### B. Minimum success threshold

Current success hierarchy:

| Level | Criterion | Interpretation |
| --- | --- | --- |
| Failure | Forecasted CTMP-GIN <= Admission-only GIN/MLP | predicted future graph adds no value |
| Weak signal | Forecasted CTMP-GIN slightly above admission-only, but diagnostics unchanged | likely noise or superficial improvement |
| Meaningful improvement | AUC improves and at least one structural diagnostic improves | possible real progress |
| Strong improvement | AUC approaches single-anchor runs around 0.93 and diagnostics improve | joint mismatch substantially reduced |
| Upper-bound target | approaches Oracle CTMP-GIN around 0.955 | near-oracle future transition recovery |

Key rule:

A candidate is not considered a solution if it only improves AUC marginally while long-stay recall and `dV_D` remain unchanged.

### C. LOS routing diagnostics

Required:

- true LOS confusion matrix
- predicted LOS confusion matrix
- middle-to-long flow:
- true `8-14` -> predicted long bins
- true `15-21` -> predicted long bins
- true `22-28` -> predicted long bins
- long-stay flow split:
- `29-31`
- `32-33`
- `34-35`
- `36-37`

For `id26_9bin_breakdown`, record the known diagnostic:

- middle-to-long flow decreased by about `0.1020`
- this indicates less long-stay sink behavior
- however, this did not translate into long-stay recall recovery

### D. True LOS-bin outcome metrics

Required per true LOS bin:

- support
- positive_count if available
- recall
- f1
- auc

Special focus:

- `29-31`
- `32-33`
- `34-35`
- `36-37`

Long-stay recall is a critical diagnostic because previous analysis showed that later long-stay bins remain weak even after 9-bin breakdown.

For `id26_9bin_breakdown`, the known pattern is:

- `29-31` recall: `0.753550`
- `32-33` recall: `0.623770`
- `34-35` recall: `0.516877`
- `36-37` recall: `0.402085`

Interpretation:

- If only `29-31` and `32-33` improve while `34-35` and `36-37` remain weak, the run has not solved long-stay degradation.
- A real solution should improve the later long-stay bins, especially `34-35` and `36-37`.

### E. Predicted LOS-bin outcome metrics

Required per predicted LOS bin:

- support
- recall
- f1
- auc

Reason:

- This detects population contamination inside predicted LOS groups.
- If predicted long bins have low recall or degraded AUC, the forecasted LOS group is semantically different from the oracle LOS group.

### F. Drift decomposition

Required:

- `mean_dV_D`
- `max_dV_D`
- `mean_abs_dV_D`
- `mean_js_D`
- `max_js_D`
- `mean_dV_LOS`
- `max_dV_LOS`
- top `dV_D` heads
- top `js_D` heads

Use the three-table decomposition:

```text
T_oracle = table(true_D, true_LOS)
T_mid    = table(true_D, pred_LOS)
T_full   = table(pred_D, pred_LOS)

dV_LOS = V(T_mid) - V(T_oracle)
dV_D   = V(T_full) - V(T_mid)
```

Explanation:

- `dV_LOS` measures LOS grouping / routing artifact.
- `dV_D` measures D-error-attributable structured mismatch.
- `SERVICES_D` can show large total drift while having low `dV_D` because its prediction is nearly correct and the drift is mostly LOS-side grouping.
- `FREQ_ATND_SELF_HELP_D`, `SUB1_D`, `FREQ1_D`, `FREQ2_D`, `EMPLOY_D`, `DETNLF_D` are more important candidates if they carry high `dV_D`.

For `id26_9bin_breakdown`:

- coarse6-comparable `dV_D` did not decrease
- delta `dV_D` was approximately `+0.000541`
- therefore, LOS breakdown reduced sink behavior but not D-side structured mismatch

## 4. Required comparison table template

| run_name | run_type | fold_status | valid_auc | test_auc | delta_vs_id26_coarse6_valid_auc | delta_vs_9bin_test_auc | long_stay_recall_mean | recall_29_31 | recall_32_33 | recall_34_35 | recall_36_37 | middle_to_long_flow | delta_middle_to_long_flow | mean_dV_D | max_dV_D | top1_dV_D_head | top1_dV_D | top2_dV_D_head | top2_dV_D | top3_dV_D_head | top3_dV_D | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `id26_coarse6` | `coarse6_baseline` | `fold_0_only; 5-fold required for stable claim` | 0.8841430523841003 | `nan` | 0.0000000000000000 | `nan` | 0.5749344484241915 | `nan` | `nan` | `nan` | `nan` | 0.5248201022932975 | 0.0000000000000000 | 0.023285171277448207 | 0.07398458167626171 | `FREQ_ATND_SELF_HELP_D` | 0.07398458167626171 | `SUB1_D` | 0.05489672667983442 | `EMPLOY_D` | 0.033293414024114815 | aggregate long-stay bin `29-37` is available, but later long-stay bins are unresolved |
| `id26_9bin_breakdown` | `breakdown9_target` | `fold_0 diagnostic run; 5-fold not yet justified` | 0.8855004628310033 | 0.8877072061816057 | 0.0013574104469030 | 0.0000000000000000 | 0.5740703158430116 | 0.7535496957403651 | 0.6237697307335190 | 0.5168772508087652 | 0.4020845860893967 | 0.4228635121058670 | -0.1019565901874305 | 0.023825749967486907 | 0.06967007892265054 | `FREQ_ATND_SELF_HELP_D` | 0.06967007892265054 | `SUB1_D` | 0.05534128809085645 | `FREQ2_D` | 0.03132280748199033 | reduced LOS sink behavior, but long-stay recovery and coarse6-comparable `dV_D` did not improve enough |
| `future_joint_generative_or_outcome_aware` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

Notes:

- `id26_coarse6` has missing test metrics in the currently available diagnostic artifact; keep those fields as `nan` until a stable artifact is available.
- `mean_dV_D` above uses the coarse6-comparable view for direct comparison between `id26_coarse6` and `id26_9bin_breakdown`.

## 5. Decision rules

### Rule 1: AUC-only improvement is insufficient

If:

- `test_auc` improves by less than `0.002`
- and long-stay recall does not improve
- and `dV_D` does not decrease

Then:

- classify as `diagnostic-only improvement`
- do not proceed to full CV unless there is another strong reason

### Rule 2: LOS-only improvement is insufficient

If:

- middle-to-long flow decreases
- but long-stay recall and `dV_D` do not improve

Then:

- classify as `LOS routing artifact partially reduced`
- do not treat as solution

This is exactly the interpretation for `id26_9bin_breakdown`.

### Rule 3: dV_D decrease must be downstream-relevant

If:

- `dV_D` decreases
- but downstream AUC and long-stay recall do not improve

Then:

- classify as `drift proxy improved but downstream insensitive`
- investigate whether the wrong heads were targeted

### Rule 4: Candidate solution threshold

A future joint-generative / outcome-aware run becomes a serious candidate only if it satisfies at least two of:

- `test_auc` clearly exceeds `0.8877` on fold 0
- valid/test AUC exceeds admission-only by a nontrivial margin
- long-stay recall improves, especially `34-35` and `36-37`
- middle-to-long flow does not worsen
- `mean_dV_D` or top-head `dV_D` decreases
- top `dV_D` heads shift downward, especially `FREQ_ATND_SELF_HELP_D` / `SUB1_D` / `FREQ1_D`

### Rule 5: CV promotion threshold

Promote a run to full 5-fold CV only if:

- fold-0 AUC improvement is not merely marginal
- and at least one structural diagnostic improves
- and no key diagnostic gets worse

Otherwise, keep it as a diagnostic run.

## 6. Interpretation of the current 9-bin result

The `id26_9bin_breakdown` run should be treated as a diagnostic reference. It showed that splitting long LOS into finer bins can reduce middle-to-long sink behavior, but it did not recover long-stay recall and did not reduce D-side structured mismatch. Therefore, future models should not merely aim to refine LOS binning. They must reduce the joint mismatch between predicted `_D` and predicted LOS.

## 7. Future candidate section

For every new candidate, fill:

- `run_dir:`
- `predictor type:`
- `LOS target scheme:`
- `D target scheme:`
- `whether outcome-aware loss is used:`
- `whether CTMP-GIN is frozen:`
- `whether predictor is trained end-to-end:`
- `valid_auc:`
- `test_auc:`
- `long-stay recall:`
- `middle-to-long flow:`
- `dV_D summary:`
- `top dV_D heads:`
- `comparison vs id26_coarse6:`
- `comparison vs id26_9bin_breakdown:`
- `decision:`
- `reject`
- `diagnostic-only`
- `needs more analysis`
- `promote to 5-fold CV`

## 8. Optional script hook

This document is intended to be used together with the future outputs of:

- `scripts/build_joint_dv_auc_table.py`
- any LOS-bin outcome diagnostic script
