# Executive Summary

1. Did structured loss improve downstream AUC beyond the old plateau?
- Best structured test AUC is 0.887949; plateau reference is 0.8877.

2. Did structured loss reduce post-hoc dV_D/js_D?
- struct_top6_l001: delta risk mean dV_D=-0.00769598, delta risk js_D=-0.0013962.

3. Did structured loss improve long-stay recall, especially 34-35 and 36-37?
- unavailable or incomplete for late long-stay bins.

4. Did structured loss improve or harm risk-head D accuracy?
- no risk-head accuracy degradation detected versus baseline in test/coarse6.

5. Failure diagnosis:
- post-hoc dV_D improved but AUC stayed plateau-level; proxy appears downstream-insensitive or insufficient alone.

# Comparison Table

| run_name                | valid_auc | test_auc | mean_risk_dV_D | max_risk_dV_D | mean_risk_js_D | long_stay_recall_mean | late_long_recall_mean | middle_to_long_flow | struct_loss_delta | interpretation                                                                 |
| ----------------------- | --------- | -------- | -------------- | ------------- | -------------- | --------------------- | --------------------- | ------------------- | ----------------- | ------------------------------------------------------------------------------ |
| baseline_id26           | 0.884143  | 0.886004 | 0.0400371      | 0.0739846     | 0.00835001     | nan                   | nan                   | 0.525201            | nan               | baseline reference                                                             |
| struct_top6_l001        | 0.885556  | 0.887949 | 0.0323411      | 0.0567874     | 0.0069538      | nan                   | nan                   | 0.513913            | -0.00154905       | dV_D effect must be judged without long-stay recall; AUC remains plateau-level |
| struct_robust_top3_l003 | 0.885447  | 0.887285 | nan            | nan           | nan            | nan                   | nan                   | nan                 | -0.00152831       | post-hoc drift unavailable; cannot classify dV_D effect                        |

# Diagnostic Decision Rules

- A. struct_loss did not decrease -> implementation/config/loss-scale issue.
- B. struct_loss decreased but post-hoc dV_D did not decrease -> batch-local soft js_D surrogate is not aligned with hard post-hoc drift.
- C. post-hoc dV_D decreased but AUC did not improve -> dV_D proxy is not sufficient for downstream recovery.
- D. AUC improved but long-stay recall did not -> possible shortcut, not true semantic recovery.
- E. AUC, dV_D, and late long-stay recall all improved -> structured-loss direction is supported.
- F. none improved -> soft js_D surrogate v1 should be considered failed.

# Recommendation

Do not combine with 9-bin yet. Treat v1 as inconclusive on long-stay recovery and prioritize outcome-aware fine-tuning or a joint-generative predictor.

# Warnings

- baseline_id26 per-row downstream outcome predictions: missing /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260519-041725__ctmp_gin_joint_fresh_id26__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3/folds/fold_0/diagnostic_predictions.csv
- struct_top6_l001 per-row downstream outcome predictions: missing /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260615-061835__ctmp_gin_joint_fresh_id26_struct_dvD_top6_lambda001__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0/diagnostic_predictions.csv
- struct_robust_top3_l003 per-row downstream outcome predictions: missing /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260615-061634__ctmp_gin_joint_fresh_id26_struct_robust_top3_lambda001__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0/diagnostic_predictions.csv
- struct_robust_top3_l003 joint predictions valid: missing /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260615-061634__ctmp_gin_joint_fresh_id26_struct_robust_top3_lambda001__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0/joint_predictor/val_predictions.csv
- struct_robust_top3_l003 joint predictions test: missing /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260615-061634__ctmp_gin_joint_fresh_id26_struct_robust_top3_lambda001__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0/joint_predictor/test_predictions.csv
