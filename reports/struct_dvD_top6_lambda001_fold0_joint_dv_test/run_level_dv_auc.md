# Joint dV_D / Downstream AUC Analysis

- analysis timestamp: 2026-06-15T08:48:44.509541+00:00
- included run IDs: 1, 2
- split: test
- requested LOS bins: coarse6
- created files: /Users/jeong-yunseong/Documents/programming/Phase_2_public/reports/struct_dvD_top6_lambda001_fold0_joint_dv_test/run_level_dv_auc.csv, /Users/jeong-yunseong/Documents/programming/Phase_2_public/reports/struct_dvD_top6_lambda001_fold0_joint_dv_test/head_level_dv_auc.csv, /Users/jeong-yunseong/Documents/programming/Phase_2_public/reports/struct_dvD_top6_lambda001_fold0_joint_dv_test/run_level_dv_auc.md, /Users/jeong-yunseong/Documents/programming/Phase_2_public/reports/struct_dvD_top6_lambda001_fold0_joint_dv_test/manifest.json

## Sorted by downstream_test_auc descending
| id | predictor_run_name | mean_dV_D | max_dV_D | mean_abs_dV_D | mean_js_D | downstream_test_auc | downstream_test_f1 | downstream_test_acc | downstream_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | runs/20260519-041725__ctmp_gin_joint_fresh_id26__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3/folds/fold_0 | 0.0232852 | 0.0739846 | 0.0232852 | 0.00512728 |  |  |  | missing |
| 2 | runs/20260615-061835__ctmp_gin_joint_fresh_id26_struct_dvD_top6_lambda001__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0 | 0.0190154 | 0.0567874 | 0.01947 | 0.00446505 |  |  |  | missing |

## Sorted by mean_dV_D ascending
| id | predictor_run_name | mean_dV_D | max_dV_D | mean_abs_dV_D | mean_js_D | downstream_test_auc | downstream_test_f1 | downstream_test_acc | downstream_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | runs/20260615-061835__ctmp_gin_joint_fresh_id26_struct_dvD_top6_lambda001__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0 | 0.0190154 | 0.0567874 | 0.01947 | 0.00446505 |  |  |  | missing |
| 1 | runs/20260519-041725__ctmp_gin_joint_fresh_id26__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3/folds/fold_0 | 0.0232852 | 0.0739846 | 0.0232852 | 0.00512728 |  |  |  | missing |

## Sorted by max_dV_D descending
| id | predictor_run_name | mean_dV_D | max_dV_D | mean_abs_dV_D | mean_js_D | downstream_test_auc | downstream_test_f1 | downstream_test_acc | downstream_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | runs/20260519-041725__ctmp_gin_joint_fresh_id26__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3/folds/fold_0 | 0.0232852 | 0.0739846 | 0.0232852 | 0.00512728 |  |  |  | missing |
| 2 | runs/20260615-061835__ctmp_gin_joint_fresh_id26_struct_dvD_top6_lambda001__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0 | 0.0190154 | 0.0567874 | 0.01947 | 0.00446505 |  |  |  | missing |

## Top dV_D Heads per Run
| id | head | dV_D | abs_dV_D | js_D | dV_LOS | acc |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | FREQ_ATND_SELF_HELP_D | 0.0739846 | 0.0739846 | 0.0235941 | 0.0472867 | 0.744947 |
| 1 | SUB1_D | 0.0548967 | 0.0548967 | 0.00924215 | 0.0596769 | 0.882225 |
| 1 | EMPLOY_D | 0.0332934 | 0.0332934 | 0.00444499 | 0.0123358 | 0.782981 |
| 1 | FREQ2_D | 0.0298824 | 0.0298824 | 0.000384092 | 0.0363017 | 0.948481 |
| 1 | FREQ1_D | 0.0275903 | 0.0275903 | 0.00765956 | 0.0235754 | 0.801483 |
| 1 | DETNLF_D | 0.0205754 | 0.0205754 | 0.00477512 | 0.0368465 | 0.889624 |
| 1 | FREQ3_D | 0.0118921 | 0.0118921 | 0.00120552 | 0.00580834 | 0.951615 |
| 1 | SUB2_D | 0.0111029 | 0.0111029 | 0.00111014 | 0.0522446 | 0.902022 |
| 2 | FREQ_ATND_SELF_HELP_D | 0.0567874 | 0.0567874 | 0.0227878 | 0.0441833 | 0.750004 |
| 2 | SUB1_D | 0.0353335 | 0.0353335 | 0.00338426 | 0.0604523 | 0.883835 |
| 2 | FREQ2_D | 0.0311957 | 0.0311957 | 0.000417365 | 0.036409 | 0.949542 |
| 2 | FREQ1_D | 0.0298056 | 0.0298056 | 0.00839726 | 0.0263636 | 0.803287 |
| 2 | DETNLF_D | 0.0236536 | 0.0236536 | 0.00442368 | 0.033026 | 0.890599 |
| 2 | FREQ3_D | 0.0184023 | 0.0184023 | 0.00162433 | 0.00505316 | 0.952368 |
| 2 | EMPLOY_D | 0.017271 | 0.017271 | 0.0023124 | 0.0138745 | 0.78509 |
| 2 | SUB2_D | 0.00795308 | 0.00795308 | 0.00125835 | 0.0564475 | 0.903087 |

## Correlations
| metric | n | pearson_vs_test_auc | spearman_vs_test_auc |
| --- | --- | --- | --- |
| mean_dV_D | 0 |  |  |
| max_dV_D | 0 |  |  |
| mean_js_D | 0 |  |  |
| mean_abs_dV_D | 0 |  |  |

## dV_D Improves but AUC Does Not
| lower_dv_id | higher_dv_id | delta_mean_dV_D | delta_auc |
| --- | --- | --- | --- |
|  |  |  |  |

## Interpretation Helper
- dV_D down + AUC up: joint regularization likely useful.
- dV_D down + AUC unchanged: joint target may be wrong or downstream insensitive.
- dV_D unchanged + AUC unchanged: regularization likely ineffective.
- dV_D up + AUC up: dV_D is not the relevant bottleneck for that run.

## File Paths Used
| id | prediction_path | predictor_run_dir | downstream_dirs |
| --- | --- | --- | --- |
| 1 | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260519-041725__ctmp_gin_joint_fresh_id26__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3/folds/fold_0/joint_predictor/test_predictions.csv | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260519-041725__ctmp_gin_joint_fresh_id26__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3/folds/fold_0 |  |
| 2 | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260615-061835__ctmp_gin_joint_fresh_id26_struct_dvD_top6_lambda001__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0/joint_predictor/test_predictions.csv | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260615-061835__ctmp_gin_joint_fresh_id26_struct_dvD_top6_lambda001__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0 |  |

## Warnings
- none
