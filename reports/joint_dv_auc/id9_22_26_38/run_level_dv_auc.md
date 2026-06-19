# Joint dV_D / Downstream AUC Analysis

- analysis timestamp: 2026-06-14T17:48:00.050348+00:00
- included run IDs: 9, 22, 26, 38
- split: test
- requested LOS bins: coarse6
- created files: /Users/jeong-yunseong/Documents/programming/Phase_2_public/reports/joint_dv_auc/id9_22_26_38/run_level_dv_auc.csv, /Users/jeong-yunseong/Documents/programming/Phase_2_public/reports/joint_dv_auc/id9_22_26_38/head_level_dv_auc.csv, /Users/jeong-yunseong/Documents/programming/Phase_2_public/reports/joint_dv_auc/id9_22_26_38/run_level_dv_auc.md, /Users/jeong-yunseong/Documents/programming/Phase_2_public/reports/joint_dv_auc/id9_22_26_38/manifest.json

## Sorted by downstream_test_auc descending
| id | predictor_run_name | mean_dV_D | max_dV_D | mean_abs_dV_D | mean_js_D | downstream_test_auc | downstream_test_f1 | downstream_test_acc | downstream_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 38 | 20260517-174939__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0162905 | 0.056595 | 0.0165042 | 0.00390508 | 0.887348 | 0.766507 | 0.799173 | fold0_only |
| 9 | 20260516-125415__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0206269 | 0.0579056 | 0.0206269 | 0.0036866 | 0.886691 | 0.765546 | 0.798968 | fold0_only |
| 22 | 20260517-084644__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0136339 | 0.0506461 | 0.0142578 | 0.00343455 | 0.886669 | 0.766658 | 0.799288 | fold0_only |
| 26 | 20260517-085112__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0141104 | 0.0476947 | 0.0147292 | 0.00356481 | 0.886004 | 0.765789 | 0.797769 | fold0_only |

## Sorted by mean_dV_D ascending
| id | predictor_run_name | mean_dV_D | max_dV_D | mean_abs_dV_D | mean_js_D | downstream_test_auc | downstream_test_f1 | downstream_test_acc | downstream_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 22 | 20260517-084644__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0136339 | 0.0506461 | 0.0142578 | 0.00343455 | 0.886669 | 0.766658 | 0.799288 | fold0_only |
| 26 | 20260517-085112__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0141104 | 0.0476947 | 0.0147292 | 0.00356481 | 0.886004 | 0.765789 | 0.797769 | fold0_only |
| 38 | 20260517-174939__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0162905 | 0.056595 | 0.0165042 | 0.00390508 | 0.887348 | 0.766507 | 0.799173 | fold0_only |
| 9 | 20260516-125415__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0206269 | 0.0579056 | 0.0206269 | 0.0036866 | 0.886691 | 0.765546 | 0.798968 | fold0_only |

## Sorted by max_dV_D descending
| id | predictor_run_name | mean_dV_D | max_dV_D | mean_abs_dV_D | mean_js_D | downstream_test_auc | downstream_test_f1 | downstream_test_acc | downstream_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 20260516-125415__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0206269 | 0.0579056 | 0.0206269 | 0.0036866 | 0.886691 | 0.765546 | 0.798968 | fold0_only |
| 38 | 20260517-174939__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0162905 | 0.056595 | 0.0165042 | 0.00390508 | 0.887348 | 0.766507 | 0.799173 | fold0_only |
| 22 | 20260517-084644__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0136339 | 0.0506461 | 0.0142578 | 0.00343455 | 0.886669 | 0.766658 | 0.799288 | fold0_only |
| 26 | 20260517-085112__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | 0.0141104 | 0.0476947 | 0.0147292 | 0.00356481 | 0.886004 | 0.765789 | 0.797769 | fold0_only |

## Top dV_D Heads per Run
| id | head | dV_D | abs_dV_D | js_D | dV_LOS | acc |
| --- | --- | --- | --- | --- | --- | --- |
| 9 | FREQ_ATND_SELF_HELP_D | 0.0579056 | 0.0579056 | 0.0177583 | 0.0436196 | 0.758217 |
| 9 | SUB1_D | 0.036197 | 0.036197 | 0.00400734 | 0.0536327 | 0.883326 |
| 9 | EMPLOY_D | 0.0311679 | 0.0311679 | 0.00264827 | 0.0144601 | 0.786693 |
| 9 | LIVARAG_D | 0.0298482 | 0.0298482 | 0.000912862 | 0.0629219 | 0.850857 |
| 9 | DETNLF_D | 0.0220051 | 0.0220051 | 0.00429619 | 0.0364652 | 0.893464 |
| 22 | FREQ_ATND_SELF_HELP_D | 0.0506461 | 0.0506461 | 0.0186742 | 0.0509207 | 0.758814 |
| 22 | SUB1_D | 0.0341879 | 0.0341879 | 0.00403062 | 0.0508692 | 0.884459 |
| 22 | DETNLF_D | 0.0190167 | 0.0190167 | 0.00248916 | 0.0347394 | 0.893693 |
| 22 | FREQ1_D | 0.0178466 | 0.0178466 | 0.00412814 | 0.0211304 | 0.807188 |
| 22 | FREQ2_D | 0.00861773 | 0.00861773 | 0.000349485 | 0.0267961 | 0.952554 |
| 26 | FREQ_ATND_SELF_HELP_D | 0.0476947 | 0.0476947 | 0.0175256 | 0.0422975 | 0.75824 |
| 26 | SUB1_D | 0.0374957 | 0.0374957 | 0.00545601 | 0.0510256 | 0.884292 |
| 26 | DETNLF_D | 0.0168568 | 0.0168568 | 0.00311843 | 0.0326506 | 0.893913 |
| 26 | FREQ1_D | 0.0156181 | 0.0156181 | 0.00514955 | 0.0190802 | 0.807939 |
| 26 | EMPLOY_D | 0.0132843 | 0.0132843 | 0.00231146 | 0.00991456 | 0.786965 |
| 38 | FREQ_ATND_SELF_HELP_D | 0.056595 | 0.056595 | 0.0179216 | 0.0409447 | 0.756203 |
| 38 | SUB1_D | 0.0420219 | 0.0420219 | 0.005963 | 0.0583305 | 0.883876 |
| 38 | FREQ1_D | 0.0284829 | 0.0284829 | 0.00703688 | 0.0268536 | 0.805271 |
| 38 | DETNLF_D | 0.0125662 | 0.0125662 | 0.00198282 | 0.0406279 | 0.892402 |
| 38 | FREQ3_D | 0.0105631 | 0.0105631 | 0.000827661 | 0.0231426 | 0.95537 |

## Correlations
| metric | n | pearson_vs_test_auc | spearman_vs_test_auc |
| --- | --- | --- | --- |
| mean_dV_D | 4 | 0.295287 | 0.6 |
| max_dV_D | 4 | 0.760725 | 0.8 |
| mean_js_D | 4 | 0.700874 | 0.8 |
| mean_abs_dV_D | 4 | 0.266566 | 0.6 |

## dV_D Improves but AUC Does Not
| lower_dv_id | higher_dv_id | lower_mean_dV_D | higher_mean_dV_D | lower_test_auc | higher_test_auc | delta_mean_dV_D | delta_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 22 | 9 | 0.0136339 | 0.0206269 | 0.886669 | 0.886691 | -0.00699304 | -2.25443e-05 |
| 22 | 38 | 0.0136339 | 0.0162905 | 0.886669 | 0.887348 | -0.00265666 | -0.000679299 |
| 26 | 9 | 0.0141104 | 0.0206269 | 0.886004 | 0.886691 | -0.00651652 | -0.000687051 |
| 26 | 38 | 0.0141104 | 0.0162905 | 0.886004 | 0.887348 | -0.00218014 | -0.00134381 |

## Interpretation Helper
- dV_D down + AUC up: joint regularization likely useful.
- dV_D down + AUC unchanged: joint target may be wrong or downstream insensitive.
- dV_D unchanged + AUC unchanged: regularization likely ineffective.
- dV_D up + AUC up: dV_D is not the relevant bottleneck for that run.

## File Paths Used
| id | prediction_path | predictor_run_dir | downstream_dirs |
| --- | --- | --- | --- |
| 9 | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260516-125415__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1/test_predictions.csv | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260516-125415__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260519-030924__ctmp_gin_joint_fresh_id9__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2 |
| 22 | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260517-084644__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1/test_predictions.csv | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260517-084644__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260519-030905__ctmp_gin_joint_fresh_id22__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3 |
| 26 | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260517-085112__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1/test_predictions.csv | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260517-085112__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260519-041725__ctmp_gin_joint_fresh_id26__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3 |
| 38 | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260517-174939__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1/test_predictions.csv | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260517-174939__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1 | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260519-153542__ctmp_gin_joint_fresh_id38__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3 |

## Warnings
- downstream map: missing /Users/jeong-yunseong/Documents/programming/Phase_2_public/configs/analysis/downstream_run_map.csv
