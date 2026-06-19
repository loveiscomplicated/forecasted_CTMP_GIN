# LOS Breakdown Downstream Diagnostic

## Executive summary
- The new test AUC is 0.8877 versus nan.
- Long-stay recall did not improve; delta=-0.0009.
- Middle-to-long flow decreased by 0.1020, indicating less long-stay sink behavior.
- dV_D did not decrease on the coarse6-comparable view; delta=0.000541.
- Conclusion: 9-bin is better interpreted as a diagnostic probe than a main solution unless longer-stay behavior improves consistently.

## Run paths and artifacts used
- `id26_9bin_breakdown`: `/Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260614-141133__ctmp_gin_joint_fresh_id26_breakdown__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0`
- `id26_coarse6`: `/Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260519-041725__ctmp_gin_joint_fresh_id26__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3/folds/fold_0`

## Overall downstream metric comparison
| run_name            | run_type          | best_epoch | best_valid_metric | valid_auc | valid_acc | valid_f1 | valid_precision | valid_recall | test_auc | test_acc | test_f1  | test_precision | test_recall | test_loss | fold | run_dir                                                                                                                                                                         | status     |
| ------------------- | ----------------- | ---------- | ----------------- | --------- | --------- | -------- | --------------- | ------------ | -------- | -------- | -------- | -------------- | ----------- | --------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| id26_9bin_breakdown | breakdown9_target | 24         | 0.885500          | 0.885500  | 0.796281  | 0.761511 | 0.793065        | 0.732373     | 0.887707 | 0.798947 | 0.763884 | 0.798498       | 0.732146    | 0.408675  | 0    | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260614-141133__ctmp_gin_joint_fresh_id26_breakdown__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2/folds/fold_0  | completed  |
| id26_coarse6        | coarse6_baseline  | 44         | 0.884143          | 0.884143  | 0.795156  | 0.763599 | 0.783197        | 0.744959     | nan      | nan      | nan      | nan            | nan         | nan       | 0    | /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/20260519-041725__ctmp_gin_joint_fresh_id26__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2_lambda=0.3/folds/fold_0 | fold0_only |

## LOS confusion matrix summary
| run_name            | true_bin_label | pred_bin_label | count |
| ------------------- | -------------- | -------------- | ----- |
| id26_9bin_breakdown | 1              | 1              | 35500 |
| id26_9bin_breakdown | 1              | 15-21          | 78    |
| id26_9bin_breakdown | 1              | 2-7            | 5531  |
| id26_9bin_breakdown | 1              | 22-28          | 416   |
| id26_9bin_breakdown | 1              | 29-31          | 302   |
| id26_9bin_breakdown | 1              | 32-33          | 987   |
| id26_9bin_breakdown | 1              | 34-35          | 742   |
| id26_9bin_breakdown | 1              | 36-37          | 1460  |
| id26_9bin_breakdown | 1              | 8-14           | 131   |
| id26_9bin_breakdown | 15-21          | 1              | 1988  |
| id26_9bin_breakdown | 15-21          | 15-21          | 774   |
| id26_9bin_breakdown | 15-21          | 2-7            | 2065  |
| id26_9bin_breakdown | 15-21          | 22-28          | 1926  |
| id26_9bin_breakdown | 15-21          | 29-31          | 930   |
| id26_9bin_breakdown | 15-21          | 32-33          | 2526  |
| id26_9bin_breakdown | 15-21          | 34-35          | 1292  |
| id26_9bin_breakdown | 15-21          | 36-37          | 1849  |
| id26_9bin_breakdown | 15-21          | 8-14           | 603   |
| id26_9bin_breakdown | 2-7            | 1              | 5207  |
| id26_9bin_breakdown | 2-7            | 15-21          | 362   |
| id26_9bin_breakdown | 2-7            | 2-7            | 36430 |
| id26_9bin_breakdown | 2-7            | 22-28          | 1947  |
| id26_9bin_breakdown | 2-7            | 29-31          | 1189  |
| id26_9bin_breakdown | 2-7            | 32-33          | 2822  |

## Middle-to-long LOS misassignment
| run_name            | true_bin | true_bin_label | support | correct_count | accuracy | to_29_31_count | to_29_31_pct | to_32_33_count | to_32_33_pct | to_34_35_count | to_34_35_pct | to_36_37_count | to_36_37_pct | total_to_long_count | total_to_long_pct | top_pred_bin | top_pred_pct |
| ------------------- | -------- | -------------- | ------- | ------------- | -------- | -------------- | ------------ | -------------- | ------------ | -------------- | ------------ | -------------- | ------------ | ------------------- | ----------------- | ------------ | ------------ |
| id26_9bin_breakdown | 2        | 8-14           | 19202   | 1036          | 0.053953 | 1031.000000    | 0.053692     | 2690.000000    | 0.140090     | 1414.000000    | 0.073638     | 1905.000000    | 0.099208     | 7040                | 0.366628          | 1            | 0.315957     |
| id26_9bin_breakdown | 3        | 15-21          | 13953   | 774           | 0.055472 | 930.000000     | 0.066652     | 2526.000000    | 0.181036     | 1292.000000    | 0.092597     | 1849.000000    | 0.132516     | 6597                | 0.472802          | 6            | 0.181036     |
| id26_9bin_breakdown | 4        | 22-28          | 16248   | 5002          | 0.307853 | 1671.000000    | 0.102843     | 2395.000000    | 0.147403     | 1225.000000    | 0.075394     | 1682.000000    | 0.103520     | 6973                | 0.429161          | 4            | 0.307853     |
| id26_coarse6        | 2        | 8-14           | 19202   | 464           | 0.024164 | nan            | nan          | nan            | nan          | nan            | nan          | nan            | nan          | 8525                | 0.443964          | 5            | 0.443964     |
| id26_coarse6        | 3        | 15-21          | 13953   | 289           | 0.020712 | nan            | nan          | nan            | nan          | nan            | nan          | nan            | nan          | 8173                | 0.585752          | 5            | 0.585752     |
| id26_coarse6        | 4        | 22-28          | 16248   | 4446          | 0.273634 | nan            | nan          | nan            | nan          | nan            | nan          | nan            | nan          | 8851                | 0.544744          | 5            | 0.544744     |

## True LOS-bin outcome metrics
| run_name            | los_bin_label | support | recall   | f1       | auc      |
| ------------------- | ------------- | ------- | -------- | -------- | -------- |
| id26_9bin_breakdown | 1             | 45147   | 0.966896 | 0.912781 | 0.987009 |
| id26_9bin_breakdown | 2-7           | 51884   | 0.888602 | 0.806367 | 0.894217 |
| id26_9bin_breakdown | 8-14          | 19202   | 0.833894 | 0.776965 | 0.888630 |
| id26_9bin_breakdown | 15-21         | 13953   | 0.821764 | 0.776782 | 0.893795 |
| id26_9bin_breakdown | 22-28         | 16248   | 0.806340 | 0.843447 | 0.903534 |
| id26_9bin_breakdown | 29-31         | 22969   | 0.753550 | 0.793726 | 0.895754 |
| id26_9bin_breakdown | 32-33         | 34942   | 0.623770 | 0.705573 | 0.850927 |
| id26_9bin_breakdown | 34-35         | 35475   | 0.516877 | 0.636357 | 0.839274 |
| id26_9bin_breakdown | 36-37         | 38708   | 0.402085 | 0.534624 | 0.824829 |

## Predicted LOS-bin outcome metrics
| run_name            | los_bin_label | support | recall   | f1       | auc      |
| ------------------- | ------------- | ------- | -------- | -------- | -------- |
| id26_9bin_breakdown | 1             | 60894   | 0.938257 | 0.958775 | 0.985262 |
| id26_9bin_breakdown | 2-7           | 55746   | 0.830942 | 0.799006 | 0.857523 |
| id26_9bin_breakdown | 8-14          | 2998    | 0.861627 | 0.782887 | 0.804701 |
| id26_9bin_breakdown | 15-21         | 2132    | 0.825994 | 0.759881 | 0.852942 |
| id26_9bin_breakdown | 22-28         | 13612   | 0.869032 | 0.781059 | 0.730548 |
| id26_9bin_breakdown | 29-31         | 10808   | 0.762706 | 0.731969 | 0.759566 |
| id26_9bin_breakdown | 32-33         | 38786   | 0.548422 | 0.615131 | 0.803931 |
| id26_9bin_breakdown | 34-35         | 35361   | 0.474536 | 0.549426 | 0.764525 |
| id26_9bin_breakdown | 36-37         | 58191   | 0.353916 | 0.473234 | 0.813876 |

## Long-stay recall comparison
| run_name            | los_scheme | bin_label | support | positive_count | recall   | f1       | auc      | delta_vs_baseline |
| ------------------- | ---------- | --------- | ------- | -------------- | -------- | -------- | -------- | ----------------- |
| id26_9bin_breakdown | breakdown9 | 29-31     | 22969   | 10846          | 0.753550 | 0.793726 | 0.895754 | 0.178615          |
| id26_9bin_breakdown | breakdown9 | 32-33     | 34942   | 16155          | 0.623770 | 0.705573 | 0.850927 | 0.048835          |
| id26_9bin_breakdown | breakdown9 | 34-35     | 35475   | 16383          | 0.516877 | 0.636357 | 0.839274 | -0.058057         |
| id26_9bin_breakdown | breakdown9 | 36-37     | 38708   | 14967          | 0.402085 | 0.534624 | 0.824829 | -0.172850         |
| id26_coarse6        | coarse6    | 29-37     | 132094  | 58351          | 0.574934 | 0.672743 | 0.844277 | nan               |

## dV_LOS / dV_D decomposition
| run_name            | los_scheme_eval | head                  | acc      | dV_LOS   | dV_D     | js_D     |
| ------------------- | --------------- | --------------------- | -------- | -------- | -------- | -------- |
| id26_9bin_breakdown | breakdown9      | FREQ_ATND_SELF_HELP_D | 0.755889 | 0.071489 | 0.070552 | 0.020982 |
| id26_9bin_breakdown | breakdown9      | SUB1_D                | 0.884908 | 0.069042 | 0.040503 | 0.009605 |
| id26_9bin_breakdown | breakdown9      | FREQ2_D               | 0.951013 | 0.031643 | 0.030375 | 0.000487 |
| id26_9bin_breakdown | breakdown9      | EMPLOY_D              | 0.787285 | 0.030028 | 0.030096 | 0.004148 |
| id26_9bin_breakdown | breakdown9      | FREQ1_D               | 0.809130 | 0.037901 | 0.024842 | 0.005645 |
| id26_9bin_breakdown | breakdown9      | DETNLF_D              | 0.892106 | 0.042461 | 0.021231 | 0.005336 |
| id26_9bin_breakdown | breakdown9      | LIVARAG_D             | 0.850159 | 0.105209 | 0.017921 | 0.001451 |
| id26_9bin_breakdown | breakdown9      | FREQ3_D               | 0.954671 | 0.014019 | 0.012537 | 0.001548 |
| id26_9bin_breakdown | breakdown9      | ARRESTS_D             | 0.875640 | 0.047439 | 0.011616 | 0.005433 |
| id26_9bin_breakdown | breakdown9      | SERVICES_D            | 0.983750 | 0.160272 | 0.009647 | 0.001122 |
| id26_9bin_breakdown | breakdown9      | SUB2_D                | 0.904816 | 0.050922 | 0.008859 | 0.001480 |
| id26_9bin_breakdown | breakdown9      | SUB3_D                | 0.959774 | 0.055001 | 0.002326 | 0.000694 |
| id26_9bin_breakdown | coarse6         | FREQ_ATND_SELF_HELP_D | 0.755889 | 0.049323 | 0.069670 | 0.019223 |
| id26_9bin_breakdown | coarse6         | SUB1_D                | 0.884908 | 0.059278 | 0.055341 | 0.009427 |
| id26_9bin_breakdown | coarse6         | FREQ2_D               | 0.951013 | 0.028031 | 0.031323 | 0.000462 |
| id26_9bin_breakdown | coarse6         | EMPLOY_D              | 0.787285 | 0.011855 | 0.026464 | 0.003742 |
| id26_9bin_breakdown | coarse6         | FREQ1_D               | 0.809130 | 0.031128 | 0.021357 | 0.005231 |
| id26_9bin_breakdown | coarse6         | DETNLF_D              | 0.892106 | 0.030716 | 0.020827 | 0.005189 |
| id26_9bin_breakdown | coarse6         | FREQ3_D               | 0.954671 | 0.005270 | 0.014012 | 0.001421 |
| id26_9bin_breakdown | coarse6         | SUB2_D                | 0.904816 | 0.051979 | 0.013144 | 0.001337 |
| id26_9bin_breakdown | coarse6         | ARRESTS_D             | 0.875640 | 0.035090 | 0.012271 | 0.005172 |
| id26_9bin_breakdown | coarse6         | SERVICES_D            | 0.983750 | 0.152601 | 0.010907 | 0.001079 |
| id26_9bin_breakdown | coarse6         | LIVARAG_D             | 0.850159 | 0.093111 | 0.009041 | 0.001156 |
| id26_9bin_breakdown | coarse6         | SUB3_D                | 0.959774 | 0.045731 | 0.001551 | 0.000640 |

## Direct coarse6 ID26 vs 9-bin ID26 conclusion
- Did 9-bin improve overall AUC enough? The new test AUC is 0.8877 versus nan.
- Did 9-bin improve long-stay recall? Long-stay recall did not improve; delta=-0.0009.
- Did 9-bin reduce middle-to-long misassignment? Middle-to-long flow decreased by 0.1020, indicating less long-stay sink behavior.
- Did dV_D decrease? dV_D did not decrease on the coarse6-comparable view; delta=0.000541.
- Is 9-bin a main solution or only a diagnostic improvement? 9-bin is better interpreted as a diagnostic probe than a main solution unless longer-stay behavior improves consistently.

## Recommended next step
- Use the 9-bin run as a diagnostic reference, but prioritize fixes that directly reduce D-side structured mismatch and long-stay sink routing together.
