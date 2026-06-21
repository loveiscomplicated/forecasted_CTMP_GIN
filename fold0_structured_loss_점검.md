• 정리하면, fold 0 기준으로 구조 loss는 제대로 켜졌고 dV_D/js_D는 줄었지만, loss weight는 매우 작았고 long-stay bin recall은 현재 artifact만으로는 판정 불가입니다.

  1. train_struct_loss 기록 여부: 기록됨
     joint_predictor/metrics.jsonl에 structured-loss row가 30개 있습니다.

  2. train_struct_loss 감소 여부: 감소함
     epoch 1: 0.00215818 → epoch 30: 0.00060913
     최저: epoch 29, 0.00060443

  3. valid_struct_loss 감소 여부: 감소함
     epoch 1: 0.00074080 → epoch 30: 0.00051639
     최저: epoch 23, 0.00051496

  4. resolved_risk_heads top6 여부: 설정상 맞음
     joint_predictor/config.final.yaml 기준:

  risk_head_set: new_dvD_top6
  joint_heads: new_dvD_top6

  top6는 다음입니다:

  FREQ_ATND_SELF_HELP_D
  SUB1_D
  FREQ2_D
  EMPLOY_D
  FREQ1_D
  DETNLF_D

  단, fold 0의 joint predictor checkpoint는 현재 남아 있지 않아서 fold 0 checkpoint metadata 안의 resolved_risk_heads는 직접 확인 불가입니다. fold 1 checkpoint metadata에는 위 top6가 정확히 저장되어 있었습니다.

  5. lambda_struct * struct_loss 크기: total loss 대비 거의 무시 가능한 수준
     lambda_struct=0.01입니다.

  epoch 1 train 기준:

  0.01 * 0.00215818 = 0.00002158
  train_loss = 8.89473
  ratio ~= 2.43e-6 = 0.000243%


  6. dV_D / js_D baseline 대비 감소 여부: 감소함
     test split, coarse6 기준:

  mean_dV_D: 0.023285 -> 0.019015  delta -0.004270
  max_dV_D:  0.073985 -> 0.056787  delta -0.017197
  mean_js_D: 0.005127 -> 0.004465  delta -0.000662
  max_js_D:  0.023594 -> 0.022788  delta -0.000806

  주요 감소는 FREQ_ATND_SELF_HELP_D, SUB1_D, EMPLOY_D에서 나왔습니다. 반면 FREQ2_D, FREQ1_D, DETNLF_D는 test 기준으로 dV_D가 약간 증가했습니다.

  7. long-stay recall 34-35 / 36-37 움직임: 현재 결과만으로는 판정 불가
     target fold에는 fold_0/checkpoints/가 없고, per-row downstream outcome prediction artifact도 없습니다. fold_result.json에는 전체 test recall만 있습니다:

  test_recall = 0.7464436973
  test_auc    = 0.8879487939

  하지만 true raw LOS bin별 34-35, 36-37 recall을 계산하려면 per-row y_true/y_pred/y_score가 필요합니다. 현재 target artifact에는 그게 없어서 long-stay bin-specific recall은 비교할 수 없습니다.
