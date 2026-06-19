import numpy as np
from sklearn.metrics import roc_auc_score # AUC 계산을 위해 필요

def compute_metrics(y_true, y_pred, y_scores, num_classes: int):
    """
    정확도, 정밀도, 재현율, F1 점수 외에 AUC를 추가로 계산합니다.
    Args:
        y_true (list/array): 실제 레이블 (0, 1, ...)
        y_pred (list/array): 이산 예측 레이블 (0, 1, ...)
        y_scores (list/array): 클래스 1에 대한 예측 확률 또는 로짓 (AUC 계산용)
        num_classes (int): 클래스 개수
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    ys = np.asarray(y_scores)
    
    # 1. Accuracy (정확도)
    acc = (yt == yp).mean() if yt.size else 0.0

    # 2. AUC (Area Under the ROC Curve)
    # AUC는 이진 분류(num_classes=2)에서만 의미가 있습니다.
    # ys의 크기가 0이 아니고, 모든 ys 값이 동일하지 않을 때만 계산합니다.
    if num_classes == 2 and ys.size > 0 and np.std(ys) > 0:
        # AUC 계산: y_true와 클래스 1에 대한 예측 점수(확률) 사용
        roc_auc = roc_auc_score(yt, ys)
    else:
        roc_auc = 0.0
        
    # 3. Precision, Recall, F1-score (정밀도, 재현율, F1 점수)
    if num_classes == 2:
        tp = ((yp == 1) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        # 반환 값 순서: acc, precision, recall, f1, auc
        return float(acc), float(precision), float(recall), float(f1), float(roc_auc)

    # 다중 클래스 처리 (Multi-class)
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = ((yp == c) & (yt == c)).sum()
        fp = ((yp == c) & (yt != c)).sum()
        fn = ((yp != c) & (yt == c)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f)
    
    precision = float(np.mean(precisions))
    recall    = float(np.mean(recalls))
    f1        = float(np.mean(f1s))
    
    # 다중 클래스에서는 AUC가 정의되지 않거나, One-vs-Rest AUC를 사용하므로 0.0 반환
    return float(acc), precision, recall, f1, 0.0