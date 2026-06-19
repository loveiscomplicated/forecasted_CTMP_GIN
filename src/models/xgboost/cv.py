import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from scipy.stats import uniform, randint


def _infer_num_class(y_train) -> int:
    # y_train이 label-encoded (0..K-1)라면 가장 좋고,
    # 아니라면 unique 개수로 K를 잡는다.
    return int(len(np.unique(y_train)))


def _build_scorer(binary: bool, metric: str = "auto"):
    """
    metric:
      - "auto": binary -> roc_auc, multiclass -> neg_log_loss
      - "roc_auc": (binary only 권장)
      - "f1_macro": multiclass 가능
      - "neg_log_loss": binary/multiclass 모두 가능 (확률 필요)
    """
    if metric == "auto":
        if binary:
            return make_scorer(roc_auc_score, response_method='predict_proba')
        return "neg_log_loss"

    if metric == "roc_auc":
        if not binary:
            raise ValueError("roc_auc metric is recommended for binary only.")
        return make_scorer(roc_auc_score, response_method='predict_proba')

    if metric == "f1_macro":
        # predict 결과 기반
        return make_scorer(f1_score, average="macro", zero_division=0)

    if metric == "neg_log_loss":
        return "neg_log_loss"

    raise ValueError(f"Unknown metric: {metric}")


def _maybe_set_multiclass_params(xgb_model: XGBClassifier, y_train, binary: bool) -> XGBClassifier:
    """
    다중분류일 때는 XGBClassifier에 objective/num_class를 세팅해준다.
    (Grid/Random search 과정에서 estimator 복제되므로 set_params로 충분)
    """
    if binary:
        return xgb_model

    K = _infer_num_class(y_train)
    # multi-class softprob: predict_proba가 (N, K) 나오도록
    return xgb_model.set_params(objective="multi:softprob", num_class=K)


def grid(
    xgb_model: XGBClassifier,
    X_train,
    y_train,
    binary: bool = True,
    scoring_metric: str = "auto",
    n_splits: int = 5,
    random_state: int = 42,
):
    # 탐색할 하이퍼파라미터 범위
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.8, 1.0],
    }

    # estimator 준비 (multi면 objective/num_class 세팅)
    xgb_model = _maybe_set_multiclass_params(xgb_model, y_train, binary=binary)

    # scoring 준비
    scorer = _build_scorer(binary=binary, metric=scoring_metric)

    # StratifiedKFold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=1,
        verbose=1,
        refit=True,
    )

    print("Grid Search 시작...")
    grid_search.fit(X_train, y_train)
    print("Grid Search 완료.")

    print("\n--- 최적의 하이퍼파라미터 조합 ---")
    print(grid_search.best_params_)
    print("\n--- 최고 CV 점수 ---")
    print(f"{grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def random(
    xgb_model: XGBClassifier,
    X_train,
    y_train,
    binary: bool = True,
    scoring_metric: str = "auto",
    n_iter: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
):
    # 탐색할 파라미터 분포
    param_distributions = {
        "n_estimators": randint(50, 500),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3),
        # 필요하면 여기에 subsample/colsample_bytree 등 추가 가능
    }

    # estimator 준비
    xgb_model = _maybe_set_multiclass_params(xgb_model, y_train, binary=binary)

    # scoring 준비
    scorer = _build_scorer(binary=binary, metric=scoring_metric)

    # StratifiedKFold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        n_jobs=1,
        random_state=random_state,
        verbose=1,
        refit=True,
    )

    print("Random Search Started...")
    random_search.fit(X_train, y_train)
    print("Random Search Finished.")

    print("\n--- Optimal Parameter Setting ---")
    print(random_search.best_params_)
    print("\n--- Best CV Score ---")
    print(f"{random_search.best_score_:.4f}")

    return random_search.best_estimator_


def cross_validate(
    xgb_model: XGBClassifier,
    X_train,
    y_train,
    cfg,
):
    """
    cfg["train"]["cv_random_or_grid"]: True면 random, False면 grid
    cfg에서 scoring을 추가로 받고 싶으면 cfg["train"]["cv_scoring"] 같은 키를 쓰면 됨.
    """
    binary = cfg["train"]["binary"]
    scoring_metric = cfg.get("train", {}).get("cv_scoring", "auto")
    if cfg["train"]["cv_random_or_grid"]: # True for random, False for grid
        seed = cfg["train"]["seed"]
        return random(xgb_model, X_train, y_train, binary=binary, scoring_metric=scoring_metric, random_state=seed)
    return grid(xgb_model, X_train, y_train, binary=binary, scoring_metric=scoring_metric)