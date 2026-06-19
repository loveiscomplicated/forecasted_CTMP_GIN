import os
import pandas as pd
from xgboost import XGBClassifier, Booster
import matplotlib.pyplot as plt
from sklearn.metrics import (
    log_loss, roc_auc_score,
    f1_score, precision_score, recall_score,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from scipy.stats import uniform, randint
from src.utils.experiment import ExperimentLogger
from src.models.xgboost.cv import cross_validate
from src.utils.send_message import send_discord_message

def get_scores(y_test, y_pred, y_pred_proba, binary=True):
    metrics = {}
    metrics["logloss"] = float(log_loss(y_test, y_pred_proba))

    if binary:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
        metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
    else:
        # NOTE:
        # Multiclass ROC-AUC is intentionally omitted due to instability and limited interpretability.
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        metrics["precision_macro"] = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    return metrics

def get_feature_importance(X_train, final_xgb_model, save_dir: str):
    feature_names = X_train.columns
    importances = final_xgb_model.feature_importances_
    feature_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    print("\n--- Feature Importance ---")
    print(feature_series)

    # save csv
    csv_path = os.path.join(save_dir, "xgboost_feature_importance.csv")
    feature_series.to_csv(csv_path, header=["importance"])

    # save plot
    plt.figure(figsize=(12, 8))
    feature_series.plot(kind="bar")
    plt.title("XGBoost Feature Importance")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "xgboost_feature_importance.png"), dpi=200)
    plt.close()


def _train_xgboost_binary(train_idx: list, val_idx: list, test_idx: list, df: pd.DataFrame, cfg):
    X = df.drop(columns=["REASONb"])
    y = df['REASONb'].copy()

    print("Converting feature dtypes to 'category'...")
    X = X.astype('category')
    print("Data types converted.")

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx] # type: ignore

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx] # type: ignore

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx] # type: ignore

    print("\n--- Training final model and evaluating with X_test ---")

    final_xgb_model = XGBClassifier(
        tree_method=cfg["train"]["tree_method"],
        enable_categorical=True,
        n_estimators=cfg["train"]["n_estimators"],
        learning_rate=cfg["train"]["learning_rate"],
        max_depth=cfg["train"]["max_depth"],

        min_child_weight=cfg["train"].get("min_child_weight", 1),
        gamma=cfg["train"].get("gamma", 0),
        subsample=cfg["train"].get("subsample", 1.0),
        colsample_bytree=cfg["train"].get("colsample_bytree", 1.0),
        reg_alpha=cfg["train"].get("reg_alpha", 0),
        reg_lambda=cfg["train"].get("reg_lambda", 1),
        eval_metric=cfg["train"]["eval_metric"],

        random_state=cfg["train"]["seed"],
        early_stopping_rounds=50
    )

    if cfg["train"]["do_cross_validation"]:
        final_xgb_model = cross_validate(final_xgb_model, X_train, y_train, cfg)

    else:
        final_xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1)

    y_pred_proba = final_xgb_model.predict_proba(X_test)[:, 1]
    y_pred = final_xgb_model.predict(X_test)
    return final_xgb_model, X_train, y_test, y_pred, y_pred_proba

def _train_xgboost_multi(train_idx: list, val_idx: list, test_idx: list, df: pd.DataFrame, cfg):
    X = df.drop(columns=["REASON"])
    y = df["REASON"].copy()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    K = len(le.classes_)

    X = X.astype("category")

    X_train = X.iloc[train_idx]
    y_train = y_encoded[train_idx] # type: ignore

    X_val = X.iloc[val_idx]
    y_val = y_encoded[val_idx] # type: ignore

    X_test = X.iloc[test_idx] 
    y_test = y_encoded[test_idx] # type: ignore
        
    final_xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=K,
        eval_metric="mlogloss",
        tree_method=cfg["train"]["tree_method"],
        enable_categorical=True,
        n_estimators=cfg["train"]["n_estimators"],
        learning_rate=cfg["train"]["learning_rate"],
        max_depth=cfg["train"]["max_depth"],

        min_child_weight=cfg["train"].get("min_child_weight", 1),
        gamma=cfg["train"].get("gamma", 0),
        subsample=cfg["train"].get("subsample", 1.0),
        colsample_bytree=cfg["train"].get("colsample_bytree", 1.0),
        reg_alpha=cfg["train"].get("reg_alpha", 0),
        reg_lambda=cfg["train"].get("reg_lambda", 1),

        random_state=cfg["train"]["seed"],
        early_stopping_rounds=50
    )

    if cfg["train"]["do_cross_validation"]:
        final_xgb_model = cross_validate(final_xgb_model, X_train, y_train, cfg)
    else:
        final_xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1)

    y_pred_proba = final_xgb_model.predict_proba(X_test)[:, 1]
    y_pred = final_xgb_model.predict(X_test)   # 평가용(정수)

    # For human-readable purposes
    # y_pred_label = le.inverse_transform(y_pred)

    return final_xgb_model, X_train, y_test, y_pred, y_pred_proba, le

def train_xgboost(train_idx, val_idx, test_idx, df, logger: ExperimentLogger | None, cfg):
    if cfg["train"]["binary"]:
        final_xgb_model, X_train, y_test, y_pred, y_pred_proba = _train_xgboost_binary(
            train_idx, val_idx, test_idx, df, cfg
        )
        metrics = get_scores(y_test, y_pred, y_pred_proba, binary=True)
        result_str = f"\n[Test] Loss: {metrics["logloss"]:.4f} | Acc: {metrics["accuracy"]:.4f}, Prec: {metrics["precision"]:.4f}, Rec: {metrics["recall"]:.4f}, F1: {metrics["f1"]:.4f}, AUC: {metrics["roc_auc"]:.4f}"

    else:
        final_xgb_model, X_train, y_test, y_pred, y_pred_proba, le = _train_xgboost_multi(
            train_idx, val_idx, test_idx, df, cfg
        )
        metrics = get_scores(y_test, y_pred, y_pred_proba, binary=False)
        result_str = f"\n[Test] Loss: {metrics["logloss"]:.4f} | Acc: {metrics["accuracy"]:.4f}, Prec: {metrics["precision_macro"]:.4f}, Rec: {metrics["recall_macro"]:.4f}, F1: {metrics["f1_macro"]:.4f}"

        y_pred_label = le.inverse_transform(y_pred)
        print("Pred label examples:", y_pred_label[:10])
    
    if logger:
        logger.log_metrics(epoch=0, metrics={"split": "test", **metrics})
        get_feature_importance(X_train, final_xgb_model, save_dir=logger.run_dir)

        # model save
        final_xgb_model.get_booster().save_model(
            os.path.join(logger.run_dir, "xgboost.json")
        )
    
    send_discord_message(message=result_str, bot_name='xgboost_training_bot')

    return metrics


