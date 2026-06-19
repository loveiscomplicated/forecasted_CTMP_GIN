import json
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.data_processing.data_utils import train_test_split_stratified
from src.models.forecast_inputs import (
    ensure_model_forecast_defaults,
    resolve_model_forecast_input_metadata,
)
from src.models.factory import build_model, build_edge
from src.trainers.forecasted_discharge import build_forecasted_discharge_provider
from src.trainers.forecasted_los import build_forecasted_los_provider
from src.trainers.base import run_train_loop
from src.utils.experiment import make_run_id, ensure_run_dir, ExperimentLogger, save_text
from src.utils.seed_set import set_seed
from src.utils.device_set import device_set
from src.trainers.utils.early_stopper import EarlyStopper

def run_single_experiment(cfg, 
                          root,
                          **kwargs):
    ensure_model_forecast_defaults(cfg)
    if bool(cfg.get("joint_forecast_pipeline", {}).get("enabled", False)):
        raise ValueError(
            "joint_forecast_pipeline is currently supported only with train.cv=true."
        )
    report_metric = kwargs.get("report_metric", "valid_auc")
    trial = kwargs.get("trial", None)
    mi_cache_path = kwargs.get("mi_cache_path", None)
    # mi_cached=kwargs.get("mi_cached", True)

    logger = None
    if trial is None: # if not parameter searching (normal training session)
        run_id = make_run_id(cfg)
        run_dir = ensure_run_dir("runs", run_id)
        logger = ExperimentLogger(cfg, run_dir) # if parameter searching, turn off the logger
        forecast_input_metadata = resolve_model_forecast_input_metadata(cfg)
        if forecast_input_metadata:
            save_text(
                os.path.join(run_dir, "forecast_input_metadata.json"),
                json.dumps(forecast_input_metadata, ensure_ascii=False, indent=2) + "\n",
            )
    else:
        forecast_input_metadata = resolve_model_forecast_input_metadata(cfg)

    seed = cfg["train"].get("seed", 42)
    split_seed = cfg["train"].get("split_seed", seed)
    model_seed = kwargs.get("model_seed", seed)
    set_seed(split_seed)

    device = device_set(cfg["device"])

    admission_only = cfg.get("admission_only", False)

    remove_los = True
    if not admission_only and cfg["model"]["name"] in ["gin", "a3tgcn_2_points", "gin_gru_2_points"]:
        remove_los = False # include LOS in calculating MI
        cfg["edge"]["remove_los"] = False

    # create dataset
    dataset = TEDSTensorDataset(
        root=root,
        binary=cfg["train"].get("binary", True),
        ig_label=cfg["train"].get("ig_label", False),
        remove_los=remove_los,
        do_preprocess=cfg["train"].get("do_preprocess", True),
        admission_only=admission_only,
    )
    discharge_provider = build_forecasted_discharge_provider(cfg, dataset, device)
    los_provider = build_forecasted_los_provider(cfg, dataset, device)

    cfg["model"]["params"]["col_info"] = dataset.col_info
    cfg["model"]["params"]["num_classes"] = dataset.num_classes
    cfg["model"]["params"]["device"] = device

    if admission_only:
        cfg["model"]["params"]["use_los"] = False

    if cfg["model"]["name"] in ['gin', 'mlp']:
        num_nodes = len(dataset.col_info[0])
    else:
        num_nodes = len(dataset.col_info[2]) # col_info: (col_list, col_dims, ad_col_index, dis_col_index)

    print(f"num_nodes set to {num_nodes}")

    # create dataloaders
    split_ratio = [cfg['train']['train_ratio'], cfg['train']['val_ratio'], cfg['train']['test_ratio']]
    train_loader, val_loader, test_loader, idx = train_test_split_stratified(dataset=dataset,  # type: ignore
                                                                                   batch_size=cfg['train']['batch_size'],
                                                                                   ratio=split_ratio,
                                                                                   seed=split_seed,
                                                                                   num_workers=cfg['train']['num_workers'],
                                                                                   )
    # split 완료 후 model seed 적용 (model 초기화/dropout 등에 영향)
    set_seed(model_seed)
    
    train_df = dataset.processed_df.iloc[idx[0]]

    if cfg["model"]["name"] == "xgboost":
        from src.models.xgboost import train_xgboost
        train_idx, val_idx, test_idx = idx
        return train_xgboost(train_idx, val_idx, test_idx, dataset.processed_df, logger, cfg)
    
    if cfg["model"]["name"] in ["a3tgcn", "a3tgcn_2_points"]:
        cfg["model"]["params"]["batch_size"] = cfg["train"].get("batch_size", 32)

    # build model
    model = build_model(
        model_name=cfg["model"]["name"],
        **cfg["model"].get("params", {})
    )
    model = model.to(device)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # build edge_index
    if cfg["model"]["name"] == "mlp":
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        if mi_cache_path is not None:
            cfg["edge"]["cache_path"] = mi_cache_path

        edge_index = build_edge(model_name=cfg["model"]["name"],
                                root=root,
                                seed=seed,
                                train_df=train_df,
                                num_nodes=num_nodes,
                                batch_size=cfg["train"]["batch_size"],
                                **cfg.get("edge", {})
                                )
        edge_index = edge_index.to(device)  # type: ignore

    # Precompute edge_index_2 (internal + cross-graph edges) once per trial.
    # Both edge_index and batch_size are fixed for the whole trial, so this
    # avoids recreating CPU tensors and H2D transfers on every forward pass.
    if hasattr(model, "precompute_edge_index_2"):
        model.precompute_edge_index_2(edge_index, cfg["train"]["batch_size"])

    if trial is None:
        print(model)
        print(f"학습 가능한 파라미터 개수: {total_trainable_params:,}")
        print(f'edge index: \n{edge_index}')
        print(f'edge index shape: \n{edge_index.shape}')

        # Save for later analysis scripts (extract, permutation, reeval)
        edge_index_save_path = os.path.join(run_dir, "edge_index.pt")
        torch.save(edge_index.cpu(), edge_index_save_path)
        print(f"edge_index saved: {edge_index_save_path}")

    if cfg["train"]["binary"]:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    if cfg["train"].get("optimizer", "adam") == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=cfg["train"]["learning_rate"], 
                                    weight_decay=cfg["train"].get("weight_decay", 0.0))
        
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                      lr=cfg["train"]["learning_rate"], 
                                      weight_decay=cfg["train"].get("weight_decay", 0.0))

    scheduler = ReduceLROnPlateau(optimizer, "min", patience=cfg["train"]["lr_scheduler_patience"])
    early_stopper = EarlyStopper(patience=cfg["train"]["early_stopping_patience"])

    out = run_train_loop(
        model=model,
        edge_index=edge_index,
        binary=cfg["train"]["binary"],
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopper=early_stopper,
        device=device,
        logger=logger,
        epochs=cfg["train"]["epochs"],
        decision_threshold=cfg["train"]["decision_threshold"],
        trial=trial,
        report_metric=report_metric,
        model_name=cfg["model"]["name"],
        los_provider=los_provider,
        discharge_provider=discharge_provider,
        checkpoint_extra={"forecast_input_metadata": forecast_input_metadata} if forecast_input_metadata else None,
    )

    return out
