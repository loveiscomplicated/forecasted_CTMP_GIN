# explainer_main.py
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

import yaml
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.data_processing.data_utils import train_test_split_stratified
from src.models.ctmp_gin import ensure_ctmp_gin_los_encoder_defaults
from src.models.factory import build_model, build_edge
from src.trainers.base import load_checkpoint
from src.utils.seed_set import set_seed
from src.utils.device_set import device_set
from src.trainers.utils.early_stopper import EarlyStopper

from src.explainers.gb_ig_main import gb_ig_main
from src.explainers.integrated_gradients import ig_main, gin_ig_main

cur_dir = os.path.dirname(__file__)
root = os.path.join(cur_dir, '..', 'data')
save_path = os.path.join(cur_dir, 'results')
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

# --------@@@@ adjust model path !!! @@@@--------

# TODO add mode arg to select the explaination method, default: None -> do all method, make save path align according to this mode, add detailed configurations in this path 
# TODO add more configs based on the modules. ex) use_mean_in_explain, use_abs
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True) # config file location
    p.add_argument("--explain_method", type=str, required=True)
    p.add_argument("--is_mi_based_edge", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--decision_threshold", type=float, default=None)
    p.add_argument("--binary", type=int, default=None)
    return p.parse_args()

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def override_cfg(cfg: dict, args) -> dict:
    ensure_ctmp_gin_los_encoder_defaults(cfg)
    if args.explain_method is not None:
        cfg["explain_method"] = args.explain_method
    if args.device is not None:
        cfg["device"] = args.device
    if args.is_mi_based_edge is not None:
        cfg.setdefault("edge", {})["is_mi_based"] = bool(args.is_mi_based_edge)
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg.setdefault("train", {})["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed
    if args.binary is not None:
        cfg.setdefault("train", {})["binary"] = bool(args.binary)
    if args.decision_threshold is not None:
        cfg.setdefault("train", {})["decision_threshold"] = args.decision_threshold
        
    return cfg

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = override_cfg(cfg, args)

    use_mean_in_explain = True
    use_abs = True
    reduce = "mean"
    explain_method = cfg["explain_method"]

    # run_id = make_run_id(cfg)
    # run_dir = ensure_run_dir("runs", run_id)
    # logger = ExperimentLogger(cfg, run_dir)

    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    device = device_set(cfg["device"])

    # create dataset
    dataset = TEDSTensorDataset(
        root=root,
        binary=cfg["train"].get("binary", True),
        ig_label=cfg["train"].get("ig_label", False),
    )

    cfg["model"]["params"]["col_info"] = dataset.col_info
    cfg["model"]["params"]["num_classes"] = dataset.num_classes
    cfg["model"]["params"]["device"] = device

    # create dataloaders
    split_ratio = [cfg['train']['train_ratio'], cfg['train']['val_ratio'], cfg['train']['test_ratio']]
    train_loader, val_loader, test_loader, idx = train_test_split_stratified(dataset=dataset,  # type: ignore
                                                                                   batch_size=cfg['train']['batch_size'],
                                                                                   ratio=split_ratio,
                                                                                   seed=seed,
                                                                                   num_workers=cfg['train']['num_workers'],
                                                                                   )
    train_df = dataset.processed_df.iloc[idx[0]]
    num_nodes = len(dataset.col_info[2]) # col_info: (col_list, col_dims, ad_col_index, dis_col_index)
    if cfg["model"]["name"] == 'gin':
        num_nodes = len(dataset.col_info[0]) + 1

    print(dataset.col_info[0])
    # build model
    model = build_model(
        model_name=cfg["model"]["name"],
        **cfg["model"].get("params", {})
    )
    model = model.to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"학습 가능한 파라미터 개수: {total_trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=cfg["train"]["lr_scheduler_patience"])

    if explain_method == "gb_ig":
        model_path = os.path.join(cur_dir, '..', '..', 'runs', 'temp_ctmp_gin_ckpt', 'ctmp_epoch_36_loss_0.2738.pth')

        import pickle
        with open(os.path.join(cur_dir, 'edge_index.pickle'), 'rb') as f:
            edge_index = pickle.load(f)
        edge_index = edge_index.to(device) # type: ignore

        load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            filename=model_path
        )    

        print("\n--------------------Interpreting Models with Graph Based Integrated Gradients--------------------")
        # Call gb_ig_main to perform explanations
        gb_ig_main(model=model,
                edge_index=edge_index,
                dataset=dataset,
                test_loader=test_loader,
                use_abs=use_abs,
                device=device,
                reduce=reduce,
                use_mean_in_explain=use_mean_in_explain,
                seed=seed,
                save_path=save_path,
                sample_ratio=0.1) 
        print("--------------------Interpreting Models with Graph Based Integrated Gradients FINISHED--------------------")

    if explain_method == "ig":
        model_path = os.path.join(cur_dir, '..', '..', 'runs', '20260221-072012__gin__bs=32__lr=1.00e-03__seed=1', 'checkpoints', 'best.pt')

        load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            filename=model_path,
            map_location=device,
        )
        edge_index = build_edge(model_name=cfg["model"]["name"],
                            root=root,
                            seed=seed,
                            train_df=train_df,
                            num_nodes=num_nodes,
                            batch_size = cfg["train"]["batch_size"],
                            **cfg.get("edge", {})
                            )
        edge_index = edge_index.to(device) # type: ignore

        print("\n--------------------Interpreting Models with Integrated Gradients--------------------")
        save_path = os.path.join(cur_dir, 'results', 'integrated_gradients', 'full', 'val_dataset')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        ig_main(
            dataset=dataset,
            dataloader=val_loader,
            model=model,
            save_path=save_path,
            edge_index=edge_index,
            target="logit",
            n_steps=400,
            reduce="mean",
            keep_all=True,
            max_batches=None,
            verbose=True,
            sample_ratio=1,
        )

        save_path = os.path.join(cur_dir, 'results', 'integrated_gradients', 'full', 'test_dataset')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        ig_main(
            dataset=dataset,
            dataloader=test_loader,
            model=model,
            save_path=save_path,
            edge_index=edge_index,
            target="logit",
            n_steps=400,
            reduce="mean",
            keep_all=True,
            max_batches=None,
            verbose=True,
            sample_ratio=1,
        )
        print("--------------------Interpreting Models with Integrated Gradients FINISHED--------------------")

    if explain_method == "gin_ig":
        model_path = os.path.join(cur_dir, '..', '..', 'runs', '20260221-072012__gin__bs=32__lr=1.00e-03__seed=1', 'checkpoints', 'best.pt')

        load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            filename=model_path,
            map_location=device,
        )
        edge_index = build_edge(model_name=cfg["model"]["name"],
                            root=root,
                            seed=seed,
                            train_df=train_df,
                            num_nodes=num_nodes,
                            batch_size=cfg["train"]["batch_size"],
                            **cfg.get("edge", {})
                            )
        edge_index = edge_index.to(device) # type: ignore

        print("\n--------------------Interpreting GIN with Integrated Gradients--------------------")

        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            split_save_path = os.path.join(cur_dir, 'results', 'integrated_gradients', 'gin', f'{split_name}_dataset')
            if not os.path.exists(split_save_path):
                os.makedirs(split_save_path, exist_ok=True)

            gin_ig_main(
                dataset=dataset,
                dataloader=loader,
                model=model,
                save_path=split_save_path,
                edge_index=edge_index,
                target="logit",
                n_steps=400,
                reduce="mean",
                keep_all=True,
                max_batches=None,
                verbose=True,
                sample_ratio=1,
            )

        print("--------------------Interpreting GIN with Integrated Gradients FINISHED--------------------")

if __name__ == "__main__":
    main()
