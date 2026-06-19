from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import src.trainers.run_kfold_cv as runner


class _TinyDataset:
    def __init__(self, n: int = 10) -> None:
        self.n = n
        self.processed_df = pd.DataFrame({"x": list(range(n))})
        self.col_info = (["A", "B"], [2, 2], [0], [1])
        self.num_classes = 2

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        return torch.tensor([0, 0], dtype=torch.long), int(index % 2), torch.tensor(1)


def _single_run_cfg() -> dict:
    return {
        "device": "cpu",
        "run_name": "test_outcome_aware_single",
        "joint_forecast_pipeline": {
            "enabled": True,
            "stage1": {
                "model": {"name": "joint_generative_predictor", "params": {}},
                "train": {"do_preprocess": False},
                "joint_predictor": {
                    "predictor_type": "joint_generative",
                    "los_target_mode": "coarse",
                },
            },
            "stage2": {
                "enabled": True,
                "mode": "outcome_aware",
                "freeze_ctmp_gin": True,
                "train_predictor": True,
                "learning_rate": 1.0e-5,
                "weight_decay": 1.0e-5,
                "max_epochs": 1,
                "early_stopping_patience": 1,
                "selection_metric": "valid_auc",
            },
            "joint_forecast_input": {"mode": "distribution"},
        },
        "edge": {"is_mi_based": False},
        "model": {
            "name": "ctmp_gin",
            "params": {
                "embedding_dim": 4,
                "forecast_input_encoder": "distribution",
            },
        },
        "train": {
            "batch_size": 2,
            "learning_rate": 1.0e-3,
            "epochs": 1,
            "seed": 1,
            "binary": True,
            "test_ratio": 0.2,
            "num_workers": 0,
            "optimizer": "adam",
            "lr_scheduler_patience": 1,
            "early_stopping_patience": 1,
            "decision_threshold": 0.5,
            "ig_label": False,
            "do_preprocess": False,
            "cv": False,
        },
    }


def test_cached_split_dataset_preserves_distribution_los_dtype() -> None:
    payload = {
        "x": torch.zeros((2, 2), dtype=torch.long),
        "los": torch.tensor(
            [[0.2, 0.8, 0.0], [0.1, 0.3, 0.6]],
            dtype=torch.float32,
        ),
        "indices": torch.tensor([0, 1], dtype=torch.long),
    }

    dataset = runner._CachedSplitDataset(_TinyDataset(n=2), payload)
    _x, _y, los = dataset[0]

    assert los.dtype == torch.float32
    assert torch.allclose(los, payload["los"][0])


def test_load_fold_forecasted_data_from_cache_uses_joint_split_metadata(tmp_path) -> None:
    fold_dir = tmp_path / "cv" / "folds" / "fold_0"
    cache_dir = fold_dir / "cached_predictions"
    cache_dir.mkdir(parents=True)
    (fold_dir / "joint_forecast_pipeline_splits.json").write_text(
        (
            '{"train_core_idx":[0,1],"gnn_val_idx":[2,3],'
            '"outer_test_idx":[4,5],"joint_run_dir":"joint_predictor"}'
        ),
        encoding="utf-8",
    )
    for name, indices in {
        "train_core_joint.pt": [0, 1],
        "gnn_val_joint.pt": [2, 3],
        "outer_test_joint.pt": [4, 5],
    }.items():
        torch.save(
            {
                "x": torch.zeros((2, 2), dtype=torch.long),
                "los": torch.ones((2, 3), dtype=torch.float32),
                "indices": torch.tensor(indices, dtype=torch.long),
            },
            cache_dir / name,
        )
    cfg = {"train": {"batch_size": 2, "num_workers": 0}}

    data = runner._load_fold_forecasted_data_from_cache(
        cfg,
        fold_dir=str(fold_dir),
        base_dataset=_TinyDataset(n=6),
    )

    assert data.train_idx.tolist() == [0, 1]
    assert data.val_idx.tolist() == [2, 3]
    assert data.test_idx.tolist() == [4, 5]
    x, y, los = next(iter(data.val_loader))
    assert x.shape == (2, 2)
    assert y.tolist() == [0, 1]
    assert los.shape == (2, 3)


def test_restore_training_state_from_last_checkpoint_preserves_existing_best(tmp_path) -> None:
    run_dir = tmp_path / "fold_0"
    (run_dir / "checkpoints").mkdir(parents=True)
    cfg = {
        "train": {
            "monitor_metric": "valid_auc",
            "monitor_mode": "max",
        }
    }
    logger = runner.ExperimentLogger(cfg, str(run_dir))
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "cfg": cfg,
    }
    torch.save(
        {**state, "epoch": 2, "metrics": {"valid_auc": 0.8}},
        run_dir / "checkpoints" / "best.pt",
    )
    torch.save(
        {**state, "epoch": 4, "metrics": {"valid_auc": 0.7}},
        run_dir / "checkpoints" / "last.pt",
    )

    start_epoch = runner._restore_training_state_from_last_checkpoint(
        fold_dir=str(run_dir),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        device=torch.device("cpu"),
    )

    assert start_epoch == 5
    assert logger.best_epoch == 2
    assert logger.best_value == pytest.approx(0.8)


def test_load_edge_index_for_single_run_reuses_saved_edge_without_building(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "fold_0"
    run_dir.mkdir()
    expected_edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    torch.save(expected_edge, run_dir / "edge_index.pt")

    def fail_build_edge(**_kwargs):
        raise AssertionError("build_edge should not be called when edge_index.pt exists")

    monkeypatch.setattr(runner, "build_edge", fail_build_edge)

    edge = runner._load_edge_index_for_single_run(
        cfg={
            "model": {"name": "ctmp_gin"},
            "train": {"seed": 1, "batch_size": 2},
            "edge": {"is_mi_based": True},
        },
        root="src/data",
        run_dir=str(run_dir),
        dataset=_TinyDataset(n=2),
        train_idx=np.array([0, 1], dtype=np.int64),
        num_nodes=2,
        device=torch.device("cpu"),
    )

    assert torch.equal(edge.cpu(), expected_edge)


def _patch_common_single_run_deps(monkeypatch, tmp_path, *, create_predictor: bool = True):
    run_dir = tmp_path / "single_run"
    (run_dir / "checkpoints").mkdir(parents=True)

    monkeypatch.setattr(
        runner,
        "ensure_run_dir",
        lambda _base, _run_id: str(run_dir),
    )
    monkeypatch.setattr(runner, "_build_dataset", lambda _cfg, _root: _TinyDataset())
    monkeypatch.setattr(
        runner,
        "holdout_test_split_stratified",
        lambda **_kwargs: (np.arange(8, dtype=np.int64), np.arange(8, 10, dtype=np.int64)),
    )
    monkeypatch.setattr(
        runner,
        "build_edge",
        lambda **_kwargs: torch.empty((2, 0), dtype=torch.long),
    )
    monkeypatch.setattr(
        runner,
        "build_model",
        lambda **_kwargs: torch.nn.Linear(1, 1),
    )

    def fake_prepare_joint_forecast_fold_data(**kwargs):
        joint_dir = run_dir / "joint_predictor"
        if create_predictor:
            (joint_dir / "checkpoints").mkdir(parents=True)
            torch.save(
                {"model_state_dict": {}, "cfg": {"model": {"params": {}}}},
                joint_dir / "checkpoints" / "best.pt",
            )
        return SimpleNamespace(
            train_idx=np.array([0, 1, 2, 3], dtype=np.int64),
            val_idx=np.array([4, 5], dtype=np.int64),
            test_idx=np.array([8, 9], dtype=np.int64),
            train_loader=[],
            val_loader=[],
            test_loader=[],
            split_payload={
                "joint_run_dir": str(joint_dir),
                "predictor_val_idx": [6, 7],
            },
        )

    monkeypatch.setattr(
        runner,
        "prepare_joint_forecast_fold_data",
        fake_prepare_joint_forecast_fold_data,
    )
    return run_dir


def test_outcome_aware_single_run_passes_new_checkpoints_to_stage2(monkeypatch, tmp_path):
    run_dir = _patch_common_single_run_deps(monkeypatch, tmp_path)
    stage2_seen = {}

    def fake_run_train_loop(**kwargs):
        logger = kwargs["logger"]
        torch.save(
            {"model_state_dict": kwargs["model"].state_dict()},
            f"{logger.run_dir}/checkpoints/best.pt",
        )
        return {
            "best_epoch": 1,
            "best_valid_metric": 0.7,
            "best_valid_metrics": {"valid_auc": 0.7, "valid_f1": 0.6, "valid_acc": 0.8},
            "test_loss": 0.5,
            "test_acc": 0.8,
            "test_precision": 0.8,
            "test_recall": 0.8,
            "test_f1": 0.8,
            "test_auc": 0.75,
        }

    def fake_stage2(**kwargs):
        stage2_seen.update(kwargs)
        assert torch.load(kwargs["baseline_checkpoint_path"])["model_state_dict"]
        return {
            "best_epoch": 1,
            "best_valid_metric": 0.9,
            "best_valid_metrics": {"valid_auc": 0.9, "valid_f1": 0.8, "valid_acc": 0.85},
            "test_loss": 0.4,
            "test_acc": 0.85,
            "test_precision": 0.85,
            "test_recall": 0.85,
            "test_f1": 0.85,
            "test_auc": 0.91,
            "baseline_valid_auc": 0.7,
            "baseline_test_auc": 0.75,
            "stage2_valid_auc": 0.9,
            "stage2_test_auc": 0.91,
            "run_dir": str(run_dir / "outcome_aware_stage2"),
        }

    monkeypatch.setattr(runner, "run_train_loop", fake_run_train_loop)
    monkeypatch.setattr(runner, "run_outcome_aware_stage2", fake_stage2)

    result = runner.run_outcome_aware_single_run(_single_run_cfg(), root="src/data")

    assert result["stage2_test_auc"] == pytest.approx(0.91)
    assert stage2_seen["predictor_checkpoint_path"].endswith(
        "joint_predictor/checkpoints/best.pt"
    )
    assert stage2_seen["baseline_checkpoint_path"].endswith("checkpoints/best.pt")
    assert (run_dir / "single_run_splits.json").exists()
    assert (run_dir / "single_run_result.json").exists()


def test_outcome_aware_single_run_requires_stage1_checkpoint(monkeypatch, tmp_path):
    _patch_common_single_run_deps(monkeypatch, tmp_path, create_predictor=False)
    monkeypatch.setattr(runner, "run_train_loop", lambda **_kwargs: {})
    monkeypatch.setattr(runner, "run_outcome_aware_stage2", lambda **_kwargs: {})

    with pytest.raises(FileNotFoundError, match="Stage1 predictor checkpoint missing"):
        runner.run_outcome_aware_single_run(_single_run_cfg(), root="src/data")


def test_stage2_only_accepts_single_run_source_dir(monkeypatch, tmp_path):
    source_run_dir = tmp_path / "single_run_source"
    (source_run_dir / "joint_predictor" / "checkpoints").mkdir(parents=True)
    (source_run_dir / "checkpoints").mkdir(parents=True)
    (source_run_dir / "config.final.yaml").write_text("{}", encoding="utf-8")
    (source_run_dir / "single_run_splits.json").write_text(
        '{"train_core_idx":[0,1],"gnn_val_idx":[2],"outer_test_idx":[3]}',
        encoding="utf-8",
    )
    (source_run_dir / "single_run_result.json").write_text(
        (
            '{"status":"completed","best_valid_metrics":{"valid_auc":0.7,'
            '"valid_f1":0.6,"valid_acc":0.8},"test_auc":0.75,'
            '"test_f1":0.65,"test_acc":0.81}'
        ),
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": {"weight": torch.tensor([[1.0]]), "bias": torch.tensor([0.0])}},
        source_run_dir / "checkpoints" / "best.pt",
    )
    torch.save(
        {"model_state_dict": {}, "cfg": {"model": {"params": {}}}},
        source_run_dir / "joint_predictor" / "checkpoints" / "best.pt",
    )

    monkeypatch.setattr(runner, "_load_yaml_file", lambda _path: _single_run_cfg())
    monkeypatch.setattr(runner, "_build_dataset", lambda _cfg, _root: _TinyDataset())
    monkeypatch.setattr(
        runner,
        "build_model",
        lambda **_kwargs: torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        runner,
        "build_edge",
        lambda **_kwargs: torch.empty((2, 0), dtype=torch.long),
    )
    monkeypatch.setattr(
        runner,
        "_stage2_only_run_dir",
        lambda *_args, **_kwargs: str((tmp_path / "stage2_only").mkdir(parents=True, exist_ok=True) or (tmp_path / "stage2_only")),
    )
    monkeypatch.setattr(
        runner,
        "run_outcome_aware_stage2",
        lambda **kwargs: {
            "best_epoch": 1,
            "best_valid_metric": 0.9,
            "best_valid_metrics": {"valid_auc": 0.9},
            "test_loss": 0.4,
            "test_acc": 0.85,
            "test_precision": 0.85,
            "test_recall": 0.85,
            "test_f1": 0.85,
            "test_auc": 0.91,
            "baseline_valid_auc": kwargs["baseline_metrics"].get("baseline_valid_auc", float("nan")),
            "baseline_test_auc": kwargs["baseline_metrics"].get("baseline_test_auc", float("nan")),
            "stage2_valid_auc": 0.9,
            "stage2_test_auc": 0.91,
            "run_dir": str(tmp_path / "stage2_only" / "outcome_aware_stage2"),
        },
    )

    result = runner.run_outcome_aware_stage2_only(
        _single_run_cfg(),
        root="src/data",
        fold=0,
        source_run_dir=str(source_run_dir),
    )

    assert result["source_kind"] == "single_run"
    assert result["source_artifact_dir"] == str(source_run_dir)
    assert result["stage2_test_auc"] == pytest.approx(0.91)


def test_stage2_only_passes_requested_lambda_aux_to_stage2(monkeypatch, tmp_path):
    source_run_dir = tmp_path / "single_run_source"
    (source_run_dir / "joint_predictor" / "checkpoints").mkdir(parents=True)
    (source_run_dir / "checkpoints").mkdir(parents=True)
    (source_run_dir / "config.final.yaml").write_text("{}", encoding="utf-8")
    (source_run_dir / "single_run_splits.json").write_text(
        '{"train_core_idx":[0,1],"gnn_val_idx":[2],"outer_test_idx":[3]}',
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": {"weight": torch.tensor([[1.0]]), "bias": torch.tensor([0.0])}},
        source_run_dir / "checkpoints" / "best.pt",
    )
    torch.save(
        {"model_state_dict": {}, "cfg": {"model": {"params": {}}}},
        source_run_dir / "joint_predictor" / "checkpoints" / "best.pt",
    )

    requested_cfg = _single_run_cfg()
    requested_cfg["joint_forecast_pipeline"]["stage2"]["lambda_aux"] = 0.03
    seen: dict[str, float] = {}

    monkeypatch.setattr(runner, "_load_yaml_file", lambda _path: _single_run_cfg())
    monkeypatch.setattr(runner, "_build_dataset", lambda _cfg, _root: _TinyDataset())
    monkeypatch.setattr(runner, "build_model", lambda **_kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(
        runner,
        "build_edge",
        lambda **_kwargs: torch.empty((2, 0), dtype=torch.long),
    )
    monkeypatch.setattr(
        runner,
        "_stage2_only_run_dir",
        lambda *_args, **_kwargs: str((tmp_path / "stage2_only").mkdir(parents=True, exist_ok=True) or (tmp_path / "stage2_only")),
    )
    monkeypatch.setattr(
        runner,
        "_load_single_run_forecasted_data",
        lambda _cfg, **_kwargs: SimpleNamespace(
            train_idx=np.array([0, 1], dtype=np.int64),
            val_idx=np.array([2], dtype=np.int64),
            test_idx=np.array([3], dtype=np.int64),
            train_loader=[],
            val_loader=[],
            test_loader=[],
            split_payload={},
        ),
    )
    monkeypatch.setattr(
        runner,
        "_reconstruct_baseline_results",
        lambda **_kwargs: {
            "best_valid_metrics": {
                "valid_auc": 0.72,
                "valid_f1": 0.62,
                "valid_acc": 0.82,
            },
            "test_auc": 0.76,
            "test_f1": 0.66,
            "test_acc": 0.83,
        },
    )

    def fake_stage2(**kwargs):
        seen["lambda_aux"] = kwargs["cfg"]["joint_forecast_pipeline"]["stage2"]["lambda_aux"]
        seen["source_artifact_dir"] = kwargs["source_artifact_dir"]
        seen["baseline_valid_auc"] = kwargs["baseline_metrics"]["baseline_valid_auc"]
        return {
            "best_epoch": 1,
            "best_valid_metric": 0.9,
            "best_valid_metrics": {"valid_auc": 0.9},
            "test_loss": 0.4,
            "test_acc": 0.85,
            "test_precision": 0.85,
            "test_recall": 0.85,
            "test_f1": 0.85,
            "test_auc": 0.91,
            "baseline_valid_auc": kwargs["baseline_metrics"].get("baseline_valid_auc", float("nan")),
            "baseline_test_auc": kwargs["baseline_metrics"].get("baseline_test_auc", float("nan")),
            "stage2_valid_auc": 0.9,
            "stage2_test_auc": 0.91,
            "run_dir": str(tmp_path / "stage2_only" / "outcome_aware_stage2"),
        }

    monkeypatch.setattr(runner, "run_outcome_aware_stage2", fake_stage2)

    runner.run_outcome_aware_stage2_only(
        requested_cfg,
        root="src/data",
        fold=0,
        source_run_dir=str(source_run_dir),
    )

    assert seen["lambda_aux"] == pytest.approx(0.03)
    assert seen["source_artifact_dir"] == str(source_run_dir)
    assert seen["baseline_valid_auc"] == pytest.approx(0.72)


def test_resume_completed_single_run_returns_saved_result(monkeypatch, tmp_path):
    run_dir = tmp_path / "completed_single_run"
    run_dir.mkdir()
    (run_dir / "config.final.yaml").write_text("device: cpu\n", encoding="utf-8")
    (run_dir / "single_run_result.json").write_text(
        '{"status":"completed","run_dir":"x","stage2_valid_auc":0.9,"stage2_test_auc":0.91}',
        encoding="utf-8",
    )
    (run_dir / "single_run_status.json").write_text('{"status":"failed"}', encoding="utf-8")
    (run_dir / "joint_predictor" / "checkpoints").mkdir(parents=True)
    (run_dir / "cached_predictions").mkdir()
    (run_dir / "checkpoints").mkdir()
    for rel in (
        "joint_predictor/checkpoints/best.pt",
        "cached_predictions/train_core_joint.pt",
        "cached_predictions/gnn_val_joint.pt",
        "cached_predictions/outer_test_joint.pt",
        "checkpoints/best.pt",
        "joint_forecast_pipeline_splits.json",
    ):
        path = run_dir / rel
        if path.suffix == ".pt":
            torch.save({}, path)
        else:
            path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(runner, "_validate_outcome_aware_single_run_cfg", lambda _cfg: None)

    result = runner.run_outcome_aware_single_run(
        _single_run_cfg(),
        root="src/data",
        resume_run_dir=str(run_dir),
    )

    assert result["stage2_test_auc"] == pytest.approx(0.91)
    status_payload = runner._load_json(str(run_dir / "single_run_status.json"))
    assert status_payload["status"] == "completed"
    assert status_payload["last_completed_stage"] == "stage2"


def test_resume_single_run_skips_stage1_and_baseline_when_checkpoints_exist(
    monkeypatch,
    tmp_path,
):
    run_dir = tmp_path / "resume_single_run"
    run_dir.mkdir()
    runner.save_yaml(str(run_dir / "config.final.yaml"), _single_run_cfg())
    (run_dir / "checkpoints").mkdir()
    (run_dir / "joint_predictor" / "checkpoints").mkdir(parents=True)
    (run_dir / "cached_predictions").mkdir()
    torch.save(
        {"model_state_dict": {"weight": torch.tensor([[1.0]]), "bias": torch.tensor([0.0])}, "epoch": 1},
        run_dir / "checkpoints" / "best.pt",
    )
    torch.save(
        {"model_state_dict": {}, "cfg": {"model": {"params": {}}}},
        run_dir / "joint_predictor" / "checkpoints" / "best.pt",
    )
    for name in ("train_core_joint.pt", "gnn_val_joint.pt", "outer_test_joint.pt"):
        torch.save({"indices": torch.tensor([0]), "x": torch.zeros((1, 2), dtype=torch.long), "los": torch.ones(1, dtype=torch.long)}, run_dir / "cached_predictions" / name)
    (run_dir / "joint_forecast_pipeline_splits.json").write_text("{}", encoding="utf-8")
    (run_dir / "single_run_splits.json").write_text(
        '{"train_core_idx":[0,1,2,3],"gnn_val_idx":[4,5],"stage2_test_idx":[8,9],"outer_train_idx":[0,1,2,3,4,5,6,7],"outer_test_idx":[8,9]}',
        encoding="utf-8",
    )
    torch.save(torch.empty((2, 0), dtype=torch.long), run_dir / "edge_index.pt")

    monkeypatch.setattr(runner, "_build_dataset", lambda _cfg, _root: _TinyDataset())
    monkeypatch.setattr(
        runner,
        "holdout_test_split_stratified",
        lambda **_kwargs: (np.arange(8, dtype=np.int64), np.arange(8, 10, dtype=np.int64)),
    )
    monkeypatch.setattr(
        runner,
        "_load_single_run_forecasted_data",
        lambda _cfg, **_kwargs: SimpleNamespace(
            train_idx=np.array([0, 1, 2, 3], dtype=np.int64),
            val_idx=np.array([4, 5], dtype=np.int64),
            test_idx=np.array([8, 9], dtype=np.int64),
            train_loader=[],
            val_loader=[],
            test_loader=[],
            split_payload={"joint_run_dir": str(run_dir / "joint_predictor")},
        ),
    )
    monkeypatch.setattr(
        runner,
        "_reconstruct_baseline_results",
        lambda **_kwargs: {
            "best_epoch": 1,
            "best_valid_metric": 0.7,
            "best_valid_metrics": {"valid_auc": 0.7, "valid_f1": 0.6, "valid_acc": 0.8},
            "test_loss": 0.5,
            "test_acc": 0.8,
            "test_precision": 0.8,
            "test_recall": 0.8,
            "test_f1": 0.8,
            "test_auc": 0.75,
        },
    )
    monkeypatch.setattr(
        runner,
        "build_model",
        lambda **_kwargs: torch.nn.Linear(1, 1),
    )
    prepare_calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "prepare_joint_forecast_fold_data",
        lambda **_kwargs: prepare_calls.append("stage1"),
    )
    train_calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "run_train_loop",
        lambda **_kwargs: train_calls.append("baseline"),
    )
    stage2_seen = {}
    monkeypatch.setattr(
        runner,
        "run_outcome_aware_stage2",
        lambda **kwargs: stage2_seen.update(kwargs) or {
            "best_epoch": 1,
            "best_valid_metric": 0.9,
            "best_valid_metrics": {"valid_auc": 0.9},
            "test_loss": 0.4,
            "test_acc": 0.85,
            "test_precision": 0.85,
            "test_recall": 0.85,
            "test_f1": 0.85,
            "test_auc": 0.91,
            "baseline_valid_auc": 0.7,
            "baseline_test_auc": 0.75,
            "stage2_valid_auc": 0.9,
            "stage2_test_auc": 0.91,
            "run_dir": str(run_dir / "outcome_aware_stage2"),
        },
    )

    result = runner.run_outcome_aware_single_run(
        _single_run_cfg(),
        root="src/data",
        resume_run_dir=str(run_dir),
    )

    assert prepare_calls == []
    assert train_calls == []
    assert result["stage2_test_auc"] == pytest.approx(0.91)
    assert stage2_seen["predictor_checkpoint_path"].endswith(
        "joint_predictor/checkpoints/best.pt"
    )
