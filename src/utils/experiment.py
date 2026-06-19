import os
import sys
import json
import yaml
import time
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def _now_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _format_float(x: Any) -> str:
    try:
        return f"{float(x):.2e}"
    except Exception:
        return str(x)


def make_run_id(cfg: Dict[str, Any]) -> str:
    """
    Example:
    20260112-174200__ctmp_gin__bs=512__lr=5e-4__seed=42

    run_name 키가 config에 있으면 model.name 대신 사용한다.
    병렬 실험처럼 model.name이 동일한 경우 run_name으로 구분한다.
    """
    ts = _now_run_id()
    model = cfg.get("run_name") or cfg.get("model", {}).get("name", "model")

    train = cfg.get("train", {})
    model_params = cfg.get("model", {}).get("params", {})
    bs = train.get("batch_size", "NA")
    lr = train.get("learning_rate", "NA")
    seed = train.get("seed", "NA")
    los_emb = str(model_params.get("los_emb", "embedding"))

    parts = [
        ts,
        model,
        f"bs={bs}",
        f"lr={_format_float(lr)}",
        f"seed={seed}",
    ]
    if los_emb not in {"embedding", "nn_embedding"}:
        parts.append(f"los_emb={los_emb}")
    return "__".join(map(str, parts))


def ensure_run_dir(base_dir: str, run_id: str) -> str:
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)  # fail if collision
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir


def _get_git_info() -> str:
    """
    Return a text blob with git commit + dirty status, if available.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.STDOUT
        ).decode().strip()
        dirty = "dirty" if status else "clean"
        return f"commit: {commit}\nstatus: {dirty}\n\nporcelain:\n{status}\n"
    except Exception as e:
        return f"git info unavailable: {e}\n"


def _get_command_line() -> str:
    return " ".join([sys.executable] + sys.argv)


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_yaml(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class CheckpointPolicy:
    save_ckpt: bool = True
    save_every: int = 1          # save every N epochs (0 or <0 disables periodic saving)
    save_best: bool = True       # save best checkpoint
    monitor: str = "valid_auc"   # metric name to monitor
    mode: str = "max"            # "min" for loss, "max" for auc/f1
    keep_last: bool = True       # keep "last.pt"


class ExperimentLogger:
    """
    - Creates run directory + basic artifacts:
      config.final.yaml, command.txt, git.txt, metrics.jsonl
    - Optional checkpoint saving (on/off via policy)
    """

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")

        train_cfg = cfg.get("train", {})
        self.policy = CheckpointPolicy(
            save_ckpt=bool(train_cfg.get("save_ckpt", True)),
            save_every=int(train_cfg.get("save_every", 1)),
            save_best=bool(train_cfg.get("save_best", True)),
            monitor=str(train_cfg.get("monitor_metric", train_cfg.get("monitor", "valid_auc"))),
            mode=str(train_cfg.get("monitor_mode", train_cfg.get("mode", "max"))).lower(),
            keep_last=bool(train_cfg.get("keep_last", True)),
        )

        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None

        # Save run artifacts immediately (so even if training crashes, we still have config)
        save_yaml(os.path.join(run_dir, "config.final.yaml"), cfg)
        save_text(os.path.join(run_dir, "command.txt"), _get_command_line() + "\n")
        save_text(os.path.join(run_dir, "git.txt"), _get_git_info())

    def log_metrics(self, epoch: int, metrics: Dict[str, Any]) -> None:
        record = {"epoch": epoch, **metrics}
        append_jsonl(self.metrics_path, record)

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.policy.mode == "min":
            return value < self.best_value
        if self.policy.mode == "max":
            return value > self.best_value
        raise ValueError(f"Unknown mode: {self.policy.mode}")

    def maybe_save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        metrics: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save checkpoints according to policy:
        - best.pt when monitored metric improves (if save_best)
        - last.pt every epoch (if keep_last)
        - epoch_{k}.pt every save_every epochs (if save_every > 0)
        """
        if not self.policy.save_ckpt:
            return

        extra = extra or {}
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "cfg": self.cfg,
            **extra,
        }

        # keep last
        if self.policy.keep_last:
            torch.save(state, os.path.join(self.ckpt_dir, "last.pt"))

        # periodic
        if self.policy.save_every and self.policy.save_every > 0:
            if epoch % self.policy.save_every == 0:
                torch.save(state, os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt"))

        # best
        if self.policy.save_best:
            if self.policy.monitor not in metrics:
                # If monitor metric isn't present, skip saving (avoid crashing training)
                # This is expected for test-metric log calls, but warns on likely misconfiguration.
                if any(k.startswith("valid_") for k in metrics):
                    print(f"Warning: monitor metric '{self.policy.monitor}' not in metrics, skipping best checkpoint.")
                return
            try:
                cur = float(metrics[self.policy.monitor])
            except Exception:
                return

            if self._is_better(cur):
                self.best_value = cur
                self.best_epoch = epoch
                torch.save(state, os.path.join(self.ckpt_dir, "best.pt"))
                # Also write a small text marker
                save_text(
                    os.path.join(self.run_dir, "best.txt"),
                    f"best_epoch: {epoch}\n{self.policy.monitor}: {cur}\n"
                    )

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ckpt_name: str = "last.pt",
        map_location: Optional[str] = "cpu",
    ) -> Dict[str, Any]:
        
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=map_location)

        # 1) model
        model.load_state_dict(state["model_state_dict"])

        # 2) optimizer
        if optimizer is not None and state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        # 3) scheduler
        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        # best tracking 복원: 체크포인트 종류와 관계없이 metrics에서 monitor 값을 읽어 복원한다.
        # 이렇게 해야 last.pt / epoch_N.pt 로 재개해도 _is_better() 비교가 올바르게 동작한다.
        m = state.get("metrics", {})
        if self.policy.monitor in m:
            try:
                self.best_value = float(m[self.policy.monitor])
                self.best_epoch = state.get("epoch", None)
            except Exception:
                self.best_value = None
                self.best_epoch = None
        else:
            self.best_value = None
            self.best_epoch = None

        return state

    def resume_if_possible(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        prefer: str = "last",   # "last" or "best"
        map_location: Optional[str] = "cpu",
    ) -> int:
        ckpt_name = f"{prefer}.pt"
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            return 0  # fresh start

        state = self.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_name=ckpt_name,
            map_location=map_location,
        )
        last_epoch = int(state.get("epoch", 0))
        return last_epoch + 1  # 다음 epoch부터 시작
