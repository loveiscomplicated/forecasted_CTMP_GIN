from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.models.discharge_predictor.model import MultiTaskDischargePredictor
from src.models.discharge_predictor.risk_heads import resolve_risk_head_selection


def normalize_forecasted_discharge_cfg(
    full_cfg: dict[str, Any],
    forecast_cfg: dict[str, Any],
) -> dict[str, Any]:
    out = dict(forecast_cfg)
    soft_cfg = dict(out.get("soft_discharge", {}))
    forecast_input_encoder = str(
        full_cfg.get("model", {}).get("params", {}).get("forecast_input_encoder", "entity_embedding")
    ).lower()
    is_distribution_forecast_model = (
        full_cfg.get("model", {}).get("name") in {"ctmp_gin", "gin"}
        and bool(
            full_cfg.get("forecasted_pipeline", {}).get("enabled", False)
            or full_cfg.get("joint_forecast_pipeline", {}).get("enabled", False)
        )
    )
    if is_distribution_forecast_model and forecast_input_encoder == "distribution":
        out["mode"] = "soft"
        soft_cfg["enabled"] = True
        soft_cfg["heads"] = "all"
        soft_cfg["use_probs_cache"] = True
        soft_cfg["save_probs"] = True
    out["soft_discharge"] = soft_cfg
    return out


class ForecastedDischargeProvider:
    """Predict discharge-side `_D` variables and replace them in the input tensor."""

    def __init__(self, cfg: dict[str, Any], dataset, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.checkpoint_path = str(cfg["checkpoint_path"])
        self.mode = str(cfg.get("mode", "hard")).lower()
        self.soft_cfg = dict(cfg.get("soft_discharge", {}))
        self.soft_enabled = bool(self.soft_cfg.get("enabled", False))
        self.temperature = float(self.soft_cfg.get("temperature", 1.0))
        self.save_logits = bool(self.soft_cfg.get("save_logits", True))
        self.save_probs = bool(self.soft_cfg.get("save_probs", True))
        self.use_probs_cache = bool(self.soft_cfg.get("use_probs_cache", True))
        self.debug_forward_once = bool(self.soft_cfg.get("debug_forward_once", False))

        ckpt = torch.load(self.checkpoint_path, map_location=device)
        predictor_cfg = ckpt.get("cfg", {})
        schema = ckpt.get("schema", {})
        state_dict = ckpt["model_state_dict"]

        col_list, col_dims, ad_col_index, dis_col_index = dataset.col_info
        self.ad_idx_t, ad_col_dims = self._resolve_admission_schema(
            col_list,
            col_dims,
            ad_col_index,
            schema,
        )

        discharge_name_to_index = {
            str(col_list[idx]): int(idx) for idx in dis_col_index if idx is not None
        }
        discharge_dim_map = {
            str(col_list[idx]): int(col_dims[idx]) for idx in dis_col_index if idx is not None
        }

        target_names = schema.get("target_col_names")
        target_dims = schema.get("target_col_dims")
        if target_names is None or target_dims is None:
            target_names, target_dims = self._infer_targets_from_state_dict(state_dict)
        if not set(target_names).issubset(discharge_name_to_index):
            missing = sorted(set(target_names) - set(discharge_name_to_index))
            raise ValueError(
                "Discharge predictor targets are not present in dataset discharge columns. "
                f"missing={missing}"
            )

        for name, dim in zip(target_names, target_dims):
            expected_dim = discharge_dim_map[name]
            dim = int(dim)
            if dim != expected_dim:
                raise ValueError(
                    f"Cardinality mismatch for {name}: predictor={dim} dataset={expected_dim}"
                )

        model_params = dict(predictor_cfg.get("model", {}).get("params", {}))
        self.model = MultiTaskDischargePredictor(
            ad_col_dims=ad_col_dims,
            target_col_names=target_names,
            target_col_dims=target_dims,
            **model_params,
        ).to(device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.target_names = target_names
        self.target_dims = [int(dim) for dim in target_dims]
        self.target_name_to_col_idx = {
            name: int(discharge_name_to_index[name]) for name in self.target_names
        }
        self.class_to_embedding_idx = {
            name: torch.arange(int(dim), dtype=torch.long) for name, dim in zip(self.target_names, self.target_dims)
        }
        self.target_col_index = torch.tensor(
            [discharge_name_to_index[name] for name in target_names],
            dtype=torch.long,
            device=device,
        )
        self.soft_head_names = self._resolve_soft_head_names()
        self._validate_mode()

    def _validate_mode(self) -> None:
        if self.mode not in {"hard", "soft", "mixed"}:
            raise ValueError(f"Unsupported forecasted_discharge.mode: {self.mode}")
        if self.mode == "hard":
            return
        if not self.soft_enabled:
            raise ValueError(
                f"forecasted_discharge.mode={self.mode} requires soft_discharge.enabled=true"
            )
        if not self.soft_head_names:
            raise ValueError(
                f"forecasted_discharge.mode={self.mode} resolved no soft heads from soft_discharge.heads"
            )

    def _resolve_soft_head_names(self) -> list[str]:
        return resolve_risk_head_selection(
            self.soft_cfg.get("heads", "all"),
            available_heads=self.target_names,
            mode="legacy_or_named_set",
            allow_all=True,
            field_name="forecasted_discharge.soft_discharge.heads",
        )

    def _infer_targets_from_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> tuple[list[str], list[int]]:
        target_names: list[str] = []
        target_dims: list[int] = []
        for key, value in state_dict.items():
            if not key.startswith("heads.") or not key.endswith(".weight"):
                continue
            target_names.append(key[len("heads.") : -len(".weight")])
            target_dims.append(int(value.shape[0]))
        if not target_names:
            raise ValueError("No discharge predictor heads found in checkpoint")
        return target_names, target_dims

    def _resolve_admission_schema(
        self,
        col_list,
        col_dims,
        ad_col_index,
        schema: dict[str, Any],
    ) -> tuple[torch.Tensor, list[int]]:
        dataset_ad_names = [str(col_list[i]) for i in ad_col_index]
        dataset_ad_dims = [int(col_dims[i]) for i in ad_col_index]
        dataset_ad_name_to_pos = {
            name: int(idx) for name, idx in zip(dataset_ad_names, ad_col_index)
        }

        schema_ad_names = schema.get("admission_col_names")
        schema_ad_dims = schema.get("admission_col_dims")
        if schema_ad_names is None:
            return (
                torch.tensor(ad_col_index, dtype=torch.long, device=self.device),
                dataset_ad_dims,
            )

        schema_ad_names = [str(name) for name in schema_ad_names]
        missing = [name for name in schema_ad_names if name not in dataset_ad_name_to_pos]
        if missing:
            raise ValueError(
                "Discharge checkpoint admission columns are not present in dataset admission columns. "
                f"missing={missing}"
            )

        resolved_ad_index = [dataset_ad_name_to_pos[name] for name in schema_ad_names]
        if schema_ad_dims is None:
            resolved_ad_dims = [int(col_dims[idx]) for idx in resolved_ad_index]
        else:
            resolved_ad_dims = [int(v) for v in schema_ad_dims]
            current_dims = [int(col_dims[idx]) for idx in resolved_ad_index]
            for name, ckpt_dim, current_dim in zip(schema_ad_names, resolved_ad_dims, current_dims):
                if ckpt_dim != current_dim:
                    raise ValueError(
                        f"Discharge admission cardinality mismatch for {name}: "
                        f"checkpoint={ckpt_dim} dataset={current_dim}"
                    )
        return (
            torch.tensor(resolved_ad_index, dtype=torch.long, device=self.device),
            resolved_ad_dims,
        )

    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        x_replaced, _ = self.predict_with_cache_payload(x_batch)
        return x_replaced

    def predict_with_cache_payload(
        self, x_batch: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        with torch.no_grad():
            ad_x = torch.index_select(x_batch.long(), dim=1, index=self.ad_idx_t)
            outputs = self.model(ad_x)
            pred_cols = []
            soft_payload_heads: dict[str, Any] = {}
            for name in self.target_names:
                logits = outputs[name]
                hard = torch.argmax(logits, dim=1)
                pred_cols.append(hard.to(dtype=x_batch.dtype))
                if self.mode == "hard" or name not in self.soft_head_names:
                    continue

                probs = F.softmax(logits / float(self.temperature), dim=1)
                head_payload: dict[str, Any] = {
                    "hard": hard.detach().to(dtype=torch.long).cpu(),
                    "target_col_idx": int(self.target_name_to_col_idx[name]),
                    "num_classes": int(logits.shape[1]),
                    "class_to_embedding_idx": self.class_to_embedding_idx[name].clone(),
                }
                if self.save_logits:
                    head_payload["logits"] = logits.detach().to(dtype=torch.float32).cpu()
                if self.save_probs or self.use_probs_cache:
                    head_payload["probs"] = probs.detach().to(dtype=torch.float32).cpu()
                soft_payload_heads[name] = head_payload

            pred_tensor = torch.stack(pred_cols, dim=1)
            x_replaced = x_batch.clone()
            x_replaced[:, self.target_col_index] = pred_tensor

            if self.mode == "hard":
                return x_replaced, None

            payload = {
                "head_names": list(self.target_names),
                "soft_head_names": list(self.soft_head_names),
                "heads": soft_payload_heads,
                "metadata": {
                    "target_names": list(self.target_names),
                    "target_to_col_idx": dict(self.target_name_to_col_idx),
                    "mode": self.mode,
                    "temperature": float(self.temperature),
                    "save_logits": bool(self.save_logits),
                    "save_probs": bool(self.save_probs),
                    "use_probs_cache": bool(self.use_probs_cache),
                },
            }
            return x_replaced, payload

    def describe_soft_config(self) -> dict[str, Any]:
        return {
            "enabled": self.mode != "hard" and self.soft_enabled,
            "mode": self.mode,
            "temperature": float(self.temperature),
            "head_names": list(self.target_names),
            "soft_head_names": list(self.soft_head_names),
            "target_to_col_idx": dict(self.target_name_to_col_idx),
            "target_dims": {name: int(dim) for name, dim in zip(self.target_names, self.target_dims)},
        }


def build_forecasted_discharge_provider(cfg: dict[str, Any], dataset, device: torch.device):
    forecast_cfg = normalize_forecasted_discharge_cfg(cfg, cfg.get("forecasted_discharge", {}))
    if not bool(forecast_cfg.get("enabled", False)):
        return None
    if "checkpoint_path" not in forecast_cfg or forecast_cfg["checkpoint_path"] in {None, ""}:
        raise ValueError(
            "forecasted_discharge.enabled=true requires forecasted_discharge.checkpoint_path"
        )
    provider = ForecastedDischargeProvider(forecast_cfg, dataset, device)
    print(
        "Forecasted discharge enabled: "
        f"checkpoint={provider.checkpoint_path} mode={provider.mode} "
        f"soft_heads={provider.soft_head_names if provider.mode != 'hard' else []}"
    )
    return provider
