from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn.functional as F

from src.models.discharge_predictor.los_ordinal_model import LOSOrdinalPredictor
from src.models.discharge_predictor.los_coarse_model import LOSCoarsePredictor
from src.models.discharge_predictor.los_utils import (
    expand_coarse_distribution_to_raw_los,
    get_los_coarse_num_classes,
    infer_los_coarse_breakdown_from_cfg,
    map_coarse_array_to_raw_los,
)
from src.models.discharge_predictor.ordinal_loss import ordinal_logits_to_class


def normalize_forecasted_los_cfg(
    full_cfg: dict[str, Any],
    forecast_cfg: dict[str, Any],
) -> dict[str, Any]:
    out = dict(forecast_cfg)
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
        out["return_type"] = "distribution"
    return out


class ForecastedLOSProvider:
    """Serve LOS predictions from a pretrained admission-only LOS model.

    The provider loads a LOS checkpoint, aligns its expected admission features
    with the active dataset schema, and exposes a callable interface that maps a
    full batch tensor to either:

    - hard LOS class predictions that can replace oracle LOS inputs, or
    - probability distributions over raw LOS classes for soft downstream use.

    The provider hides checkpoint-specific details such as loss head type,
    calibration temperature, coarse-to-raw LOS expansion, and ordinal
    thresholding so callers only need to pass the current batch tensor.
    """

    def __init__(self, cfg: dict[str, Any], dataset, device: torch.device) -> None:
        """Initialize the provider from runtime config and a saved LOS checkpoint.

        Args:
            cfg: `forecasted_los` configuration. Supported keys include:
                - `checkpoint_path`: path to the LOS checkpoint to restore.
                - `variant`: threshold variant name for ordinal hard decisions.
                  Defaults to `"baseline"`.
                - `return_type`: `"hard"` or `"distribution"`. Defaults to
                  `"hard"`.
                - `probability_source`: which head to use when a distribution is
                  requested. Defaults to `"auto"`.
                - `temperature`: optional override for probability calibration.
                - `output_offset`: class index offset for fine-grained hard
                  predictions. Defaults to `1`.
                - `target_mode`: optional hint about the LOS target space.
                - `calibration_path`: optional explicit calibration JSON path.
            dataset: Dataset object exposing `col_info`, used to align admission
                columns between the training checkpoint schema and the current
                runtime dataset.
            device: Device on which the checkpoint and inference tensors should
                live.
        """
        self.cfg = cfg
        self.device = device
        self.variant = str(cfg.get("variant", "baseline"))
        self.return_type = str(cfg.get("return_type", "hard")).lower()
        self.probability_source = str(cfg.get("probability_source", "auto")).lower()
        temperature_override = cfg.get("temperature")
        self.temperature = None if temperature_override in {None, ""} else float(temperature_override)
        self.checkpoint_path = str(cfg["checkpoint_path"])
        self.output_offset = int(cfg.get("output_offset", 1))
        self.target_mode = str(cfg.get("target_mode", "fine")).lower()
        self.calibration_path = cfg.get("calibration_path")
        self.calibration_payload: dict[str, Any] | None = None

        ckpt = torch.load(self.checkpoint_path, map_location=device)
        los_cfg = ckpt.get("cfg", {})
        schema = ckpt.get("schema", {})
        self.loss_type = str(los_cfg.get("loss", {}).get("type", "ordinal_bce"))
        self.los_target_mode = str(los_cfg.get("los_target_mode", los_cfg.get("target_mode", "fine"))).lower()
        self.coarse_breakdown = False
        self.coarse_num_classes: int | None = None
        if self.los_target_mode == "coarse":
            inferred_num_classes = self._infer_los_num_classes(ckpt)
            self.coarse_num_classes = int(inferred_num_classes)
            self.coarse_breakdown = self._resolve_coarse_breakdown(
                los_cfg,
                schema,
                cfg,
                inferred_num_classes,
            )
        self.output_mode = {
            "ordinal_bce": "ordinal",
            "ce": "ce",
            "focal": "ce",
            "focal_alpha": "ce",
            "cb_focal": "ce",
            "hybrid_ce_ordinal": "hybrid",
        }.get(self.loss_type)
        if self.output_mode is None:
            raise ValueError(f"Unsupported LOS loss type in checkpoint: {self.loss_type}")

        self.ad_idx_t, ad_col_dims = self._resolve_admission_schema(dataset, schema)
        model_params = dict(los_cfg.get("model", {}).get("params", {}))
        model_params.pop("output_mode", None)
        if self.los_target_mode == "coarse":
            self.model = self._build_coarse_model(ad_col_dims, los_cfg, cfg, model_params)
        else:
            los_num_classes = int(
                cfg.get("los_num_classes", schema.get("los_num_classes", self._infer_los_num_classes(ckpt)))
            )
            self.model = LOSOrdinalPredictor(
                ad_col_dims=ad_col_dims,
                los_num_classes=los_num_classes,
                output_mode=self.output_mode,
                **model_params,
            ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()

        self.thresholds = None
        if self.loss_type not in {"ce", "focal", "focal_alpha", "cb_focal"} and self.los_target_mode != "coarse":
            los_num_classes = int(
                cfg.get("los_num_classes", schema.get("los_num_classes", self._infer_los_num_classes(ckpt)))
            )
            self.thresholds = self._load_thresholds(cfg, los_num_classes)
        self.temperature = self._resolve_temperature()

    def _load_thresholds(self, cfg: dict[str, Any], los_num_classes: int) -> torch.Tensor:
        """Load ordinal thresholds for hard class decoding.

        Baseline runs use a fixed `0.5` threshold per boundary. Other variants
        read the calibrated thresholds from `calibration.json`.
        """
        if self.variant == "baseline":
            thresholds = [0.5] * (los_num_classes - 1)
        else:
            calibration_path = cfg.get("calibration_path")
            if calibration_path is None:
                run_dir = os.path.dirname(os.path.dirname(self.checkpoint_path))
                calibration_path = os.path.join(run_dir, "calibration.json")
            with open(str(calibration_path), "r", encoding="utf-8") as f:
                calibration = json.load(f)
            thresholds = calibration["variants"][self.variant]["thresholds"]
        return torch.tensor(thresholds, dtype=torch.float32, device=self.device)

    def _load_calibration_payload(self) -> dict[str, Any] | None:
        """Load calibration metadata once and cache it in memory."""
        if self.calibration_payload is not None:
            return self.calibration_payload
        calibration_path = self.calibration_path
        if calibration_path is None:
            run_dir = os.path.dirname(os.path.dirname(self.checkpoint_path))
            calibration_path = os.path.join(run_dir, "calibration.json")
        if not os.path.exists(str(calibration_path)):
            return None
        with open(str(calibration_path), "r", encoding="utf-8") as f:
            self.calibration_payload = json.load(f)
        return self.calibration_payload

    def _resolve_temperature(self) -> float:
        """Choose the temperature used for probability calibration.

        Priority:
        1. Explicit `forecasted_los.temperature` override.
        2. Fitted temperature stored in calibration metadata.
        3. Neutral temperature `1.0`.
        """
        if self.temperature is not None:
            return float(self.temperature)
        calibration = self._load_calibration_payload()
        if calibration is None:
            return 1.0
        temperature_section = calibration.get("temperature", {})
        fitted_temperature = temperature_section.get("fitted")
        if fitted_temperature is None:
            return 1.0
        return float(fitted_temperature)

    def _resolve_admission_schema(
        self,
        dataset,
        schema: dict[str, Any],
    ) -> tuple[torch.Tensor, list[int]]:
        """Match checkpoint admission features against the runtime dataset.

        Returns:
            A tuple of:
            - tensor indices selecting admission columns from the runtime batch
            - per-column categorical cardinalities in checkpoint order

        Raises:
            ValueError: If the checkpoint expects admission columns or
                cardinalities that are incompatible with the active dataset.
        """
        col_list, col_dims, ad_col_index, _ = dataset.col_info
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
                "LOS checkpoint admission columns are not present in dataset admission columns. "
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
                        f"LOS admission cardinality mismatch for {name}: "
                        f"checkpoint={ckpt_dim} dataset={current_dim}"
                    )
        return (
            torch.tensor(resolved_ad_index, dtype=torch.long, device=self.device),
            resolved_ad_dims,
        )

    def _infer_los_num_classes(self, ckpt: dict[str, Any]) -> int:
        """Infer the LOS class count from checkpoint head shapes."""
        state = ckpt["model_state_dict"]
        if self.loss_type in {"ce", "focal", "focal_alpha", "cb_focal"}:
            if "head.weight" in state:
                return int(state["head.weight"].shape[0])
            return int(state["ce_head.weight"].shape[0])
        return int(state["head.weight"].shape[0]) + 1

    def _resolve_coarse_breakdown(
        self,
        los_cfg: dict[str, Any],
        schema: dict[str, Any],
        provider_cfg: dict[str, Any],
        inferred_num_classes: int,
    ) -> bool:
        """Resolve whether a coarse checkpoint uses the long-stay breakdown bins."""
        if inferred_num_classes == get_los_coarse_num_classes(breakdown=True):
            return True
        if inferred_num_classes == get_los_coarse_num_classes(breakdown=False):
            metadata_breakdown = any(
                infer_los_coarse_breakdown_from_cfg(source)
                for source in (los_cfg, schema, provider_cfg)
                if isinstance(source, dict)
            )
            if metadata_breakdown:
                raise ValueError(
                    "Coarse LOS checkpoint metadata requests breakdown bins, but "
                    "the checkpoint head has 6 classes."
                )
            return False
        raise ValueError(
            "Unsupported coarse LOS checkpoint class count: "
            f"{inferred_num_classes}. Expected 6 or 9."
        )

    def _ordinal_logits_to_distribution(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert ordinal boundary logits into a proper class distribution."""
        probs_gt = torch.sigmoid(logits / float(self.temperature))
        batch_size, num_thresholds = probs_gt.shape
        num_classes = num_thresholds + 1
        class_probs = torch.empty(
            (batch_size, num_classes), device=logits.device, dtype=logits.dtype
        )
        class_probs[:, 0] = 1.0 - probs_gt[:, 0]
        if num_thresholds > 1:
            class_probs[:, 1:-1] = probs_gt[:, :-1] - probs_gt[:, 1:]
        class_probs[:, -1] = probs_gt[:, -1]
        class_probs = class_probs.clamp_min(0.0)
        return class_probs / class_probs.sum(dim=1, keepdim=True).clamp_min(1.0e-12)

    def _resolve_probability_source(self) -> str:
        """Resolve `probability_source=auto` into a concrete decoding strategy."""
        source = self.probability_source
        if source == "auto":
            if self.loss_type in {"ce", "focal", "focal_alpha", "cb_focal", "hybrid_ce_ordinal"}:
                source = "calibrated" if self.return_type == "distribution" else "ce"
            else:
                source = "ordinal"
        return source

    def _extract_ce_logits(self, outputs):
        """Return CE-style logits from the checkpoint outputs.

        Raises:
            ValueError: If the active checkpoint does not expose CE logits but a
                CE-based probability source was requested.
        """
        if self.loss_type in {"ce", "focal", "focal_alpha", "cb_focal"}:
            return outputs
        if self.loss_type == "hybrid_ce_ordinal":
            return outputs["ce"]
        raise ValueError(
            "forecasted_los.probability_source requires CE logits, but the LOS "
            f"checkpoint loss type is {self.loss_type!r}. Supported CE-backed "
            "losses are 'ce', 'focal', 'focal_alpha', 'cb_focal', and "
            "'hybrid_ce_ordinal'."
        )

    def _extract_ordinal_logits(self, outputs):
        """Return ordinal logits from the checkpoint outputs."""
        return outputs["ordinal"] if self.loss_type == "hybrid_ce_ordinal" else outputs

    def _softmax_with_source_temperature(self, logits: torch.Tensor, source: str) -> torch.Tensor:
        """Apply softmax using either raw or calibrated temperature."""
        temperature = 1.0 if source in {"ce", "raw"} else float(self.temperature)
        return F.softmax(logits / temperature, dim=1)

    def _outputs_to_distribution(self, outputs) -> torch.Tensor:
        """Convert fine-grained model outputs to raw LOS class probabilities.

        The conversion path depends on both the training loss and the requested
        probability source:
        - CE-like heads use softmax over logits.
        - Ordinal heads reconstruct class probabilities from cumulative logits.
        """
        source = self._resolve_probability_source()
        if source in {"ce", "raw", "calibrated"}:
            return self._softmax_with_source_temperature(self._extract_ce_logits(outputs), source)

        if source == "ordinal":
            return self._ordinal_logits_to_distribution(self._extract_ordinal_logits(outputs))

        raise ValueError(f"Unsupported forecasted_los.probability_source: {self.probability_source}")

    def _build_coarse_model(
        self,
        ad_col_dims: list[int],
        los_cfg: dict[str, Any],
        cfg: dict[str, Any],
        model_params: dict[str, Any],
    ) -> torch.nn.Module:
        """Instantiate the checkpoint-compatible model for coarse LOS targets."""
        num_classes = int(
            self.coarse_num_classes
            or los_cfg.get("num_classes", cfg.get("num_classes", 6))
        )
        if self.loss_type in {"ce", "focal", "focal_alpha", "cb_focal"}:
            return LOSCoarsePredictor(
                ad_col_dims=ad_col_dims,
                num_classes=num_classes,
                **model_params,
            ).to(self.device)
        return LOSOrdinalPredictor(
            ad_col_dims=ad_col_dims,
            los_num_classes=num_classes,
            output_mode=self.output_mode,
            **model_params,
        ).to(self.device)

    def _coarse_outputs_to_coarse_distribution(self, outputs) -> torch.Tensor:
        """Convert coarse-model outputs into coarse-bin probabilities."""
        source = self._resolve_probability_source()
        if self.loss_type in {"ce", "focal", "focal_alpha", "cb_focal"}:
            return self._softmax_with_source_temperature(outputs, source)
        if self.loss_type == "hybrid_ce_ordinal":
            if source in {"ce", "raw", "calibrated"}:
                return self._softmax_with_source_temperature(self._extract_ce_logits(outputs), source)
            if source == "ordinal":
                return self._ordinal_logits_to_distribution(self._extract_ordinal_logits(outputs))
            raise ValueError(
                "Unsupported forecasted_los.probability_source for coarse hybrid: "
                f"{self.probability_source}"
            )
        return self._ordinal_logits_to_distribution(outputs)

    def _coarse_outputs_to_distribution(self, outputs) -> torch.Tensor:
        """Convert coarse LOS outputs to the raw 37-bin LOS distribution."""
        coarse_probs = self._coarse_outputs_to_coarse_distribution(outputs)
        return expand_coarse_distribution_to_raw_los(
            coarse_probs,
            breakdown=self.coarse_breakdown,
        )

    def _select_admission_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Extract admission-only columns expected by the LOS checkpoint."""
        return torch.index_select(x_batch.long(), dim=1, index=self.ad_idx_t)

    def _predict_coarse_hard(self, outputs) -> torch.Tensor:
        """Decode a coarse-target checkpoint to raw LOS hard predictions."""
        if self.loss_type in {"ce", "focal", "focal_alpha", "cb_focal"}:
            pred = torch.argmax(outputs, dim=1).long()
        else:
            pred = ordinal_logits_to_class(self._extract_ordinal_logits(outputs), threshold=0.5).long()
        return map_coarse_array_to_raw_los(pred, breakdown=self.coarse_breakdown)

    def _predict_fine_hard(self, outputs) -> torch.Tensor:
        """Decode a fine-target checkpoint to raw LOS hard predictions."""
        if self.loss_type in {"ce", "focal", "focal_alpha", "cb_focal"}:
            pred = torch.argmax(outputs, dim=1).long()
            return pred + self.output_offset
        pred = ordinal_logits_to_class(self._extract_ordinal_logits(outputs), threshold=self.thresholds).long()
        return pred + self.output_offset

    def _process_outputs(self, outputs) -> torch.Tensor:
        """Route model outputs to the requested return format."""
        if self.return_type == "distribution":
            if self.los_target_mode == "coarse":
                return self._coarse_outputs_to_distribution(outputs)
            return self._outputs_to_distribution(outputs)
        if self.return_type != "hard":
            raise ValueError(
                "Unsupported forecasted_los.return_type: "
                f"{self.return_type}. Supported values are 'hard' and 'distribution'."
            )
        if self.los_target_mode == "coarse":
            return self._predict_coarse_hard(outputs)
        return self._predict_fine_hard(outputs)

    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Run checkpoint inference on a batch and decode the requested output.

        Args:
            x_batch: Full feature batch containing admission and discharge-side
                columns.

        Returns:
            A tensor of either hard LOS predictions or LOS probability
            distributions, depending on `forecasted_los.return_type`.
        """
        with torch.no_grad():
            ad_x = self._select_admission_batch(x_batch)
            outputs = self.model(ad_x)
            return self._process_outputs(outputs)


def build_forecasted_los_provider(cfg: dict[str, Any], dataset, device: torch.device):
    forecast_cfg = normalize_forecasted_los_cfg(cfg, cfg.get("forecasted_los", {}))
    if not bool(forecast_cfg.get("enabled", False)):
        return None
    if "checkpoint_path" not in forecast_cfg or forecast_cfg["checkpoint_path"] in {None, ""}:
        raise ValueError("forecasted_los.enabled=true requires forecasted_los.checkpoint_path")
    if (
        str(forecast_cfg.get("return_type", "hard")).lower() == "distribution"
        and cfg.get("model", {}).get("name") != "ctmp_gin"
    ):
        raise ValueError("forecasted_los.return_type=distribution is currently supported only for ctmp_gin")
    provider = ForecastedLOSProvider(forecast_cfg, dataset, device)
    print(
        "Forecasted LOS enabled: "
        f"checkpoint={provider.checkpoint_path} variant={provider.variant} "
        f"loss_type={provider.loss_type} return_type={provider.return_type} "
        f"coarse_breakdown={provider.coarse_breakdown}"
    )
    return provider
