"""Atomic checkpoints for hierarchical behavior-cloning training."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import torch

from RL.agents.policies.hierarchical_recurrent import (
    HierarchicalRecurrentPolicy,
)
from RL.training.imitation.hierarchical_behavior_cloning import (
    HierarchicalBehaviorCloningTrainer,
    HierarchicalClassWeights,
    HierarchicalLossWeights,
)


LEGACY_HIERARCHICAL_CHECKPOINT_FORMAT_VERSION = 1
CLASS_WEIGHT_HIERARCHICAL_CHECKPOINT_FORMAT_VERSION = 2
TELEMETRY_HIERARCHICAL_CHECKPOINT_FORMAT_VERSION = 3
HIERARCHICAL_CHECKPOINT_FORMAT_VERSION = (
    TELEMETRY_HIERARCHICAL_CHECKPOINT_FORMAT_VERSION
)
_CHECKPOINT_KIND = "hierarchical_behavior_cloning"

_MODEL_CONFIG_FIELDS_V2 = frozenset(
    {
        "frame_stack",
        "rgb_channels",
        "frame_height",
        "frame_width",
        "visual_feature_dim",
        "previous_action_dim",
        "mode_embedding_dim",
        "recurrent_input_dim",
        "recurrent_hidden_dim",
        "recurrent_layers",
        "game_mode_count",
        "tactical_intent_count",
    }
)

_MODEL_CONFIG_FIELDS_V3 = (
    _MODEL_CONFIG_FIELDS_V2
    | frozenset(
        {
            "telemetry_feature_count",
            "telemetry_feature_dim",
            "telemetry_weapon_count",
            "telemetry_weapon_embedding_dim",
        }
    )
)

_MODEL_CONFIG_FIELDS = _MODEL_CONFIG_FIELDS_V3

_TRAINER_CONFIG_FIELDS_V1 = frozenset(
    {
        "loss_weights",
        "max_gradient_norm",
        "optimizer_step_count",
    }
)

_TRAINER_CONFIG_FIELDS_V2 = frozenset(
    {
        "loss_weights",
        "class_weights",
        "max_gradient_norm",
        "optimizer_step_count",
    }
)

_LOSS_WEIGHT_FIELDS = frozenset(
    {
        "forward",
        "strafe",
        "mouse",
        "fire",
        "jump",
        "weapon",
        "legacy_action",
    }
)


_CLASS_WEIGHT_FIELDS = frozenset(
    {
        "fire_positive",
        "jump_positive",
        "weapon_previous",
        "weapon_neutral",
        "weapon_next",
    }
)

_CHECKPOINT_FIELDS = frozenset(
    {
        "format_version",
        "checkpoint_kind",
        "created_at",
        "policy_name",
        "policy_version",
        "model_config",
        "model_state_dict",
        "optimizer_type",
        "optimizer_state_dict",
        "trainer_config",
        "dataset_metadata",
        "metadata",
    }
)


@dataclass(frozen=True)
class LoadedHierarchicalCheckpoint:
    """Strictly reconstructed hierarchical training state."""

    path: Path
    model: HierarchicalRecurrentPolicy
    optimizer: torch.optim.Adam
    trainer: HierarchicalBehaviorCloningTrainer
    policy_name: str
    policy_version: str
    created_at: str
    dataset_metadata: Mapping[str, object]
    metadata: Mapping[str, object]


def _require_nonempty_string(
    value: object,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a nonempty string"
        )

    return value


def _require_nonnegative_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{field_name} must be a nonnegative integer"
        )

    return value


def _require_positive_integer(
    value: object,
    *,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{field_name} must be a positive integer"
        )

    return value


def _require_positive_float(
    value: object,
    *,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(
            f"{field_name} must be finite and positive"
        )

    return float(value)


def _require_exact_fields(
    value: object,
    *,
    expected_fields: frozenset[str],
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field_name} must be a mapping"
        )

    actual_fields = set(value)

    if actual_fields != expected_fields:
        raise ValueError(
            f"{field_name} fields do not match "
            "the checkpoint contract"
        )

    return value


def _normalize_json_value(
    value: object,
    *,
    field_name: str,
) -> object:
    if value is None or isinstance(
        value,
        (bool, str, int),
    ):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"{field_name} contains a nonfinite number"
            )

        return value

    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{field_name} keys must be strings"
                )

            normalized[key] = _normalize_json_value(
                item,
                field_name=f"{field_name}.{key}",
            )

        return normalized

    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(
                item,
                field_name=(
                    f"{field_name}[{index}]"
                ),
            )
            for index, item in enumerate(value)
        ]

    raise ValueError(
        f"{field_name} contains an unsupported value"
    )


def _normalize_metadata(
    value: Mapping[str, object] | None,
    *,
    field_name: str,
) -> dict[str, object]:
    raw: Mapping[str, object] = (
        {} if value is None else value
    )

    normalized = _normalize_json_value(
        raw,
        field_name=field_name,
    )

    if not isinstance(normalized, dict):
        raise ValueError(
            f"{field_name} must normalize to a mapping"
        )

    return normalized


def _cpu_clone(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return (
            value.detach()
            .to(device="cpu")
            .contiguous()
            .clone()
        )

    if isinstance(value, Mapping):
        return {
            key: _cpu_clone(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _cpu_clone(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _cpu_clone(item)
            for item in value
        )

    return value


def _move_to_device(
    value: object,
    device: torch.device,
) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)

    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _move_to_device(
                item,
                device,
            )

        return value

    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _move_to_device(
                item,
                device,
            )

        return value

    if isinstance(value, tuple):
        return tuple(
            _move_to_device(item, device)
            for item in value
        )

    return value


def _model_config(
    model: HierarchicalRecurrentPolicy,
) -> dict[str, int]:
    if not isinstance(
        model,
        HierarchicalRecurrentPolicy,
    ):
        raise TypeError(
            "model must be HierarchicalRecurrentPolicy"
        )

    return {
        "frame_stack": model.frame_stack,
        "rgb_channels": model.rgb_channels,
        "frame_height": model.frame_height,
        "frame_width": model.frame_width,
        "visual_feature_dim": (
            model.visual_feature_dim
        ),
        "previous_action_dim": (
            model.previous_action_dim
        ),
        "mode_embedding_dim": (
            model.mode_embedding_dim
        ),
        "recurrent_input_dim": (
            model.recurrent_input_dim
        ),
        "recurrent_hidden_dim": (
            model.recurrent_hidden_dim
        ),
        "recurrent_layers": (
            model.recurrent_layers
        ),
        "game_mode_count": (
            model.game_mode_count
        ),
        "tactical_intent_count": (
            model.tactical_intent_count
        ),
        "telemetry_feature_count": (
            model.telemetry_feature_count
        ),
        "telemetry_feature_dim": (
            model.telemetry_feature_dim
        ),
        "telemetry_weapon_count": (
            model.telemetry_weapon_count
        ),
        "telemetry_weapon_embedding_dim": (
            model.telemetry_weapon_embedding_dim
        ),
    }


def _validate_model_config(
    value: object,
    *,
    format_version: int,
) -> dict[str, int]:
    legacy_format = format_version in {
        LEGACY_HIERARCHICAL_CHECKPOINT_FORMAT_VERSION,
        CLASS_WEIGHT_HIERARCHICAL_CHECKPOINT_FORMAT_VERSION,
    }

    if not isinstance(value, Mapping):
        raise ValueError(
            "model_config must be a mapping"
        )

    actual_fields = set(value)

    if legacy_format:
        if actual_fields == set(
            _MODEL_CONFIG_FIELDS_V2
        ):
            raw = value
        elif actual_fields == set(
            _MODEL_CONFIG_FIELDS_V3
        ):
            raw = value

            for field_name in (
                "telemetry_feature_count",
                "telemetry_feature_dim",
                "telemetry_weapon_count",
                "telemetry_weapon_embedding_dim",
            ):
                if raw[field_name] != 0:
                    raise ValueError(
                        "legacy checkpoint cannot "
                        "enable telemetry model fields"
                    )
        else:
            raise ValueError(
                "model_config fields do not match "
                "the checkpoint contract"
            )
    else:
        raw = _require_exact_fields(
            value,
            expected_fields=(
                _MODEL_CONFIG_FIELDS_V3
            ),
            field_name="model_config",
        )

    config = {
        field_name: _require_positive_integer(
            raw[field_name],
            field_name=(
                f"model_config.{field_name}"
            ),
        )
        for field_name in sorted(
            _MODEL_CONFIG_FIELDS_V2
        )
    }

    telemetry_fields = (
        "telemetry_feature_count",
        "telemetry_feature_dim",
        "telemetry_weapon_count",
        "telemetry_weapon_embedding_dim",
    )

    if legacy_format:
        telemetry_values = {
            field_name: 0
            for field_name in telemetry_fields
        }
    else:
        telemetry_values = {
            field_name: _require_nonnegative_integer(
                raw[field_name],
                field_name=(
                    f"model_config.{field_name}"
                ),
            )
            for field_name in telemetry_fields
        }

    values = tuple(
        telemetry_values[field_name]
        for field_name in telemetry_fields
    )

    if any(values) and not all(values):
        raise ValueError(
            "telemetry model dimensions must "
            "either all be zero or all be positive"
        )

    config.update(telemetry_values)

    return config



def _validate_loss_weights(
    value: object,
) -> HierarchicalLossWeights:
    raw = _require_exact_fields(
        value,
        expected_fields=_LOSS_WEIGHT_FIELDS,
        field_name="loss_weights",
    )

    try:
        return HierarchicalLossWeights(
            **{
                field_name: raw[field_name]
                for field_name in (
                    "forward",
                    "strafe",
                    "mouse",
                    "fire",
                    "jump",
                    "weapon",
                    "legacy_action",
                )
            }
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "loss_weights are invalid"
        ) from error


def _validate_class_weights(
    value: object,
) -> HierarchicalClassWeights:
    raw = _require_exact_fields(
        value,
        expected_fields=_CLASS_WEIGHT_FIELDS,
        field_name="class_weights",
    )

    try:
        return HierarchicalClassWeights(
            **{
                field_name: raw[field_name]
                for field_name in (
                    "fire_positive",
                    "jump_positive",
                    "weapon_previous",
                    "weapon_neutral",
                    "weapon_next",
                )
            }
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "class_weights are invalid"
        ) from error


def _validate_created_at(
    value: object,
) -> str:
    created_at = _require_nonempty_string(
        value,
        field_name="created_at",
    )

    try:
        parsed = datetime.fromisoformat(
            created_at
        )
    except ValueError as error:
        raise ValueError(
            "created_at must be ISO-8601"
        ) from error

    if parsed.tzinfo is None:
        raise ValueError(
            "created_at must include a timezone"
        )

    return created_at


def _validate_model_state(
    value: object,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "model_state_dict must be a nonempty mapping"
        )

    state: dict[str, torch.Tensor] = {}

    for name, tensor in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(
                tensor,
                torch.Tensor,
            )
        ):
            raise ValueError(
                "model_state_dict is invalid"
            )

        copied = (
            tensor.detach()
            .to(device="cpu")
            .contiguous()
            .clone()
        )

        if (
            copied.is_floating_point()
            and not bool(
                torch.isfinite(
                    copied
                ).all().item()
            )
        ):
            raise ValueError(
                "model_state_dict contains "
                "a nonfinite tensor"
            )

        state[name] = copied

    return state


def _validate_optimizer_state(
    value: object,
) -> dict[str, object]:
    raw = _require_exact_fields(
        value,
        expected_fields=frozenset(
            {
                "state",
                "param_groups",
            }
        ),
        field_name="optimizer_state_dict",
    )

    if not isinstance(raw["state"], Mapping):
        raise ValueError(
            "optimizer state is invalid"
        )

    if not isinstance(
        raw["param_groups"],
        list,
    ):
        raise ValueError(
            "optimizer parameter groups are invalid"
        )

    cloned = _cpu_clone(dict(raw))

    if not isinstance(cloned, dict):
        raise ValueError(
            "optimizer state could not be copied"
        )

    return cloned


def _resolve_device(
    device: str | torch.device,
) -> torch.device:
    resolved = torch.device(device)

    if (
        resolved.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    return resolved


def save_hierarchical_checkpoint(
    path: str | Path,
    *,
    trainer: HierarchicalBehaviorCloningTrainer,
    policy_name: str,
    policy_version: str,
    dataset_metadata: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically save resumable hierarchical training state."""

    if not isinstance(
        trainer,
        HierarchicalBehaviorCloningTrainer,
    ):
        raise TypeError(
            "trainer must be "
            "HierarchicalBehaviorCloningTrainer"
        )

    if not isinstance(
        trainer.optimizer,
        torch.optim.Adam,
    ):
        raise ValueError(
            "hierarchical checkpoints currently "
            "support only torch.optim.Adam"
        )

    if not isinstance(
        trainer.weights,
        HierarchicalLossWeights,
    ):
        raise TypeError(
            "trainer weights are invalid"
        )

    if not isinstance(
        trainer.class_weights,
        HierarchicalClassWeights,
    ):
        raise TypeError(
            "trainer class weights are invalid"
        )

    checkpoint_path = Path(path)

    if checkpoint_path.exists() and (
        not checkpoint_path.is_file()
    ):
        raise ValueError(
            "checkpoint path must reference a file"
        )

    normalized_dataset_metadata = (
        _normalize_metadata(
            dataset_metadata,
            field_name="dataset_metadata",
        )
    )
    normalized_metadata = _normalize_metadata(
        metadata,
        field_name="metadata",
    )

    payload = {
        "format_version": (
            HIERARCHICAL_CHECKPOINT_FORMAT_VERSION
        ),
        "checkpoint_kind": _CHECKPOINT_KIND,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "policy_name": _require_nonempty_string(
            policy_name,
            field_name="policy_name",
        ),
        "policy_version": _require_nonempty_string(
            policy_version,
            field_name="policy_version",
        ),
        "model_config": _model_config(
            trainer.model
        ),
        "model_state_dict": {
            name: (
                tensor.detach()
                .to(device="cpu")
                .contiguous()
                .clone()
            )
            for name, tensor
            in trainer.model.state_dict().items()
        },
        "optimizer_type": "Adam",
        "optimizer_state_dict": _cpu_clone(
            trainer.optimizer.state_dict()
        ),
        "trainer_config": {
            "loss_weights": asdict(
                trainer.weights
            ),
            "class_weights": asdict(
                trainer.class_weights
            ),
            "max_gradient_norm": (
                trainer.max_gradient_norm
            ),
            "optimizer_step_count": (
                trainer.optimizer_step_count
            ),
        },
        "dataset_metadata": (
            normalized_dataset_metadata
        ),
        "metadata": normalized_metadata,
    }

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = checkpoint_path.with_name(
        "."
        + checkpoint_path.name
        + "."
        + uuid4().hex
        + ".tmp"
    )

    try:
        with temporary_path.open("wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            checkpoint_path,
        )
    finally:
        temporary_path.unlink(
            missing_ok=True
        )

    return checkpoint_path


def load_hierarchical_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedHierarchicalCheckpoint:
    """Load and strictly validate resumable training state."""

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            str(checkpoint_path)
        )

    resolved_device = _resolve_device(device)

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    raw = _require_exact_fields(
        payload,
        expected_fields=_CHECKPOINT_FIELDS,
        field_name="checkpoint",
    )

    format_version = _require_positive_integer(
        raw["format_version"],
        field_name="format_version",
    )

    if format_version not in {
        LEGACY_HIERARCHICAL_CHECKPOINT_FORMAT_VERSION,
        CLASS_WEIGHT_HIERARCHICAL_CHECKPOINT_FORMAT_VERSION,
        HIERARCHICAL_CHECKPOINT_FORMAT_VERSION,
    }:
        raise ValueError(
            "unsupported hierarchical "
            "checkpoint format version"
        )

    if raw["checkpoint_kind"] != (
        _CHECKPOINT_KIND
    ):
        raise ValueError(
            "checkpoint kind is not hierarchical "
            "behavior cloning"
        )

    policy_name = _require_nonempty_string(
        raw["policy_name"],
        field_name="policy_name",
    )
    policy_version = _require_nonempty_string(
        raw["policy_version"],
        field_name="policy_version",
    )
    created_at = _validate_created_at(
        raw["created_at"]
    )

    config = _validate_model_config(
        raw["model_config"],
        format_version=format_version,
    )
    model_state = _validate_model_state(
        raw["model_state_dict"]
    )

    if raw["optimizer_type"] != "Adam":
        raise ValueError(
            "checkpoint optimizer type is unsupported"
        )

    optimizer_state = _validate_optimizer_state(
        raw["optimizer_state_dict"]
    )

    trainer_config_fields = (
        _TRAINER_CONFIG_FIELDS_V1
        if format_version
        == LEGACY_HIERARCHICAL_CHECKPOINT_FORMAT_VERSION
        else _TRAINER_CONFIG_FIELDS_V2
    )

    trainer_config = _require_exact_fields(
        raw["trainer_config"],
        expected_fields=trainer_config_fields,
        field_name="trainer_config",
    )

    loss_weights = _validate_loss_weights(
        trainer_config["loss_weights"]
    )

    class_weights = (
        HierarchicalClassWeights()
        if format_version
        == LEGACY_HIERARCHICAL_CHECKPOINT_FORMAT_VERSION
        else _validate_class_weights(
            trainer_config["class_weights"]
        )
    )
    max_gradient_norm = (
        _require_positive_float(
            trainer_config[
                "max_gradient_norm"
            ],
            field_name="max_gradient_norm",
        )
    )
    optimizer_step_count = (
        _require_nonnegative_integer(
            trainer_config[
                "optimizer_step_count"
            ],
            field_name="optimizer_step_count",
        )
    )

    dataset_metadata = _normalize_metadata(
        raw["dataset_metadata"],
        field_name="dataset_metadata",
    )
    metadata = _normalize_metadata(
        raw["metadata"],
        field_name="metadata",
    )

    model = HierarchicalRecurrentPolicy(
        **config
    )

    try:
        model.load_state_dict(
            model_state,
            strict=True,
        )
    except RuntimeError as error:
        raise ValueError(
            "model state is incompatible "
            "with the checkpoint configuration"
        ) from error

    model.to(resolved_device)

    optimizer = torch.optim.Adam(
        model.parameters()
    )

    try:
        optimizer.load_state_dict(
            optimizer_state
        )
    except (ValueError, RuntimeError) as error:
        raise ValueError(
            "optimizer state is incompatible "
            "with the checkpoint model"
        ) from error

    for optimizer_value in optimizer.state.values():
        _move_to_device(
            optimizer_value,
            resolved_device,
        )

    trainer = HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
        device=resolved_device,
        weights=loss_weights,
        class_weights=class_weights,
        max_gradient_norm=max_gradient_norm,
        optimizer_step_count=(
            optimizer_step_count
        ),
    )

    return LoadedHierarchicalCheckpoint(
        path=checkpoint_path.resolve(),
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        policy_name=policy_name,
        policy_version=policy_version,
        created_at=created_at,
        dataset_metadata=MappingProxyType(
            dataset_metadata
        ),
        metadata=MappingProxyType(
            metadata
        ),
    )
