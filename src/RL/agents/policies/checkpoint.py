"""Safe checkpoint storage for the Xonotic visual policy."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import torch

from RL.actions.contracts import DiscreteAction
from RL.agents.policies.visual_network import (
    VisualPolicyNetwork,
)


CHECKPOINT_FORMAT_VERSION = 1

_MODEL_CONFIG_FIELDS = {
    "frame_stack",
    "rgb_channels",
    "frame_height",
    "frame_width",
    "hidden_dim",
    "action_count",
}


@dataclass(frozen=True)
class LoadedVisualPolicyCheckpoint:
    """A validated model and its inspectable metadata."""

    path: Path
    model: VisualPolicyNetwork
    policy_name: str
    policy_version: str
    created_at: str
    metadata: Mapping[str, Any]


def _require_nonempty_string(
    value: object,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{field_name} must be a nonempty string"
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


def _normalize_json_value(
    value: object,
    *,
    field_name: str,
) -> Any:
    """Return a JSON-compatible copy of metadata."""

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"{field_name} must contain finite numbers"
            )

        return float(value)

    if isinstance(value, str):
        return value

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
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
                field_name=f"{field_name}[]",
            )
            for item in value
        ]

    raise TypeError(
        f"{field_name} must be JSON-compatible"
    )


def _model_config(
    model: VisualPolicyNetwork,
) -> dict[str, int]:
    return {
        "frame_stack": model.frame_stack,
        "rgb_channels": model.rgb_channels,
        "frame_height": model.frame_height,
        "frame_width": model.frame_width,
        "hidden_dim": model.hidden_dim,
        "action_count": model.action_count,
    }


def _validate_model_config(
    raw_config: object,
) -> dict[str, int]:
    if not isinstance(raw_config, Mapping):
        raise ValueError(
            "checkpoint model_config must be a mapping"
        )

    actual_fields = set(raw_config)

    if actual_fields != _MODEL_CONFIG_FIELDS:
        raise ValueError(
            "checkpoint model_config fields are invalid"
        )

    config = {
        field_name: _require_positive_integer(
            raw_config[field_name],
            field_name=(
                f"model_config.{field_name}"
            ),
        )
        for field_name in sorted(
            _MODEL_CONFIG_FIELDS
        )
    }

    expected_action_count = len(
        tuple(DiscreteAction)
    )

    if config["action_count"] != (
        expected_action_count
    ):
        raise ValueError(
            "checkpoint action count does not match "
            "DiscreteAction"
        )

    return config


def _expected_action_names() -> list[str]:
    return [
        action.name
        for action in DiscreteAction
    ]


def save_visual_policy_checkpoint(
    model: VisualPolicyNetwork,
    path: str | Path,
    *,
    policy_name: str,
    policy_version: str,
    metadata: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> Path:
    """Atomically save a portable inference checkpoint."""

    if not isinstance(
        model,
        VisualPolicyNetwork,
    ):
        raise TypeError(
            "model must be a VisualPolicyNetwork"
        )

    if model.action_count != len(
        tuple(DiscreteAction)
    ):
        raise ValueError(
            "model action count must match "
            "DiscreteAction"
        )

    validated_policy_name = (
        _require_nonempty_string(
            policy_name,
            field_name="policy_name",
        )
    )

    validated_policy_version = (
        _require_nonempty_string(
            policy_version,
            field_name="policy_version",
        )
    )

    if created_at is None:
        validated_created_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    else:
        validated_created_at = (
            _require_nonempty_string(
                created_at,
                field_name="created_at",
            )
        )

    raw_metadata: Mapping[str, Any] = (
        {}
        if metadata is None
        else metadata
    )

    if not isinstance(raw_metadata, Mapping):
        raise TypeError(
            "metadata must be a mapping"
        )

    normalized_metadata = _normalize_json_value(
        raw_metadata,
        field_name="metadata",
    )

    if not isinstance(
        normalized_metadata,
        dict,
    ):
        raise TypeError(
            "metadata must normalize to a dictionary"
        )

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor
        in model.state_dict().items()
    }

    payload = {
        "format_version": (
            CHECKPOINT_FORMAT_VERSION
        ),
        "policy_name": (
            validated_policy_name
        ),
        "policy_version": (
            validated_policy_version
        ),
        "created_at": (
            validated_created_at
        ),
        "torch_version": str(
            torch.__version__
        ),
        "action_names": (
            _expected_action_names()
        ),
        "model_config": (
            _model_config(model)
        ),
        "metadata": normalized_metadata,
        "state_dict": state_dict,
    }

    temporary_path = checkpoint_path.with_name(
        "."
        + checkpoint_path.name
        + "."
        + uuid4().hex
        + ".tmp"
    )

    try:
        with temporary_path.open("wb") as file:
            torch.save(
                payload,
                file,
            )
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


def load_visual_policy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedVisualPolicyCheckpoint:
    """Load and strictly validate an inference checkpoint."""

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            str(checkpoint_path)
        )

    resolved_device = torch.device(
        device
    )

    if (
        resolved_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable"
        )

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(payload, Mapping):
        raise ValueError(
            "checkpoint payload must be a mapping"
        )

    format_version = payload.get(
        "format_version"
    )

    if (
        isinstance(format_version, bool)
        or format_version
        != CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            "unsupported checkpoint format version"
        )

    policy_name = _require_nonempty_string(
        payload.get("policy_name"),
        field_name="policy_name",
    )

    policy_version = _require_nonempty_string(
        payload.get("policy_version"),
        field_name="policy_version",
    )

    created_at = _require_nonempty_string(
        payload.get("created_at"),
        field_name="created_at",
    )

    action_names = payload.get(
        "action_names"
    )

    if action_names != _expected_action_names():
        raise ValueError(
            "checkpoint action mapping does not match "
            "DiscreteAction"
        )

    config = _validate_model_config(
        payload.get("model_config")
    )

    raw_metadata = payload.get(
        "metadata"
    )

    if not isinstance(raw_metadata, Mapping):
        raise ValueError(
            "checkpoint metadata must be a mapping"
        )

    normalized_metadata = _normalize_json_value(
        raw_metadata,
        field_name="metadata",
    )

    if not isinstance(
        normalized_metadata,
        dict,
    ):
        raise ValueError(
            "checkpoint metadata is invalid"
        )

    state_dict = payload.get(
        "state_dict"
    )

    if not isinstance(state_dict, Mapping):
        raise ValueError(
            "checkpoint state_dict must be a mapping"
        )

    model = VisualPolicyNetwork(
        **config
    )

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except RuntimeError as error:
        raise ValueError(
            "checkpoint state_dict is incompatible "
            "with the model"
        ) from error

    model.to(resolved_device)
    model.eval()

    return LoadedVisualPolicyCheckpoint(
        path=checkpoint_path,
        model=model,
        policy_name=policy_name,
        policy_version=policy_version,
        created_at=created_at,
        metadata=MappingProxyType(
            normalized_metadata
        ),
    )
