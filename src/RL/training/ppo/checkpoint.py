"""Atomic, resumable PPO actor-critic training checkpoints."""

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
from RL.agents.policies.actor_critic import (
    VisualActorCriticNetwork,
)
from RL.training.ppo.core import (
    PPOHyperparameters,
    PPOTrainer,
)


PPO_CHECKPOINT_FORMAT_VERSION = 1

_CHECKPOINT_KIND = "ppo_actor_critic_training"

_PAYLOAD_FIELDS = {
    "format_version",
    "checkpoint_kind",
    "policy_name",
    "policy_version",
    "created_at",
    "torch_version",
    "action_names",
    "model_config",
    "model_state_dict",
    "optimizer_type",
    "optimizer_state_dict",
    "trainer_config",
    "progress",
    "metadata",
    "cpu_rng_state",
    "cuda_rng_states",
}

_MODEL_CONFIG_FIELDS = {
    "frame_stack",
    "rgb_channels",
    "frame_height",
    "frame_width",
    "hidden_dim",
    "action_count",
}

_TRAINER_CONFIG_FIELDS = {
    "clip_epsilon",
    "value_coefficient",
    "entropy_coefficient",
    "value_clip_epsilon",
    "max_gradient_norm",
    "optimizer_step_count",
}

_PROGRESS_FIELDS = {
    "rollout_count",
    "environment_step_count",
    "completed_episode_count",
    "cumulative_reward",
}

_SUPPORTED_OPTIMIZERS = {
    torch.optim.Adam: "Adam",
    torch.optim.AdamW: "AdamW",
}

_OPTIMIZER_CLASSES = {
    value: key
    for key, value in _SUPPORTED_OPTIMIZERS.items()
}


def _nonempty_string(
    value: object,
    *,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{field_name} must not be empty"
        )

    return value


def _nonnegative_integer(
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
            f"{field_name} must be "
            "a nonnegative integer"
        )

    return value


def _positive_integer(
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
            f"{field_name} must be "
            "a positive integer"
        )

    return value


def _finite_number(
    value: object,
    *,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float),
        )
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"{field_name} must be finite"
        )

    return float(value)


def _normalize_json_value(
    value: object,
    *,
    field_name: str,
) -> Any:
    if value is None or isinstance(
        value,
        (str, bool, int),
    ):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"{field_name} contains "
                "a non-finite number"
            )

        return value

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{field_name} keys "
                    "must be strings"
                )

            normalized[key] = (
                _normalize_json_value(
                    item,
                    field_name=(
                        f"{field_name}.{key}"
                    ),
                )
            )

        return normalized

    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(
                item,
                field_name=field_name,
            )
            for item in value
        ]

    raise TypeError(
        f"{field_name} must be "
        "JSON-compatible"
    )


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (
            value.detach()
            .to(device="cpu")
            .contiguous()
            .clone()
        )

    if isinstance(value, Mapping):
        return {
            key: _cpu_tree(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _cpu_tree(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _cpu_tree(item)
            for item in value
        )

    return value


def _expected_action_names() -> list[str]:
    return [
        action.name
        for action in DiscreteAction
    ]


def _model_config(
    model: VisualActorCriticNetwork,
) -> dict[str, int]:
    return {
        "frame_stack": model.frame_stack,
        "rgb_channels": model.rgb_channels,
        "frame_height": model.frame_height,
        "frame_width": model.frame_width,
        "hidden_dim": model.hidden_dim,
        "action_count": model.action_count,
    }


def _trainer_config(
    trainer: PPOTrainer,
) -> dict[str, Any]:
    settings = trainer.hyperparameters

    return {
        "clip_epsilon": (
            settings.clip_epsilon
        ),
        "value_coefficient": (
            settings.value_coefficient
        ),
        "entropy_coefficient": (
            settings.entropy_coefficient
        ),
        "value_clip_epsilon": (
            settings.value_clip_epsilon
        ),
        "max_gradient_norm": (
            trainer.max_gradient_norm
        ),
        "optimizer_step_count": (
            trainer.optimizer_step_count
        ),
    }


def _validate_mapping_fields(
    value: object,
    *,
    expected_fields: set[str],
    field_name: str,
) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field_name} must be a mapping"
        )

    actual_fields = set(value)

    if actual_fields != expected_fields:
        raise ValueError(
            f"{field_name} fields are invalid"
        )

    return value


def _validate_model_config(
    value: object,
) -> dict[str, int]:
    raw = _validate_mapping_fields(
        value,
        expected_fields=_MODEL_CONFIG_FIELDS,
        field_name="model_config",
    )

    config = {
        field_name: _positive_integer(
            raw[field_name],
            field_name=(
                f"model_config.{field_name}"
            ),
        )
        for field_name
        in sorted(_MODEL_CONFIG_FIELDS)
    }

    if config["action_count"] != len(
        tuple(DiscreteAction)
    ):
        raise ValueError(
            "checkpoint action count does not "
            "match DiscreteAction"
        )

    return config


def _validate_trainer_config(
    value: object,
) -> tuple[
    PPOHyperparameters,
    float,
    int,
]:
    raw = _validate_mapping_fields(
        value,
        expected_fields=_TRAINER_CONFIG_FIELDS,
        field_name="trainer_config",
    )

    value_clip = raw[
        "value_clip_epsilon"
    ]

    if value_clip is not None:
        value_clip = _finite_number(
            value_clip,
            field_name=(
                "trainer_config."
                "value_clip_epsilon"
            ),
        )

    hyperparameters = PPOHyperparameters(
        clip_epsilon=_finite_number(
            raw["clip_epsilon"],
            field_name=(
                "trainer_config."
                "clip_epsilon"
            ),
        ),
        value_coefficient=_finite_number(
            raw["value_coefficient"],
            field_name=(
                "trainer_config."
                "value_coefficient"
            ),
        ),
        entropy_coefficient=_finite_number(
            raw["entropy_coefficient"],
            field_name=(
                "trainer_config."
                "entropy_coefficient"
            ),
        ),
        value_clip_epsilon=value_clip,
    )

    max_gradient_norm = _finite_number(
        raw["max_gradient_norm"],
        field_name=(
            "trainer_config."
            "max_gradient_norm"
        ),
    )

    if max_gradient_norm <= 0.0:
        raise ValueError(
            "trainer_config.max_gradient_norm "
            "must be positive"
        )

    optimizer_step_count = (
        _nonnegative_integer(
            raw["optimizer_step_count"],
            field_name=(
                "trainer_config."
                "optimizer_step_count"
            ),
        )
    )

    return (
        hyperparameters,
        max_gradient_norm,
        optimizer_step_count,
    )


def _validate_state_dict(
    value: object,
    *,
    field_name: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field_name} must be a mapping"
        )

    if not value:
        raise ValueError(
            f"{field_name} must not be empty"
        )

    normalized: dict[
        str,
        torch.Tensor,
    ] = {}

    for key, tensor in value.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{field_name} keys "
                "must be strings"
            )

        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            raise ValueError(
                f"{field_name} values "
                "must be tensors"
            )

        normalized[key] = tensor

    return normalized


def _validate_optimizer_state(
    value: object,
) -> Mapping[str, Any]:
    raw = _validate_mapping_fields(
        value,
        expected_fields={
            "state",
            "param_groups",
        },
        field_name="optimizer_state_dict",
    )

    if not isinstance(
        raw["state"],
        Mapping,
    ):
        raise ValueError(
            "optimizer state must be a mapping"
        )

    if not isinstance(
        raw["param_groups"],
        list,
    ):
        raise ValueError(
            "optimizer param_groups must be a list"
        )

    if not raw["param_groups"]:
        raise ValueError(
            "optimizer param_groups "
            "must not be empty"
        )

    return raw


def _validate_rng_state(
    value: object,
    *,
    field_name: str,
) -> torch.Tensor:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise ValueError(
            f"{field_name} must be a tensor"
        )

    if (
        value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() <= 0
    ):
        raise ValueError(
            f"{field_name} is invalid"
        )

    return (
        value.detach()
        .to(device="cpu")
        .contiguous()
        .clone()
    )


@dataclass(frozen=True)
class PPOTrainingProgress:
    """Serializable bounded-training progress counters."""

    rollout_count: int = 0
    environment_step_count: int = 0
    completed_episode_count: int = 0
    cumulative_reward: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "rollout_count",
            "environment_step_count",
            "completed_episode_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_integer(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )

        object.__setattr__(
            self,
            "cumulative_reward",
            _finite_number(
                self.cumulative_reward,
                field_name="cumulative_reward",
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "rollout_count": (
                self.rollout_count
            ),
            "environment_step_count": (
                self.environment_step_count
            ),
            "completed_episode_count": (
                self.completed_episode_count
            ),
            "cumulative_reward": (
                self.cumulative_reward
            ),
        }


def _validate_progress(
    value: object,
) -> PPOTrainingProgress:
    raw = _validate_mapping_fields(
        value,
        expected_fields=_PROGRESS_FIELDS,
        field_name="progress",
    )

    return PPOTrainingProgress(
        rollout_count=raw["rollout_count"],
        environment_step_count=(
            raw["environment_step_count"]
        ),
        completed_episode_count=(
            raw["completed_episode_count"]
        ),
        cumulative_reward=(
            raw["cumulative_reward"]
        ),
    )


@dataclass(frozen=True)
class LoadedPPOTrainingCheckpoint:
    """Fully restored PPO model, optimizer, trainer, and metadata."""

    path: Path
    trainer: PPOTrainer
    policy_name: str
    policy_version: str
    created_at: str
    progress: PPOTrainingProgress
    metadata: Mapping[str, Any]
    cpu_rng_state: torch.Tensor
    cuda_rng_states: tuple[
        torch.Tensor,
        ...,
    ]

    @property
    def model(
        self,
    ) -> VisualActorCriticNetwork:
        return self.trainer.model

    @property
    def optimizer(
        self,
    ) -> torch.optim.Optimizer:
        return self.trainer.optimizer


def save_ppo_training_checkpoint(
    path: str | Path,
    trainer: PPOTrainer,
    *,
    progress: (
        PPOTrainingProgress | None
    ) = None,
    policy_name: str = (
        "xonotic-visual-actor-critic"
    ),
    policy_version: str = "v1",
    metadata: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> Path:
    """Atomically save a resumable PPO training checkpoint."""

    if not isinstance(
        trainer,
        PPOTrainer,
    ):
        raise TypeError(
            "trainer must be a PPOTrainer"
        )

    if not isinstance(
        trainer.model,
        VisualActorCriticNetwork,
    ):
        raise TypeError(
            "trainer model must be a "
            "VisualActorCriticNetwork"
        )

    optimizer_class = type(
        trainer.optimizer
    )

    if optimizer_class not in (
        _SUPPORTED_OPTIMIZERS
    ):
        raise ValueError(
            "checkpoint supports only "
            "Adam and AdamW optimizers"
        )

    model_parameter_ids = {
        id(parameter)
        for parameter
        in trainer.model.parameters()
    }

    optimizer_parameter_ids = {
        id(parameter)
        for group
        in trainer.optimizer.param_groups
        for parameter in group["params"]
    }

    if (
        optimizer_parameter_ids
        != model_parameter_ids
    ):
        raise ValueError(
            "optimizer parameters must exactly "
            "match the actor-critic model"
        )

    validated_progress = (
        progress
        if progress is not None
        else PPOTrainingProgress()
    )

    if not isinstance(
        validated_progress,
        PPOTrainingProgress,
    ):
        raise TypeError(
            "progress must be "
            "PPOTrainingProgress"
        )

    validated_name = _nonempty_string(
        policy_name,
        field_name="policy_name",
    )

    validated_version = _nonempty_string(
        policy_version,
        field_name="policy_version",
    )

    timestamp = (
        created_at
        if created_at is not None
        else datetime.now(
            timezone.utc
        ).isoformat()
    )

    validated_timestamp = (
        _nonempty_string(
            timestamp,
            field_name="created_at",
        )
    )

    try:
        datetime.fromisoformat(
            validated_timestamp.replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError as error:
        raise ValueError(
            "created_at must be "
            "an ISO-8601 timestamp"
        ) from error

    normalized_metadata = (
        _normalize_json_value(
            metadata or {},
            field_name="metadata",
        )
    )

    if not isinstance(
        normalized_metadata,
        dict,
    ):
        raise TypeError(
            "metadata must normalize "
            "to a mapping"
        )

    cpu_rng_state = (
        torch.get_rng_state()
        .detach()
        .cpu()
        .clone()
    )

    cuda_rng_states = (
        tuple(
            state.detach()
            .cpu()
            .clone()
            for state
            in torch.cuda.get_rng_state_all()
        )
        if torch.cuda.is_available()
        else ()
    )

    payload = {
        "format_version": (
            PPO_CHECKPOINT_FORMAT_VERSION
        ),
        "checkpoint_kind": (
            _CHECKPOINT_KIND
        ),
        "policy_name": validated_name,
        "policy_version": (
            validated_version
        ),
        "created_at": (
            validated_timestamp
        ),
        "torch_version": str(
            torch.__version__
        ),
        "action_names": (
            _expected_action_names()
        ),
        "model_config": _model_config(
            trainer.model
        ),
        "model_state_dict": {
            key: tensor.detach()
            .to(device="cpu")
            .contiguous()
            .clone()
            for key, tensor
            in trainer.model.state_dict().items()
        },
        "optimizer_type": (
            _SUPPORTED_OPTIMIZERS[
                optimizer_class
            ]
        ),
        "optimizer_state_dict": (
            _cpu_tree(
                trainer.optimizer.state_dict()
            )
        ),
        "trainer_config": (
            _trainer_config(trainer)
        ),
        "progress": (
            validated_progress.to_record()
        ),
        "metadata": normalized_metadata,
        "cpu_rng_state": cpu_rng_state,
        "cuda_rng_states": list(
            cuda_rng_states
        ),
    }

    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        checkpoint_path.with_name(
            "."
            + checkpoint_path.name
            + "."
            + uuid4().hex
            + ".tmp"
        )
    )

    try:
        with temporary_path.open(
            "wb"
        ) as file:
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
        if temporary_path.exists():
            temporary_path.unlink()

    return checkpoint_path


def load_ppo_training_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedPPOTrainingCheckpoint:
    """Strictly load and rebuild a PPO trainer."""

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    resolved_device = torch.device(
        device
    )

    if (
        resolved_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested "
            "but is unavailable"
        )

    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError(
            "checkpoint could not be "
            "loaded safely"
        ) from error

    raw = _validate_mapping_fields(
        payload,
        expected_fields=_PAYLOAD_FIELDS,
        field_name="checkpoint",
    )

    if raw["format_version"] != (
        PPO_CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            "checkpoint format version "
            "is unsupported"
        )

    if raw["checkpoint_kind"] != (
        _CHECKPOINT_KIND
    ):
        raise ValueError(
            "checkpoint kind is invalid"
        )

    policy_name = _nonempty_string(
        raw["policy_name"],
        field_name="policy_name",
    )

    policy_version = _nonempty_string(
        raw["policy_version"],
        field_name="policy_version",
    )

    created_at = _nonempty_string(
        raw["created_at"],
        field_name="created_at",
    )

    action_names = raw["action_names"]

    if action_names != (
        _expected_action_names()
    ):
        raise ValueError(
            "checkpoint action mapping "
            "does not match DiscreteAction"
        )

    config = _validate_model_config(
        raw["model_config"]
    )

    model_state = _validate_state_dict(
        raw["model_state_dict"],
        field_name="model_state_dict",
    )

    optimizer_type = (
        _nonempty_string(
            raw["optimizer_type"],
            field_name="optimizer_type",
        )
    )

    if optimizer_type not in (
        _OPTIMIZER_CLASSES
    ):
        raise ValueError(
            "checkpoint optimizer type "
            "is unsupported"
        )

    optimizer_state = (
        _validate_optimizer_state(
            raw["optimizer_state_dict"]
        )
    )

    (
        hyperparameters,
        max_gradient_norm,
        optimizer_step_count,
    ) = _validate_trainer_config(
        raw["trainer_config"]
    )

    progress = _validate_progress(
        raw["progress"]
    )

    normalized_metadata = (
        _normalize_json_value(
            raw["metadata"],
            field_name="metadata",
        )
    )

    if not isinstance(
        normalized_metadata,
        dict,
    ):
        raise ValueError(
            "checkpoint metadata is invalid"
        )

    cpu_rng_state = (
        _validate_rng_state(
            raw["cpu_rng_state"],
            field_name="cpu_rng_state",
        )
    )

    raw_cuda_states = raw[
        "cuda_rng_states"
    ]

    if not isinstance(
        raw_cuda_states,
        list,
    ):
        raise ValueError(
            "cuda_rng_states must be a list"
        )

    cuda_rng_states = tuple(
        _validate_rng_state(
            state,
            field_name=(
                f"cuda_rng_states[{index}]"
            ),
        )
        for index, state
        in enumerate(raw_cuda_states)
    )

    model = VisualActorCriticNetwork(
        **config
    )

    try:
        model.load_state_dict(
            model_state,
            strict=True,
        )
    except RuntimeError as error:
        raise ValueError(
            "checkpoint model state "
            "is incompatible"
        ) from error

    model.to(resolved_device)

    optimizer_class = (
        _OPTIMIZER_CLASSES[
            optimizer_type
        ]
    )

    optimizer = optimizer_class(
        model.parameters(),
        lr=1e-3,
    )

    try:
        optimizer.load_state_dict(
            optimizer_state
        )
    except (
        RuntimeError,
        ValueError,
        KeyError,
    ) as error:
        raise ValueError(
            "checkpoint optimizer state "
            "is incompatible"
        ) from error

    trainer = PPOTrainer(
        model,
        optimizer,
        device=resolved_device,
        hyperparameters=(
            hyperparameters
        ),
        max_gradient_norm=(
            max_gradient_norm
        ),
        optimizer_step_count=(
            optimizer_step_count
        ),
    )

    return LoadedPPOTrainingCheckpoint(
        path=checkpoint_path,
        trainer=trainer,
        policy_name=policy_name,
        policy_version=policy_version,
        created_at=created_at,
        progress=progress,
        metadata=MappingProxyType(
            normalized_metadata
        ),
        cpu_rng_state=(
            cpu_rng_state
        ),
        cuda_rng_states=(
            cuda_rng_states
        ),
    )


def restore_ppo_checkpoint_rng_state(
    checkpoint: LoadedPPOTrainingCheckpoint,
    *,
    include_cuda: bool = True,
) -> None:
    """Restore saved Torch sampling state explicitly."""

    if not isinstance(
        checkpoint,
        LoadedPPOTrainingCheckpoint,
    ):
        raise TypeError(
            "checkpoint must be a "
            "LoadedPPOTrainingCheckpoint"
        )

    if not isinstance(
        include_cuda,
        bool,
    ):
        raise TypeError(
            "include_cuda must be bool"
        )

    torch.set_rng_state(
        checkpoint.cpu_rng_state.clone()
    )

    if (
        include_cuda
        and checkpoint.cuda_rng_states
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG "
                "state but CUDA is unavailable"
            )

        current_count = (
            torch.cuda.device_count()
        )

        saved_count = len(
            checkpoint.cuda_rng_states
        )

        if current_count != saved_count:
            raise RuntimeError(
                "CUDA device count does not "
                "match checkpoint RNG state"
            )

        torch.cuda.set_rng_state_all(
            [
                state.clone()
                for state
                in checkpoint.cuda_rng_states
            ]
        )
