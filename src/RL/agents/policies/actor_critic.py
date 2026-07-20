"""Visual actor-critic network for discrete Xonotic control."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from RL.actions.contracts import DiscreteAction


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
            f"{field_name} must be a positive integer"
        )

    return value


@dataclass(frozen=True)
class ActorCriticActionBatch:
    """Actions and estimates produced from one frame batch."""

    action_indices: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor
    values: torch.Tensor
    logits: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(
            self.logits,
            torch.Tensor,
        ):
            raise TypeError(
                "logits must be a torch.Tensor"
            )

        if self.logits.ndim != 2:
            raise ValueError(
                "logits must have two dimensions"
            )

        batch_size = int(
            self.logits.shape[0]
        )

        action_count = len(
            tuple(DiscreteAction)
        )

        if tuple(self.logits.shape) != (
            batch_size,
            action_count,
        ):
            raise ValueError(
                "logits have an unexpected shape"
            )

        vector_fields = (
            (
                "action_indices",
                self.action_indices,
            ),
            ("log_probs", self.log_probs),
            ("entropies", self.entropies),
            ("values", self.values),
        )

        for field_name, tensor in vector_fields:
            if not isinstance(
                tensor,
                torch.Tensor,
            ):
                raise TypeError(
                    f"{field_name} must be a torch.Tensor"
                )

            if tuple(tensor.shape) != (
                batch_size,
            ):
                raise ValueError(
                    f"{field_name} has an unexpected shape"
                )

        if (
            self.action_indices.dtype
            != torch.long
        ):
            raise TypeError(
                "action_indices must use torch.long"
            )

        for tensor in (
            self.logits,
            self.log_probs,
            self.entropies,
            self.values,
        ):
            if not tensor.is_floating_point():
                raise TypeError(
                    "actor-critic outputs must be floating point"
                )

            if not bool(
                torch.isfinite(tensor).all().item()
            ):
                raise ValueError(
                    "actor-critic outputs must be finite"
                )


class VisualActorCriticNetwork(nn.Module):
    """Shared visual encoder with policy and value heads."""

    def __init__(
        self,
        *,
        frame_stack: int = 4,
        rgb_channels: int = 3,
        frame_height: int = 90,
        frame_width: int = 160,
        hidden_dim: int = 256,
        action_count: int = len(
            tuple(DiscreteAction)
        ),
    ) -> None:
        super().__init__()

        self.frame_stack = _positive_integer(
            frame_stack,
            field_name="frame_stack",
        )

        self.rgb_channels = _positive_integer(
            rgb_channels,
            field_name="rgb_channels",
        )

        self.frame_height = _positive_integer(
            frame_height,
            field_name="frame_height",
        )

        self.frame_width = _positive_integer(
            frame_width,
            field_name="frame_width",
        )

        self.hidden_dim = _positive_integer(
            hidden_dim,
            field_name="hidden_dim",
        )

        self.action_count = _positive_integer(
            action_count,
            field_name="action_count",
        )

        expected_action_count = len(
            tuple(DiscreteAction)
        )

        if self.action_count != (
            expected_action_count
        ):
            raise ValueError(
                "action_count must match DiscreteAction"
            )

        input_channels = (
            self.frame_stack
            * self.rgb_channels
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels,
                32,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
        )

        with torch.no_grad():
            encoded = self.encoder(
                torch.zeros(
                    1,
                    input_channels,
                    self.frame_height,
                    self.frame_width,
                    dtype=torch.float32,
                )
            )

        self.encoded_shape = tuple(
            int(dimension)
            for dimension in encoded.shape[1:]
        )

        encoded_size = int(
            encoded.numel()
        )

        self.feature_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                encoded_size,
                self.hidden_dim,
            ),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(
            self.hidden_dim,
            self.action_count,
        )

        self.value_head = nn.Linear(
            self.hidden_dim,
            1,
        )

    @property
    def frame_shape(
        self,
    ) -> tuple[int, int, int, int]:
        """Return one unbatched policy-frame shape."""

        return (
            self.frame_stack,
            self.rgb_channels,
            self.frame_height,
            self.frame_width,
        )

    def _validate_frames(
        self,
        frames: torch.Tensor,
    ) -> int:
        if not isinstance(
            frames,
            torch.Tensor,
        ):
            raise TypeError(
                "frames must be a torch.Tensor"
            )

        if frames.ndim != 5:
            raise ValueError(
                "frames must have shape "
                "(batch, stack, channels, height, width)"
            )

        expected_tail = self.frame_shape
        actual_tail = tuple(
            int(dimension)
            for dimension in frames.shape[1:]
        )

        if actual_tail != expected_tail:
            raise ValueError(
                "frames have an unexpected shape: "
                f"expected (*, {expected_tail}), "
                f"received {tuple(frames.shape)}"
            )

        batch_size = int(
            frames.shape[0]
        )

        if batch_size <= 0:
            raise ValueError(
                "frame batch must not be empty"
            )

        if not frames.is_floating_point():
            raise TypeError(
                "frames must use a floating-point dtype"
            )

        if not bool(
            torch.isfinite(frames).all().item()
        ):
            raise ValueError(
                "frames must contain only finite values"
            )

        return batch_size

    def encode(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Return shared visual features."""

        batch_size = self._validate_frames(
            frames
        )

        merged = frames.reshape(
            batch_size,
            self.frame_stack
            * self.rgb_channels,
            self.frame_height,
            self.frame_width,
        )

        encoded = self.encoder(merged)

        features = self.feature_layer(
            encoded
        )

        expected_shape = (
            batch_size,
            self.hidden_dim,
        )

        if tuple(features.shape) != (
            expected_shape
        ):
            raise RuntimeError(
                "visual encoder returned an "
                "unexpected feature shape"
            )

        return features

    def forward(
        self,
        frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and scalar state values."""

        features = self.encode(frames)

        logits = self.policy_head(
            features
        )

        values = self.value_head(
            features
        ).squeeze(-1)

        expected_logits_shape = (
            int(frames.shape[0]),
            self.action_count,
        )

        expected_value_shape = (
            int(frames.shape[0]),
        )

        if tuple(logits.shape) != (
            expected_logits_shape
        ):
            raise RuntimeError(
                "policy head returned an "
                "unexpected shape"
            )

        if tuple(values.shape) != (
            expected_value_shape
        ):
            raise RuntimeError(
                "value head returned an "
                "unexpected shape"
            )

        if not bool(
            torch.isfinite(logits).all().item()
        ):
            raise RuntimeError(
                "policy logits are not finite"
            )

        if not bool(
            torch.isfinite(values).all().item()
        ):
            raise RuntimeError(
                "state values are not finite"
            )

        return logits, values

    def act(
        self,
        frames: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> ActorCriticActionBatch:
        """Select actions and retain PPO collection estimates."""

        if not isinstance(
            deterministic,
            bool,
        ):
            raise TypeError(
                "deterministic must be bool"
            )

        logits, values = self(frames)

        distribution = Categorical(
            logits=logits
        )

        if deterministic:
            action_indices = torch.argmax(
                logits,
                dim=1,
            )
        else:
            action_indices = (
                distribution.sample()
            )

        log_probs = distribution.log_prob(
            action_indices
        )

        entropies = distribution.entropy()

        return ActorCriticActionBatch(
            action_indices=action_indices,
            log_probs=log_probs,
            entropies=entropies,
            values=values,
            logits=logits,
        )
