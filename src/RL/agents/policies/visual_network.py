"""Convolutional visual policy for stacked Xonotic RGB frames."""

from __future__ import annotations

import math

import torch
from torch import nn

from RL.actions.contracts import DiscreteAction


def _positive_integer(
    value: int,
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


class VisualPolicyNetwork(nn.Module):
    """Map stacked RGB policy frames to discrete-action logits.

    Input shape:
        (batch, frame_stack, rgb_channels, height, width)

    Default Xonotic shape:
        (batch, 4, 3, 90, 160)

    Output shape:
        (batch, 11)
    """

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
            example = torch.zeros(
                1,
                input_channels,
                self.frame_height,
                self.frame_width,
                dtype=torch.float32,
            )

            encoded = self.encoder(example)

        self.encoded_shape = tuple(
            int(dimension)
            for dimension in encoded.shape[1:]
        )

        encoded_size = math.prod(
            self.encoded_shape
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                encoded_size,
                self.hidden_dim,
            ),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(
            self.hidden_dim,
            self.action_count,
        )

    @property
    def frame_shape(
        self,
    ) -> tuple[int, int, int, int]:
        """Return the required unbatched frame shape."""

        return (
            self.frame_stack,
            self.rgb_channels,
            self.frame_height,
            self.frame_width,
        )

    def forward(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Return one action-logit vector per batch item."""

        if not isinstance(frames, torch.Tensor):
            raise TypeError(
                "frames must be a torch.Tensor"
            )

        if frames.ndim != 5:
            raise ValueError(
                "frames must have shape "
                "(batch, stack, channels, height, width)"
            )

        actual_shape = tuple(
            int(dimension)
            for dimension in frames.shape[1:]
        )

        if actual_shape != self.frame_shape:
            raise ValueError(
                "unexpected policy frame shape: "
                f"expected {self.frame_shape}, "
                f"received {actual_shape}"
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

        batch_size = int(frames.shape[0])

        combined_frames = frames.reshape(
            batch_size,
            self.frame_stack
            * self.rgb_channels,
            self.frame_height,
            self.frame_width,
        )

        encoded = self.encoder(
            combined_frames
        )

        features = self.projection(encoded)

        return self.action_head(features)
