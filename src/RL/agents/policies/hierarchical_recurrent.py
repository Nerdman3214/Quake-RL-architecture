"""Hierarchical recurrent policy for composite Xonotic controls."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import nn

from RL.actions.contracts import DiscreteAction


AXIS_CLASS_COUNT = 3
WEAPON_CLASS_COUNT = 3
GAME_MODE_COUNT = 5
PREVIOUS_ACTION_FEATURE_COUNT = 8
TELEMETRY_FEATURE_COUNT = 8
DEFAULT_TELEMETRY_WEAPON_COUNT = 64
LEGACY_ACTION_COUNT = len(tuple(DiscreteAction))


class TacticalIntent(IntEnum):
    """High-level tactical intentions.

    UNKNOWN remains the only authoritative target until tactical
    annotations or reliable event-derived labels are available.
    """

    UNKNOWN = 0
    ENGAGE = 1
    RETREAT = 2
    SEEK_RESOURCE = 3
    FOLLOW_ROUTE = 4
    DEFEND = 5
    ATTACK_OBJECTIVE = 6
    ESCORT = 7
    RETURN_FLAG = 8
    HOLD_POINT = 9
    REGROUP = 10


@dataclass(frozen=True)
class HierarchicalPolicyOutput:
    """Multitask predictions for one recurrent sequence."""

    intent_logits: torch.Tensor
    forward_logits: torch.Tensor
    strafe_logits: torch.Tensor
    mouse_deltas: torch.Tensor
    fire_logits: torch.Tensor
    jump_logits: torch.Tensor
    weapon_logits: torch.Tensor
    legacy_action_logits: torch.Tensor
    hidden_state: torch.Tensor


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



def _nonnegative_integer(
    value: int,
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


class HierarchicalRecurrentPolicy(nn.Module):
    """CNN-GRU policy with simultaneous composite-control heads.

    Input shapes:
        frames:
            (batch, sequence, stack, channels, height, width)

        previous_action_features:
            (batch, sequence, 8)

        mode_indices:
            (batch,)

    The tactical-intent head is present for the hierarchical
    architecture, but supervised intent loss must remain disabled
    until authoritative intent labels exist.
    """

    def __init__(
        self,
        *,
        frame_stack: int = 4,
        rgb_channels: int = 3,
        frame_height: int = 90,
        frame_width: int = 160,
        visual_feature_dim: int = 256,
        previous_action_dim: int = 32,
        mode_embedding_dim: int = 16,
        telemetry_feature_count: int = 0,
        telemetry_feature_dim: int = 0,
        telemetry_weapon_count: int = 0,
        telemetry_weapon_embedding_dim: int = 0,
        recurrent_input_dim: int = 256,
        recurrent_hidden_dim: int = 256,
        recurrent_layers: int = 2,
        game_mode_count: int = GAME_MODE_COUNT,
        tactical_intent_count: int = len(
            tuple(TacticalIntent)
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
        self.visual_feature_dim = _positive_integer(
            visual_feature_dim,
            field_name="visual_feature_dim",
        )
        self.previous_action_dim = _positive_integer(
            previous_action_dim,
            field_name="previous_action_dim",
        )
        self.mode_embedding_dim = _positive_integer(
            mode_embedding_dim,
            field_name="mode_embedding_dim",
        )
        self.telemetry_feature_count = (
            _nonnegative_integer(
                telemetry_feature_count,
                field_name=(
                    "telemetry_feature_count"
                ),
            )
        )
        self.telemetry_feature_dim = (
            _nonnegative_integer(
                telemetry_feature_dim,
                field_name=(
                    "telemetry_feature_dim"
                ),
            )
        )
        self.telemetry_weapon_count = (
            _nonnegative_integer(
                telemetry_weapon_count,
                field_name=(
                    "telemetry_weapon_count"
                ),
            )
        )
        self.telemetry_weapon_embedding_dim = (
            _nonnegative_integer(
                telemetry_weapon_embedding_dim,
                field_name=(
                    "telemetry_weapon_embedding_dim"
                ),
            )
        )

        telemetry_dimensions = (
            self.telemetry_feature_count,
            self.telemetry_feature_dim,
            self.telemetry_weapon_count,
            self.telemetry_weapon_embedding_dim,
        )

        if any(telemetry_dimensions) and not all(
            telemetry_dimensions
        ):
            raise ValueError(
                "telemetry model dimensions must "
                "either all be zero or all be positive"
            )

        self.telemetry_enabled = all(
            value > 0
            for value in telemetry_dimensions
        )
        self.recurrent_input_dim = _positive_integer(
            recurrent_input_dim,
            field_name="recurrent_input_dim",
        )
        self.recurrent_hidden_dim = _positive_integer(
            recurrent_hidden_dim,
            field_name="recurrent_hidden_dim",
        )
        self.recurrent_layers = _positive_integer(
            recurrent_layers,
            field_name="recurrent_layers",
        )
        self.game_mode_count = _positive_integer(
            game_mode_count,
            field_name="game_mode_count",
        )
        self.tactical_intent_count = _positive_integer(
            tactical_intent_count,
            field_name="tactical_intent_count",
        )

        input_channels = (
            self.frame_stack
            * self.rgb_channels
        )

        self.visual_encoder = nn.Sequential(
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
            encoded = self.visual_encoder(
                example
            )

        encoded_size = math.prod(
            encoded.shape[1:]
        )

        self.visual_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                encoded_size,
                self.visual_feature_dim,
            ),
            nn.ReLU(),
        )

        self.previous_action_encoder = nn.Sequential(
            nn.LayerNorm(
                PREVIOUS_ACTION_FEATURE_COUNT
            ),
            nn.Linear(
                PREVIOUS_ACTION_FEATURE_COUNT,
                self.previous_action_dim,
            ),
            nn.ReLU(),
        )

        self.mode_embedding = nn.Embedding(
            self.game_mode_count,
            self.mode_embedding_dim,
        )

        if self.telemetry_enabled:
            self.telemetry_encoder = nn.Sequential(
                nn.LayerNorm(
                    self.telemetry_feature_count
                ),
                nn.Linear(
                    self.telemetry_feature_count,
                    self.telemetry_feature_dim,
                ),
                nn.ReLU(),
            )
            self.telemetry_weapon_embedding = (
                nn.Embedding(
                    self.telemetry_weapon_count,
                    (
                        self
                        .telemetry_weapon_embedding_dim
                    ),
                )
            )
        else:
            self.telemetry_encoder = None
            self.telemetry_weapon_embedding = None

        fused_feature_dim = (
            self.visual_feature_dim
            + self.previous_action_dim
            + self.mode_embedding_dim
            + (
                self.telemetry_feature_dim
                if self.telemetry_enabled
                else 0
            )
            + (
                self.telemetry_weapon_embedding_dim
                if self.telemetry_enabled
                else 0
            )
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(
                fused_feature_dim,
                self.recurrent_input_dim,
            ),
            nn.ReLU(),
        )

        self.recurrent = nn.GRU(
            input_size=self.recurrent_input_dim,
            hidden_size=self.recurrent_hidden_dim,
            num_layers=self.recurrent_layers,
            batch_first=True,
            dropout=(
                0.1
                if self.recurrent_layers > 1
                else 0.0
            ),
        )

        self.intent_head = nn.Linear(
            self.recurrent_hidden_dim,
            self.tactical_intent_count,
        )
        self.forward_head = nn.Linear(
            self.recurrent_hidden_dim,
            AXIS_CLASS_COUNT,
        )
        self.strafe_head = nn.Linear(
            self.recurrent_hidden_dim,
            AXIS_CLASS_COUNT,
        )
        self.mouse_head = nn.Linear(
            self.recurrent_hidden_dim,
            2,
        )
        self.fire_head = nn.Linear(
            self.recurrent_hidden_dim,
            1,
        )
        self.jump_head = nn.Linear(
            self.recurrent_hidden_dim,
            1,
        )
        self.weapon_head = nn.Linear(
            self.recurrent_hidden_dim,
            WEAPON_CLASS_COUNT,
        )
        self.legacy_action_head = nn.Linear(
            self.recurrent_hidden_dim,
            LEGACY_ACTION_COUNT,
        )

    @property
    def frame_shape(
        self,
    ) -> tuple[int, int, int, int]:
        return (
            self.frame_stack,
            self.rgb_channels,
            self.frame_height,
            self.frame_width,
        )

    def forward(
        self,
        frames: torch.Tensor,
        previous_action_features: torch.Tensor,
        mode_indices: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        *,
        telemetry_features: (
            torch.Tensor | None
        ) = None,
        telemetry_weapon_indices: (
            torch.Tensor | None
        ) = None,
    ) -> HierarchicalPolicyOutput:
        """Return recurrent multitask control predictions."""

        if not isinstance(frames, torch.Tensor):
            raise TypeError(
                "frames must be a torch.Tensor"
            )

        if frames.ndim != 6:
            raise ValueError(
                "frames must have shape "
                "(batch, sequence, stack, channels, height, width)"
            )

        batch_size = int(frames.shape[0])
        sequence_length = int(
            frames.shape[1]
        )

        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError(
                "batch and sequence dimensions must be positive"
            )

        actual_frame_shape = tuple(
            int(value)
            for value in frames.shape[2:]
        )

        if actual_frame_shape != self.frame_shape:
            raise ValueError(
                "unexpected policy frame shape: "
                f"expected {self.frame_shape}, "
                f"received {actual_frame_shape}"
            )

        if frames.dtype != torch.float32:
            raise TypeError(
                "frames must use torch.float32"
            )

        if not bool(
            torch.isfinite(frames).all().item()
        ):
            raise ValueError(
                "frames must contain only finite values"
            )

        expected_previous_shape = (
            batch_size,
            sequence_length,
            PREVIOUS_ACTION_FEATURE_COUNT,
        )

        if (
            not isinstance(
                previous_action_features,
                torch.Tensor,
            )
            or tuple(
                previous_action_features.shape
            )
            != expected_previous_shape
        ):
            raise ValueError(
                "previous_action_features have "
                "an unexpected shape"
            )

        if previous_action_features.dtype != (
            torch.float32
        ):
            raise TypeError(
                "previous_action_features must "
                "use torch.float32"
            )

        if not bool(
            torch.isfinite(
                previous_action_features
            ).all().item()
        ):
            raise ValueError(
                "previous_action_features must "
                "contain only finite values"
            )

        if self.telemetry_enabled:
            expected_telemetry_shape = (
                batch_size,
                sequence_length,
                self.telemetry_feature_count,
            )

            if (
                not isinstance(
                    telemetry_features,
                    torch.Tensor,
                )
                or tuple(
                    telemetry_features.shape
                )
                != expected_telemetry_shape
            ):
                raise ValueError(
                    "telemetry_features have "
                    "an unexpected shape"
                )

            if telemetry_features.dtype != (
                torch.float32
            ):
                raise TypeError(
                    "telemetry_features must "
                    "use torch.float32"
                )

            if not bool(
                torch.isfinite(
                    telemetry_features
                ).all().item()
            ):
                raise ValueError(
                    "telemetry_features must "
                    "contain only finite values"
                )

            expected_weapon_shape = (
                batch_size,
                sequence_length,
            )

            if (
                not isinstance(
                    telemetry_weapon_indices,
                    torch.Tensor,
                )
                or tuple(
                    telemetry_weapon_indices.shape
                )
                != expected_weapon_shape
                or telemetry_weapon_indices.dtype
                != torch.long
            ):
                raise TypeError(
                    "telemetry_weapon_indices must "
                    "be a torch.long tensor matching "
                    "the batch and sequence"
                )

            minimum_weapon = int(
                telemetry_weapon_indices.min().item()
            )
            maximum_weapon = int(
                telemetry_weapon_indices.max().item()
            )

            if (
                minimum_weapon < 0
                or maximum_weapon
                >= self.telemetry_weapon_count
            ):
                raise ValueError(
                    "telemetry_weapon_indices contain "
                    "an unknown weapon"
                )

        if (
            not isinstance(
                mode_indices,
                torch.Tensor,
            )
            or tuple(mode_indices.shape)
            != (batch_size,)
            or mode_indices.dtype != torch.long
        ):
            raise TypeError(
                "mode_indices must be a torch.long "
                "vector matching the batch size"
            )

        minimum_mode = int(
            mode_indices.min().item()
        )
        maximum_mode = int(
            mode_indices.max().item()
        )

        if (
            minimum_mode < 0
            or maximum_mode
            >= self.game_mode_count
        ):
            raise ValueError(
                "mode_indices contain an unknown mode"
            )

        if hidden_state is not None:
            expected_hidden_shape = (
                self.recurrent_layers,
                batch_size,
                self.recurrent_hidden_dim,
            )

            if (
                not isinstance(
                    hidden_state,
                    torch.Tensor,
                )
                or tuple(hidden_state.shape)
                != expected_hidden_shape
            ):
                raise ValueError(
                    "hidden_state has an unexpected shape"
                )

            if hidden_state.dtype != torch.float32:
                raise TypeError(
                    "hidden_state must use torch.float32"
                )

            if not bool(
                torch.isfinite(
                    hidden_state
                ).all().item()
            ):
                raise ValueError(
                    "hidden_state must contain "
                    "only finite values"
                )

        combined_frames = frames.reshape(
            batch_size * sequence_length,
            self.frame_stack
            * self.rgb_channels,
            self.frame_height,
            self.frame_width,
        )

        visual_features = self.visual_projection(
            self.visual_encoder(
                combined_frames
            )
        ).reshape(
            batch_size,
            sequence_length,
            self.visual_feature_dim,
        )

        previous_features = (
            self.previous_action_encoder(
                previous_action_features
            )
        )

        mode_features = self.mode_embedding(
            mode_indices
        ).unsqueeze(1).expand(
            -1,
            sequence_length,
            -1,
        )

        fused_parts = [
            visual_features,
            previous_features,
            mode_features,
        ]

        if self.telemetry_enabled:
            if (
                self.telemetry_encoder is None
                or self.telemetry_weapon_embedding
                is None
                or telemetry_features is None
                or telemetry_weapon_indices is None
            ):
                raise RuntimeError(
                    "telemetry model modules or "
                    "inputs are unavailable"
                )

            fused_parts.append(
                self.telemetry_encoder(
                    telemetry_features
                )
            )
            fused_parts.append(
                self.telemetry_weapon_embedding(
                    telemetry_weapon_indices
                )
            )

        fused_features = self.feature_fusion(
            torch.cat(
                tuple(fused_parts),
                dim=2,
            )
        )

        recurrent_features, next_hidden = (
            self.recurrent(
                fused_features,
                hidden_state,
            )
        )

        return HierarchicalPolicyOutput(
            intent_logits=self.intent_head(
                recurrent_features
            ),
            forward_logits=self.forward_head(
                recurrent_features
            ),
            strafe_logits=self.strafe_head(
                recurrent_features
            ),
            mouse_deltas=self.mouse_head(
                recurrent_features
            ),
            fire_logits=self.fire_head(
                recurrent_features
            ).squeeze(-1),
            jump_logits=self.jump_head(
                recurrent_features
            ).squeeze(-1),
            weapon_logits=self.weapon_head(
                recurrent_features
            ),
            legacy_action_logits=(
                self.legacy_action_head(
                    recurrent_features
                )
            ),
            hidden_state=next_hidden,
        )
