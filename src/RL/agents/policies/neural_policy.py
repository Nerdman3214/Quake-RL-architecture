"""Inference-only agent backed by a visual policy network."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.agents.baselines.contracts import Agent
from RL.agents.policies.visual_network import (
    VisualPolicyNetwork,
)
from RL.observations.contracts import Observation


class NeuralPolicyAgent(Agent):
    """Select discrete actions using inference-only CNN logits."""

    def __init__(
        self,
        model: VisualPolicyNetwork,
        *,
        device: str | torch.device = "cpu",
        duration_ticks: int = 1,
        policy_name: str = "xonotic-visual-cnn",
        policy_version: str = "v1",
    ) -> None:
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

        if (
            isinstance(duration_ticks, bool)
            or not isinstance(
                duration_ticks,
                int,
            )
            or duration_ticks <= 0
        ):
            raise ValueError(
                "duration_ticks must be a positive integer"
            )

        if not policy_name:
            raise ValueError(
                "policy_name must not be empty"
            )

        if not policy_version:
            raise ValueError(
                "policy_version must not be empty"
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

        self.model = model.to(
            resolved_device
        )

        self.model.eval()

        self._device = resolved_device
        self._duration_ticks = duration_ticks
        self._policy_name = policy_name
        self._policy_version = policy_version

        self._last_logits: (
            tuple[float, ...] | None
        ) = None

        self._last_action_scores: (
            Mapping[str, float]
        ) = MappingProxyType({})

    @property
    def device(self) -> torch.device:
        """Return the device used for inference."""

        return self._device

    @property
    def duration_ticks(self) -> int:
        """Return the duration assigned to selected actions."""

        return self._duration_ticks

    @property
    def policy_name(self) -> str:
        """Return the inspectable policy name."""

        return self._policy_name

    @property
    def policy_version(self) -> str:
        """Return the inspectable policy version."""

        return self._policy_version

    @property
    def last_logits(
        self,
    ) -> tuple[float, ...] | None:
        """Return logits from the most recent decision."""

        return self._last_logits

    @property
    def last_action_scores(
        self,
    ) -> Mapping[str, float]:
        """Return action-name logits from the latest decision."""

        return self._last_action_scores

    def _frame_tensor(
        self,
        observation: Observation,
    ) -> torch.Tensor:
        if not isinstance(
            observation,
            Observation,
        ):
            raise TypeError(
                "observation must be an Observation"
            )

        frame = observation.frame

        if frame is None:
            raise ValueError(
                "observation frame must not be None"
            )

        if isinstance(frame, np.ndarray):
            actual_shape = tuple(
                int(dimension)
                for dimension in frame.shape
            )

            if actual_shape != self.model.frame_shape:
                raise ValueError(
                    "unexpected observation frame shape: "
                    f"expected {self.model.frame_shape}, "
                    f"received {actual_shape}"
                )

            if not np.issubdtype(
                frame.dtype,
                np.floating,
            ):
                raise TypeError(
                    "observation frame must use "
                    "a floating-point dtype"
                )

            if not bool(np.isfinite(frame).all()):
                raise ValueError(
                    "observation frame must contain "
                    "only finite values"
                )

            contiguous = np.ascontiguousarray(
                frame
            )

            tensor = torch.from_numpy(
                contiguous
            )

        elif isinstance(frame, torch.Tensor):
            actual_shape = tuple(
                int(dimension)
                for dimension in frame.shape
            )

            if actual_shape != self.model.frame_shape:
                raise ValueError(
                    "unexpected observation frame shape: "
                    f"expected {self.model.frame_shape}, "
                    f"received {actual_shape}"
                )

            if not frame.is_floating_point():
                raise TypeError(
                    "observation frame must use "
                    "a floating-point dtype"
                )

            if not bool(
                torch.isfinite(frame).all().item()
            ):
                raise ValueError(
                    "observation frame must contain "
                    "only finite values"
                )

            tensor = frame.detach()

        else:
            raise TypeError(
                "observation frame must be a NumPy "
                "array or torch.Tensor"
            )

        return tensor.to(
            device=self._device,
            dtype=torch.float32,
        ).unsqueeze(0)

    def act(
        self,
        observation: Observation,
    ) -> ActionCommand:
        """Select the highest-logit action without training."""

        frames = self._frame_tensor(
            observation
        )

        self.model.eval()

        with torch.inference_mode():
            logits = self.model(frames)

        expected_shape = (
            1,
            len(tuple(DiscreteAction)),
        )

        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "model returned an unexpected "
                f"logit shape: {tuple(logits.shape)}"
            )

        if not bool(
            torch.isfinite(logits).all().item()
        ):
            raise RuntimeError(
                "model returned non-finite logits"
            )

        cpu_logits = logits[0].detach().cpu()

        logit_values = tuple(
            float(value)
            for value in cpu_logits.tolist()
        )

        self._last_logits = logit_values

        self._last_action_scores = (
            MappingProxyType(
                {
                    action.name: logit_values[
                        int(action)
                    ]
                    for action in DiscreteAction
                }
            )
        )

        action_index = int(
            torch.argmax(
                logits,
                dim=1,
            ).item()
        )

        selected_action = DiscreteAction(
            action_index
        )

        return ActionCommand(
            action=selected_action,
            duration_ticks=(
                self._duration_ticks
            ),
        )
