"""Inference agent backed by the visual actor-critic network."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import torch

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.agents.baselines.contracts import Agent
from RL.agents.policies.actor_critic import (
    VisualActorCriticNetwork,
)
from RL.observations.contracts import Observation


class ActorCriticPolicyAgent(Agent):
    """Select discrete actions and expose PPO collection estimates."""

    def __init__(
        self,
        model: VisualActorCriticNetwork,
        *,
        device: str | torch.device = "cpu",
        deterministic: bool = False,
        duration_ticks: int = 1,
        policy_name: str = "xonotic-visual-actor-critic",
        policy_version: str = "v1",
    ) -> None:
        if not isinstance(
            model,
            VisualActorCriticNetwork,
        ):
            raise TypeError(
                "model must be a VisualActorCriticNetwork"
            )

        if not isinstance(
            deterministic,
            bool,
        ):
            raise TypeError(
                "deterministic must be bool"
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

        if (
            not isinstance(policy_name, str)
            or not policy_name
        ):
            raise ValueError(
                "policy_name must not be empty"
            )

        if (
            not isinstance(policy_version, str)
            or not policy_version
        ):
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
        self._deterministic = deterministic
        self._duration_ticks = duration_ticks
        self._policy_name = policy_name
        self._policy_version = policy_version

        self._last_action: (
            DiscreteAction | None
        ) = None

        self._last_log_prob: (
            float | None
        ) = None

        self._last_value: (
            float | None
        ) = None

        self._last_entropy: (
            float | None
        ) = None

        self._last_logits: (
            tuple[float, ...] | None
        ) = None

        self._last_action_scores: (
            Mapping[str, float]
        ) = MappingProxyType({})

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @property
    def duration_ticks(self) -> int:
        return self._duration_ticks

    @property
    def policy_name(self) -> str:
        return self._policy_name

    @property
    def policy_version(self) -> str:
        return self._policy_version

    @property
    def last_action(
        self,
    ) -> DiscreteAction | None:
        return self._last_action

    @property
    def last_log_prob(
        self,
    ) -> float | None:
        return self._last_log_prob

    @property
    def last_value(
        self,
    ) -> float | None:
        return self._last_value

    @property
    def last_entropy(
        self,
    ) -> float | None:
        return self._last_entropy

    @property
    def last_logits(
        self,
    ) -> tuple[float, ...] | None:
        return self._last_logits

    @property
    def last_action_scores(
        self,
    ) -> Mapping[str, float]:
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

            if actual_shape != (
                self.model.frame_shape
            ):
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

            if not bool(
                np.isfinite(frame).all()
            ):
                raise ValueError(
                    "observation frame must contain "
                    "only finite values"
                )

            copied = np.array(
                frame,
                dtype=np.float32,
                order="C",
                copy=True,
            )

            tensor = torch.from_numpy(
                copied
            )
        elif isinstance(
            frame,
            torch.Tensor,
        ):
            actual_shape = tuple(
                int(dimension)
                for dimension in frame.shape
            )

            if actual_shape != (
                self.model.frame_shape
            ):
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
                torch.isfinite(frame)
                .all()
                .item()
            ):
                raise ValueError(
                    "observation frame must contain "
                    "only finite values"
                )

            tensor = (
                frame.detach()
                .to(dtype=torch.float32)
                .contiguous()
            )
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
        """Select one action and retain its PPO estimates."""

        frames = self._frame_tensor(
            observation
        )

        self.model.eval()

        with torch.inference_mode():
            decision = self.model.act(
                frames,
                deterministic=(
                    self._deterministic
                ),
            )

        action_index = int(
            decision.action_indices.item()
        )

        action = DiscreteAction(
            action_index
        )

        log_prob = float(
            decision.log_probs.item()
        )

        value = float(
            decision.values.item()
        )

        entropy = float(
            decision.entropies.item()
        )

        logits = tuple(
            float(item)
            for item in (
                decision.logits[0]
                .detach()
                .cpu()
                .tolist()
            )
        )

        for field_name, number in (
            ("log_prob", log_prob),
            ("value", value),
            ("entropy", entropy),
        ):
            if not math.isfinite(number):
                raise RuntimeError(
                    f"actor-critic {field_name} "
                    "is not finite"
                )

        self._last_action = action
        self._last_log_prob = log_prob
        self._last_value = value
        self._last_entropy = entropy
        self._last_logits = logits

        self._last_action_scores = (
            MappingProxyType(
                {
                    candidate.name: logits[
                        int(candidate)
                    ]
                    for candidate
                    in DiscreteAction
                }
            )
        )

        return ActionCommand(
            action=action,
            duration_ticks=(
                self._duration_ticks
            ),
        )

    def estimate_value(
        self,
        observation: Observation,
    ) -> float:
        """Estimate one observation without selecting an action."""

        frames = self._frame_tensor(
            observation
        )

        self.model.eval()

        with torch.inference_mode():
            _, values = self.model(
                frames
            )

        value = float(
            values.item()
        )

        if not math.isfinite(value):
            raise RuntimeError(
                "actor-critic value is not finite"
            )

        return value
