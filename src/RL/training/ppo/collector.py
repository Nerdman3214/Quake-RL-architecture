"""Bounded live rollout collection for the PPO actor-critic."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from RL.actions.contracts import ActionCommand
from RL.agents.policies.actor_critic_agent import (
    ActorCriticPolicyAgent,
)
from RL.env.core.contracts import (
    Environment,
    StepResult,
)
from RL.observations.contracts import Observation
from RL.training.ppo.core import (
    PPORolloutBatch,
    RolloutBuffer,
    RolloutTransition,
)


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


def _validate_observation(
    observation: object,
    *,
    source: str,
) -> Observation:
    if not isinstance(
        observation,
        Observation,
    ):
        raise TypeError(
            f"{source} must return an Observation"
        )

    return observation


def _frame_copy(
    observation: Observation,
) -> torch.Tensor:
    frame = observation.frame

    if frame is None:
        raise ValueError(
            "rollout observation frame must not be None"
        )

    if isinstance(frame, np.ndarray):
        if tuple(frame.shape) != (
            4,
            3,
            90,
            160,
        ):
            raise ValueError(
                "rollout frame has an unexpected shape"
            )

        if not np.issubdtype(
            frame.dtype,
            np.floating,
        ):
            raise TypeError(
                "rollout frame must be floating point"
            )

        if not bool(
            np.isfinite(frame).all()
        ):
            raise ValueError(
                "rollout frame must be finite"
            )

        array = np.array(
            frame,
            dtype=np.float32,
            order="C",
            copy=True,
        )

        return torch.from_numpy(array)

    if isinstance(frame, torch.Tensor):
        if tuple(frame.shape) != (
            4,
            3,
            90,
            160,
        ):
            raise ValueError(
                "rollout frame has an unexpected shape"
            )

        if not frame.is_floating_point():
            raise TypeError(
                "rollout frame must be floating point"
            )

        if not bool(
            torch.isfinite(frame)
            .all()
            .item()
        ):
            raise ValueError(
                "rollout frame must be finite"
            )

        return (
            frame.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .contiguous()
            .clone()
        )

    raise TypeError(
        "rollout frame must be a NumPy array "
        "or torch.Tensor"
    )


@dataclass(frozen=True)
class CollectedRolloutTransition:
    """Auditable environment transition used by one PPO rollout."""

    step_index: int
    observation: Observation
    action: ActionCommand
    next_observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    old_log_prob: float
    old_value: float
    entropy: float
    info: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(
                self.step_index,
                int,
            )
            or self.step_index < 0
        ):
            raise ValueError(
                "step_index must be nonnegative"
            )

        if not isinstance(
            self.observation,
            Observation,
        ):
            raise TypeError(
                "observation must be an Observation"
            )

        if not isinstance(
            self.next_observation,
            Observation,
        ):
            raise TypeError(
                "next_observation must be an Observation"
            )

        if not isinstance(
            self.action,
            ActionCommand,
        ):
            raise TypeError(
                "action must be an ActionCommand"
            )

        _finite_number(
            self.reward,
            field_name="reward",
        )

        _finite_number(
            self.old_log_prob,
            field_name="old_log_prob",
        )

        _finite_number(
            self.old_value,
            field_name="old_value",
        )

        _finite_number(
            self.entropy,
            field_name="entropy",
        )

        if not isinstance(
            self.terminated,
            bool,
        ):
            raise TypeError(
                "terminated must be bool"
            )

        if not isinstance(
            self.truncated,
            bool,
        ):
            raise TypeError(
                "truncated must be bool"
            )

        if (
            self.terminated
            and self.truncated
        ):
            raise ValueError(
                "transition cannot be both "
                "terminated and truncated"
            )

        if not isinstance(
            self.info,
            Mapping,
        ):
            raise TypeError(
                "info must be a mapping"
            )

        object.__setattr__(
            self,
            "info",
            MappingProxyType(
                dict(self.info)
            ),
        )


@dataclass(frozen=True)
class CollectedRolloutResult:
    """One bounded live rollout and its finalized PPO tensor batch."""

    initial_observation: Observation
    reset_info: Mapping[str, Any]
    transitions: tuple[
        CollectedRolloutTransition,
        ...
    ]
    batch: PPORolloutBatch
    total_reward: float
    terminated: bool
    truncated: bool
    termination_reason: str
    bootstrap_value: float

    def __post_init__(self) -> None:
        if not isinstance(
            self.initial_observation,
            Observation,
        ):
            raise TypeError(
                "initial_observation must be "
                "an Observation"
            )

        if not isinstance(
            self.batch,
            PPORolloutBatch,
        ):
            raise TypeError(
                "batch must be a PPORolloutBatch"
            )

        if not self.transitions:
            raise ValueError(
                "rollout must contain transitions"
            )

        if (
            len(self.transitions)
            != int(self.batch.frames.shape[0])
        ):
            raise ValueError(
                "transition count must match batch size"
            )

        _finite_number(
            self.total_reward,
            field_name="total_reward",
        )

        _finite_number(
            self.bootstrap_value,
            field_name="bootstrap_value",
        )

        if not isinstance(
            self.terminated,
            bool,
        ):
            raise TypeError(
                "terminated must be bool"
            )

        if not isinstance(
            self.truncated,
            bool,
        ):
            raise TypeError(
                "truncated must be bool"
            )

        if (
            not isinstance(
                self.termination_reason,
                str,
            )
            or not self.termination_reason
        ):
            raise ValueError(
                "termination_reason must not be empty"
            )

        object.__setattr__(
            self,
            "reset_info",
            MappingProxyType(
                dict(self.reset_info)
            ),
        )

    @property
    def steps(self) -> int:
        return len(self.transitions)

    @property
    def final_observation(
        self,
    ) -> Observation:
        return (
            self.transitions[-1]
            .next_observation
        )


def collect_bounded_rollout(
    environment: Environment,
    agent: ActorCriticPolicyAgent,
    *,
    max_steps: int,
    seed: int | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
    bootstrap_on_environment_truncation: bool = False,
) -> CollectedRolloutResult:
    """Collect exactly one rollout and always close the environment."""

    if not isinstance(
        environment,
        Environment,
    ):
        raise TypeError(
            "environment must be an Environment"
        )

    if not isinstance(
        agent,
        ActorCriticPolicyAgent,
    ):
        raise TypeError(
            "agent must be an ActorCriticPolicyAgent"
        )

    validated_max_steps = (
        _positive_integer(
            max_steps,
            field_name="max_steps",
        )
    )

    if not isinstance(
        normalize_advantages,
        bool,
    ):
        raise TypeError(
            "normalize_advantages must be bool"
        )

    if not isinstance(
        bootstrap_on_environment_truncation,
        bool,
    ):
        raise TypeError(
            "bootstrap_on_environment_truncation "
            "must be bool"
        )

    buffer = RolloutBuffer(
        max_steps=validated_max_steps
    )

    transitions: list[
        CollectedRolloutTransition
    ] = []

    total_reward = 0.0
    terminated = False
    environment_truncated = False
    reason = "max_steps_reached"

    initial_observation: (
        Observation | None
    ) = None

    reset_info: Mapping[
        str,
        Any,
    ] = {}

    final_observation: (
        Observation | None
    ) = None

    try:
        reset_result = environment.reset(
            seed=seed
        )

        if (
            not isinstance(
                reset_result,
                tuple,
            )
            or len(reset_result) != 2
        ):
            raise TypeError(
                "environment reset must return "
                "(Observation, info)"
            )

        initial_observation = (
            _validate_observation(
                reset_result[0],
                source="environment reset",
            )
        )

        if not isinstance(
            reset_result[1],
            Mapping,
        ):
            raise TypeError(
                "environment reset info "
                "must be a mapping"
            )

        reset_info = dict(
            reset_result[1]
        )

        observation = (
            initial_observation
        )

        for step_index in range(
            validated_max_steps
        ):
            action = agent.act(
                observation
            )

            if (
                agent.last_log_prob is None
                or agent.last_value is None
                or agent.last_entropy is None
            ):
                raise RuntimeError(
                    "agent did not expose PPO "
                    "decision estimates"
                )

            result = environment.step(
                action
            )

            if not isinstance(
                result,
                StepResult,
            ):
                raise TypeError(
                    "environment step must return "
                    "a StepResult"
                )

            next_observation = (
                _validate_observation(
                    result.observation,
                    source="environment step",
                )
            )

            if (
                result.terminated
                and result.truncated
            ):
                raise ValueError(
                    "environment step cannot be both "
                    "terminated and truncated"
                )

            reward = _finite_number(
                result.reward,
                field_name="environment reward",
            )

            frame = _frame_copy(
                observation
            )

            buffer.append(
                RolloutTransition(
                    frames=frame,
                    action_index=int(
                        action.action
                    ),
                    reward=reward,
                    terminated=(
                        result.terminated
                    ),
                    truncated=(
                        result.truncated
                    ),
                    old_log_prob=(
                        agent.last_log_prob
                    ),
                    old_value=(
                        agent.last_value
                    ),
                    duration_ticks=(
                        action.duration_ticks
                    ),
                )
            )

            transitions.append(
                CollectedRolloutTransition(
                    step_index=step_index,
                    observation=observation,
                    action=action,
                    next_observation=(
                        next_observation
                    ),
                    reward=reward,
                    terminated=(
                        result.terminated
                    ),
                    truncated=(
                        result.truncated
                    ),
                    old_log_prob=(
                        agent.last_log_prob
                    ),
                    old_value=(
                        agent.last_value
                    ),
                    entropy=(
                        agent.last_entropy
                    ),
                    info=result.info,
                )
            )

            total_reward += reward
            observation = (
                next_observation
            )
            final_observation = (
                next_observation
            )

            if result.terminated:
                terminated = True

                raw_reason = result.info.get(
                    "termination_reason"
                )

                reason = (
                    raw_reason
                    if isinstance(
                        raw_reason,
                        str,
                    )
                    and raw_reason
                    else "environment_terminated"
                )
                break

            if result.truncated:
                environment_truncated = True

                raw_reason = result.info.get(
                    "termination_reason"
                )

                reason = (
                    raw_reason
                    if isinstance(
                        raw_reason,
                        str,
                    )
                    and raw_reason
                    else "environment_truncated"
                )
                break

        if final_observation is None:
            raise RuntimeError(
                "rollout produced no final observation"
            )

        if terminated:
            bootstrap_value = 0.0
        elif environment_truncated:
            bootstrap_value = (
                agent.estimate_value(
                    final_observation
                )
                if (
                    bootstrap_on_environment_truncation
                )
                else 0.0
            )
        else:
            bootstrap_value = (
                agent.estimate_value(
                    final_observation
                )
            )

        batch = buffer.finish(
            last_value=bootstrap_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=(
                normalize_advantages
            ),
        )

        return CollectedRolloutResult(
            initial_observation=(
                initial_observation
            ),
            reset_info=reset_info,
            transitions=tuple(
                transitions
            ),
            batch=batch,
            total_reward=total_reward,
            terminated=terminated,
            truncated=(
                environment_truncated
                or not terminated
            ),
            termination_reason=reason,
            bootstrap_value=(
                bootstrap_value
            ),
        )
    finally:
        environment.close()
