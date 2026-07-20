"""Strictly bounded execution of one agent episode."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from RL.actions.contracts import ActionCommand
from RL.agents.baselines.contracts import Agent
from RL.env.core.contracts import (
    Environment,
    StepResult,
)
from RL.observations.contracts import Observation


@dataclass(frozen=True)
class EpisodeTransition:
    """One validated transition produced by a live episode runner."""

    step_index: int
    observation: Observation
    action: ActionCommand
    next_observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError(
                "step_index must be a nonnegative integer"
            )

        if not isinstance(
            self.observation,
            Observation,
        ):
            raise TypeError(
                "observation must be an Observation"
            )

        if not isinstance(
            self.action,
            ActionCommand,
        ):
            raise TypeError(
                "action must be an ActionCommand"
            )

        if not isinstance(
            self.next_observation,
            Observation,
        ):
            raise TypeError(
                "next_observation must be an Observation"
            )

        if (
            isinstance(self.reward, bool)
            or not isinstance(
                self.reward,
                (int, float),
            )
            or not math.isfinite(float(self.reward))
        ):
            raise ValueError(
                "reward must be a finite number"
            )

        if not isinstance(self.terminated, bool):
            raise TypeError(
                "terminated must be bool"
            )

        if not isinstance(self.truncated, bool):
            raise TypeError(
                "truncated must be bool"
            )

    def to_summary_record(self) -> dict[str, Any]:
        """Return a lightweight JSON-compatible transition summary."""

        return {
            "step_index": self.step_index,
            "observation_tick": self.observation.tick,
            "action": {
                "name": self.action.action.name,
                "value": int(self.action.action),
                "duration_ticks": (
                    self.action.duration_ticks
                ),
            },
            "next_observation_tick": (
                self.next_observation.tick
            ),
            "reward": float(self.reward),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": dict(self.info),
        }


@dataclass(frozen=True)
class EpisodeRunResult:
    """Summary and transitions from one bounded episode."""

    initial_observation: Observation
    reset_info: Mapping[str, Any]
    transitions: tuple[EpisodeTransition, ...]
    total_reward: float
    terminated: bool
    truncated: bool
    termination_reason: str | None

    @property
    def steps(self) -> int:
        """Return the number of completed steps."""

        return len(self.transitions)

    @property
    def final_observation(self) -> Observation:
        """Return the final observation produced by the episode."""

        if not self.transitions:
            return self.initial_observation

        return self.transitions[-1].next_observation

    def to_summary_record(self) -> dict[str, Any]:
        """Return a JSON-compatible episode summary."""

        return {
            "steps": self.steps,
            "initial_observation_tick": (
                self.initial_observation.tick
            ),
            "final_observation_tick": (
                self.final_observation.tick
            ),
            "total_reward": float(self.total_reward),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": (
                self.termination_reason
            ),
            "reset_info": dict(self.reset_info),
            "transitions": [
                transition.to_summary_record()
                for transition in self.transitions
            ],
        }


TransitionCallback = Callable[
    [EpisodeTransition],
    None,
]


def _validate_max_steps(max_steps: int) -> int:
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps <= 0
    ):
        raise ValueError(
            "max_steps must be a positive integer"
        )

    return max_steps


def _validate_observation(
    observation: object,
    *,
    source: str,
) -> Observation:
    if not isinstance(observation, Observation):
        raise TypeError(
            f"{source} must return an Observation"
        )

    return observation


def _validate_action(action: object) -> ActionCommand:
    if not isinstance(action, ActionCommand):
        raise TypeError(
            "agent must return an ActionCommand"
        )

    return action


def _validate_step_result(
    result: object,
) -> StepResult:
    if not isinstance(result, StepResult):
        raise TypeError(
            "environment step must return a StepResult"
        )

    _validate_observation(
        result.observation,
        source="environment step",
    )

    if (
        isinstance(result.reward, bool)
        or not isinstance(
            result.reward,
            (int, float),
        )
        or not math.isfinite(float(result.reward))
    ):
        raise ValueError(
            "environment reward must be finite"
        )

    if not isinstance(result.terminated, bool):
        raise TypeError(
            "environment terminated must be bool"
        )

    if not isinstance(result.truncated, bool):
        raise TypeError(
            "environment truncated must be bool"
        )

    if not isinstance(result.info, dict):
        raise TypeError(
            "environment info must be a dict"
        )

    return result


def run_bounded_episode(
    environment: Environment,
    agent: Agent,
    *,
    max_steps: int,
    seed: int | None = None,
    on_transition: TransitionCallback | None = None,
) -> EpisodeRunResult:
    """Run exactly one episode with a strict maximum step count.

    The environment is always closed before this function returns or
    propagates an exception. Reaching ``max_steps`` creates a runner-level
    truncation without requesting a server restart or another episode.
    """

    if not isinstance(environment, Environment):
        raise TypeError(
            "environment must implement Environment"
        )

    if not isinstance(agent, Agent):
        raise TypeError(
            "agent must implement Agent"
        )

    validated_max_steps = _validate_max_steps(
        max_steps
    )

    if (
        on_transition is not None
        and not callable(on_transition)
    ):
        raise TypeError(
            "on_transition must be callable"
        )

    transitions: list[EpisodeTransition] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    termination_reason: str | None = None

    try:
        reset_result = environment.reset(
            seed=seed
        )

        if (
            not isinstance(reset_result, tuple)
            or len(reset_result) != 2
        ):
            raise TypeError(
                "environment reset must return "
                "(Observation, info)"
            )

        initial_observation = _validate_observation(
            reset_result[0],
            source="environment reset",
        )

        reset_info_value = reset_result[1]

        if not isinstance(reset_info_value, dict):
            raise TypeError(
                "environment reset info must be a dict"
            )

        reset_info = dict(reset_info_value)
        observation = initial_observation

        for step_index in range(
            validated_max_steps
        ):
            action = _validate_action(
                agent.act(observation)
            )

            step_result = _validate_step_result(
                environment.step(action)
            )

            reward = float(step_result.reward)
            total_reward += reward

            effective_terminated = (
                step_result.terminated
            )
            effective_truncated = (
                step_result.truncated
            )
            transition_info = dict(
                step_result.info
            )

            reached_max_steps = (
                step_index + 1
                == validated_max_steps
            )

            if (
                reached_max_steps
                and not effective_terminated
                and not effective_truncated
            ):
                effective_truncated = True
                transition_info[
                    "runner_truncation_reason"
                ] = "max_steps_reached"
                transition_info[
                    "runner_max_steps"
                ] = validated_max_steps

            transition = EpisodeTransition(
                step_index=step_index,
                observation=observation,
                action=action,
                next_observation=(
                    step_result.observation
                ),
                reward=reward,
                terminated=effective_terminated,
                truncated=effective_truncated,
                info=transition_info,
            )

            transitions.append(transition)

            if on_transition is not None:
                on_transition(transition)

            observation = step_result.observation

            if (
                effective_terminated
                or effective_truncated
            ):
                terminated = effective_terminated
                truncated = effective_truncated

                raw_reason = transition_info.get(
                    "termination_reason"
                )

                if (
                    isinstance(raw_reason, str)
                    and raw_reason
                ):
                    termination_reason = raw_reason
                elif reached_max_steps:
                    termination_reason = (
                        "max_steps_reached"
                    )
                elif effective_terminated:
                    termination_reason = (
                        "environment_terminated"
                    )
                else:
                    termination_reason = (
                        "environment_truncated"
                    )

                break

        return EpisodeRunResult(
            initial_observation=(
                initial_observation
            ),
            reset_info=reset_info,
            transitions=tuple(transitions),
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            termination_reason=(
                termination_reason
            ),
        )

    finally:
        environment.close()
