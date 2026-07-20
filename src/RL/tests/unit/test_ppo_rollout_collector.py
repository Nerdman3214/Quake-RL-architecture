"""Tests for bounded live PPO rollout collection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from RL.actions.contracts import DiscreteAction
from RL.agents import (
    ActorCriticPolicyAgent,
    VisualActorCriticNetwork,
)
from RL.env.core.contracts import (
    Environment,
    StepResult,
)
from RL.observations.contracts import Observation
from RL.training.ppo import (
    collect_bounded_rollout,
)


def observation(
    tick: int,
) -> Observation:
    return Observation(
        frame=np.full(
            (4, 3, 90, 160),
            fill_value=tick / 10.0,
            dtype=np.float32,
        ),
        telemetry=None,
        tick=tick,
    )


class FakeEnvironment(Environment):
    def __init__(
        self,
        results: list[StepResult],
    ) -> None:
        self.results = list(results)
        self.closed = False
        self.actions = []

    def reset(
        self,
        *,
        seed: int | None = None,
    ):
        return (
            observation(0),
            {
                "seed": seed,
            },
        )

    def step(self, action):
        self.actions.append(action)

        if not self.results:
            raise AssertionError(
                "no queued step result"
            )

        return self.results.pop(0)

    def close(self) -> None:
        self.closed = True


def make_agent() -> ActorCriticPolicyAgent:
    model = VisualActorCriticNetwork()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        model.policy_head.bias[
            int(DiscreteAction.FORWARD)
        ] = 3.0

        model.value_head.bias[0] = 1.5

    return ActorCriticPolicyAgent(
        model,
        deterministic=True,
    )


def test_max_steps_rollout_bootstraps() -> None:
    environment = FakeEnvironment(
        [
            StepResult(
                observation=observation(1),
                reward=0.25,
                terminated=False,
                truncated=False,
                info={},
            ),
            StepResult(
                observation=observation(2),
                reward=1.0,
                terminated=False,
                truncated=False,
                info={},
            ),
        ]
    )

    result = collect_bounded_rollout(
        environment,
        make_agent(),
        max_steps=2,
        seed=7,
    )

    assert result.steps == 2
    assert result.total_reward == (
        pytest.approx(1.25)
    )
    assert not result.terminated
    assert result.truncated
    assert result.termination_reason == (
        "max_steps_reached"
    )
    assert result.bootstrap_value == (
        pytest.approx(1.5)
    )
    assert result.batch.frames.shape == (
        2,
        4,
        3,
        90,
        160,
    )
    assert result.batch.action_indices.tolist() == [
        int(DiscreteAction.FORWARD),
        int(DiscreteAction.FORWARD),
    ]
    assert environment.closed


def test_terminal_rollout_uses_zero_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = FakeEnvironment(
        [
            StepResult(
                observation=observation(1),
                reward=5.0,
                terminated=True,
                truncated=False,
                info={
                    "termination_reason": (
                        "match_ended"
                    ),
                },
            )
        ]
    )

    agent = make_agent()

    def fail_estimate(
        _: Observation,
    ) -> float:
        raise AssertionError(
            "terminal rollout must not bootstrap"
        )

    monkeypatch.setattr(
        agent,
        "estimate_value",
        fail_estimate,
    )

    result = collect_bounded_rollout(
        environment,
        agent,
        max_steps=4,
    )

    assert result.terminated
    assert not result.truncated
    assert result.bootstrap_value == 0.0
    assert result.termination_reason == (
        "match_ended"
    )
    assert environment.closed


def test_environment_truncation_defaults_to_zero_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = FakeEnvironment(
        [
            StepResult(
                observation=observation(1),
                reward=-2.0,
                terminated=False,
                truncated=True,
                info={
                    "termination_reason": (
                        "controlled_player_disconnected"
                    ),
                },
            )
        ]
    )

    agent = make_agent()

    def fail_estimate(
        _: Observation,
    ) -> float:
        raise AssertionError(
            "default truncation must not bootstrap"
        )

    monkeypatch.setattr(
        agent,
        "estimate_value",
        fail_estimate,
    )

    result = collect_bounded_rollout(
        environment,
        agent,
        max_steps=4,
    )

    assert result.truncated
    assert result.bootstrap_value == 0.0
    assert environment.closed


def test_environment_truncation_can_bootstrap() -> None:
    environment = FakeEnvironment(
        [
            StepResult(
                observation=observation(1),
                reward=0.0,
                terminated=False,
                truncated=True,
                info={
                    "termination_reason": (
                        "time_limit"
                    ),
                },
            )
        ]
    )

    result = collect_bounded_rollout(
        environment,
        make_agent(),
        max_steps=4,
        bootstrap_on_environment_truncation=True,
    )

    assert result.bootstrap_value == (
        pytest.approx(1.5)
    )
    assert environment.closed


def test_collector_closes_after_agent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = FakeEnvironment([])

    agent = make_agent()

    def fail_act(
        _: Observation,
    ):
        raise RuntimeError(
            "decision failed"
        )

    monkeypatch.setattr(
        agent,
        "act",
        fail_act,
    )

    with pytest.raises(
        RuntimeError,
        match="decision failed",
    ):
        collect_bounded_rollout(
            environment,
            agent,
            max_steps=1,
        )

    assert environment.closed
