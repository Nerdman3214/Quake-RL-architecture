"""Tests for the actor-critic PPO collection agent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from RL.actions.contracts import DiscreteAction
from RL.agents import (
    ActorCriticPolicyAgent,
    VisualActorCriticNetwork,
)
from RL.observations.contracts import Observation


def observation(
    tick: int = 0,
) -> Observation:
    return Observation(
        frame=np.zeros(
            (4, 3, 90, 160),
            dtype=np.float32,
        ),
        telemetry=None,
        tick=tick,
    )


def test_deterministic_agent_exposes_estimates() -> None:
    model = VisualActorCriticNetwork()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        model.policy_head.bias[
            int(DiscreteAction.FIRE)
        ] = 4.0

        model.value_head.bias[0] = 2.25

    agent = ActorCriticPolicyAgent(
        model,
        deterministic=True,
    )

    command = agent.act(
        observation()
    )

    assert command.action == (
        DiscreteAction.FIRE
    )

    assert agent.last_action == (
        DiscreteAction.FIRE
    )

    assert agent.last_log_prob is not None
    assert agent.last_value == pytest.approx(
        2.25
    )
    assert agent.last_entropy is not None
    assert agent.last_logits is not None
    assert len(agent.last_logits) == 11
    assert len(agent.last_action_scores) == 11


def test_stochastic_agent_returns_known_action() -> None:
    torch.manual_seed(81)

    agent = ActorCriticPolicyAgent(
        VisualActorCriticNetwork(),
        deterministic=False,
    )

    command = agent.act(
        observation()
    )

    assert isinstance(
        command.action,
        DiscreteAction,
    )

    assert agent.last_log_prob is not None
    assert agent.last_value is not None


def test_value_estimate_preserves_last_action() -> None:
    model = VisualActorCriticNetwork()

    agent = ActorCriticPolicyAgent(
        model,
        deterministic=True,
    )

    command = agent.act(
        observation(0)
    )

    previous_action = agent.last_action
    previous_log_prob = agent.last_log_prob

    value = agent.estimate_value(
        observation(1)
    )

    assert torch.isfinite(
        torch.tensor(value)
    )

    assert agent.last_action == (
        previous_action
    )

    assert agent.last_log_prob == (
        previous_log_prob
    )

    assert command.action == (
        previous_action
    )


def test_agent_rejects_bad_frame_shape() -> None:
    agent = ActorCriticPolicyAgent(
        VisualActorCriticNetwork()
    )

    bad = Observation(
        frame=np.zeros(
            (4, 3, 84, 84),
            dtype=np.float32,
        ),
        telemetry=None,
        tick=0,
    )

    with pytest.raises(
        ValueError,
        match="unexpected",
    ):
        agent.act(bad)
