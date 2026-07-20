"""Tests for the deterministic fixed-action baseline agent."""

from __future__ import annotations

import pytest

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.agents import (
    Agent,
    FixedActionAgent,
)
from RL.observations.contracts import Observation


def make_observation(
    tick: int = 0,
) -> Observation:
    return Observation(
        frame=None,
        telemetry=None,
        tick=tick,
    )


def forward_command() -> ActionCommand:
    return ActionCommand(
        action=DiscreteAction.FORWARD,
        duration_ticks=1,
    )


def test_fixed_action_agent_implements_agent_contract() -> None:
    command = forward_command()
    agent = FixedActionAgent(command)

    assert isinstance(agent, Agent)
    assert agent.command is command
    assert agent.act(make_observation()) is command


def test_fixed_action_agent_is_deterministic() -> None:
    command = ActionCommand(
        action=DiscreteAction.TURN_RIGHT,
        duration_ticks=2,
    )

    agent = FixedActionAgent(command)

    first = agent.act(make_observation(10))
    second = agent.act(make_observation(11))

    assert first is command
    assert second is command
    assert first == second


def test_fixed_action_agent_rejects_invalid_command() -> None:
    with pytest.raises(
        TypeError,
        match="command must be an ActionCommand",
    ):
        FixedActionAgent(
            "FORWARD",  # type: ignore[arg-type]
        )


def test_fixed_action_agent_rejects_invalid_observation() -> None:
    agent = FixedActionAgent(
        forward_command()
    )

    with pytest.raises(
        TypeError,
        match="observation must be an Observation",
    ):
        agent.act(
            "not-an-observation",  # type: ignore[arg-type]
        )
