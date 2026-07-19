"""Tests for the conservative engine-bridge environment."""

from __future__ import annotations

import pytest

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.env import BridgeEnvironment
from RL.observations.contracts import Observation


def make_observation(
    tick: int,
) -> Observation:
    return Observation(
        frame=None,
        telemetry=None,
        tick=tick,
    )


class FakeBridge:
    """Record bridge lifecycle and return queued observations."""

    def __init__(
        self,
        *,
        reset_observations: list[object] | None = None,
        step_observations: list[object] | None = None,
        fail_close: bool = False,
    ) -> None:
        self.reset_observations = list(
            reset_observations
            if reset_observations is not None
            else [make_observation(0)]
        )

        self.step_observations = list(
            step_observations
            if step_observations is not None
            else [make_observation(1)]
        )

        self.fail_close = fail_close

        self.events: list[str] = []
        self.actions: list[ActionCommand] = []

    def connect(self) -> None:
        self.events.append("connect")

    def reset_match(self) -> object:
        self.events.append("reset_match")

        if not self.reset_observations:
            raise AssertionError(
                "no queued reset observation"
            )

        return self.reset_observations.pop(0)

    def send_action(
        self,
        action: ActionCommand,
    ) -> None:
        self.events.append("send_action")
        self.actions.append(action)

    def read_observation(self) -> object:
        self.events.append("read_observation")

        if not self.step_observations:
            raise AssertionError(
                "no queued step observation"
            )

        return self.step_observations.pop(0)

    def close(self) -> None:
        self.events.append("close")

        if self.fail_close:
            raise RuntimeError("close failed")


def forward_command() -> ActionCommand:
    return ActionCommand(
        action=DiscreteAction.FORWARD,
        duration_ticks=1,
    )


def test_reset_connects_and_returns_metadata() -> None:
    bridge = FakeBridge()
    environment = BridgeEnvironment(bridge)

    observation, info = environment.reset(
        seed=123,
    )

    assert observation.tick == 0
    assert bridge.events == [
        "connect",
        "reset_match",
    ]

    assert environment.connected
    assert environment.ready
    assert not environment.closed
    assert environment.episode_step == 0

    assert info == {
        "reward_mode": "disabled",
        "termination_mode": "disabled",
        "reset_mode": (
            "bridge_local_observation_state_only"
        ),
        "seed_requested": 123,
        "seed_applied": False,
        "episode_step": 0,
    }


def test_repeated_reset_reuses_connection() -> None:
    bridge = FakeBridge(
        reset_observations=[
            make_observation(0),
            make_observation(10),
        ],
        step_observations=[
            make_observation(1),
        ],
    )

    environment = BridgeEnvironment(bridge)

    environment.reset()
    environment.step(forward_command())

    observation, info = environment.reset()

    assert observation.tick == 10
    assert environment.episode_step == 0
    assert info["episode_step"] == 0

    assert bridge.events.count("connect") == 1
    assert bridge.events.count("reset_match") == 2


def test_step_requires_reset() -> None:
    environment = BridgeEnvironment(
        FakeBridge()
    )

    with pytest.raises(
        RuntimeError,
        match="reset before step",
    ):
        environment.step(
            forward_command()
        )


def test_step_delegates_action_then_observation() -> None:
    bridge = FakeBridge()
    environment = BridgeEnvironment(bridge)

    environment.reset()
    result = environment.step(
        forward_command()
    )

    assert bridge.events == [
        "connect",
        "reset_match",
        "send_action",
        "read_observation",
    ]

    assert bridge.actions == [
        forward_command()
    ]

    assert result.observation.tick == 1
    assert result.reward == 0.0
    assert result.terminated is False
    assert result.truncated is False

    assert result.info[
        "reward_mode"
    ] == "disabled"

    assert result.info[
        "termination_mode"
    ] == "disabled"

    assert result.info["episode_step"] == 1


def test_step_increments_episode_step() -> None:
    bridge = FakeBridge(
        step_observations=[
            make_observation(1),
            make_observation(2),
        ]
    )

    environment = BridgeEnvironment(bridge)

    environment.reset()

    first = environment.step(
        forward_command()
    )

    second = environment.step(
        forward_command()
    )

    assert first.info["episode_step"] == 1
    assert second.info["episode_step"] == 2
    assert environment.episode_step == 2


def test_reset_rejects_invalid_observation() -> None:
    bridge = FakeBridge(
        reset_observations=[
            "not-an-observation"
        ]
    )

    environment = BridgeEnvironment(bridge)

    with pytest.raises(
        TypeError,
        match="must return an Observation",
    ):
        environment.reset()

    assert not environment.ready


def test_step_rejects_invalid_observation() -> None:
    bridge = FakeBridge(
        step_observations=[
            "not-an-observation"
        ]
    )

    environment = BridgeEnvironment(bridge)
    environment.reset()

    with pytest.raises(
        TypeError,
        match="must return an Observation",
    ):
        environment.step(
            forward_command()
        )

    assert environment.episode_step == 0


def test_close_is_idempotent() -> None:
    bridge = FakeBridge()
    environment = BridgeEnvironment(bridge)

    environment.reset()
    environment.close()
    environment.close()

    assert bridge.events.count("close") == 1
    assert environment.closed
    assert not environment.connected
    assert not environment.ready


def test_close_marks_closed_when_bridge_close_fails() -> None:
    bridge = FakeBridge(
        fail_close=True,
    )

    environment = BridgeEnvironment(bridge)
    environment.reset()

    with pytest.raises(
        RuntimeError,
        match="close failed",
    ):
        environment.close()

    assert environment.closed
    assert not environment.connected
    assert not environment.ready


def test_operations_are_rejected_after_close() -> None:
    environment = BridgeEnvironment(
        FakeBridge()
    )

    environment.close()

    with pytest.raises(
        RuntimeError,
        match="environment is closed",
    ):
        environment.reset()

    with pytest.raises(
        RuntimeError,
        match="environment is closed",
    ):
        environment.step(
            forward_command()
        )
