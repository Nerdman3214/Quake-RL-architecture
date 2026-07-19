"""Tests for the conservative engine-bridge environment."""

from __future__ import annotations

import pytest

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.env import BridgeEnvironment
from RL.events import Event
from RL.rewards import (
    EventStepOutcome,
    RewardLedger,
)
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


class FakeEventProcessor:
    """Record event-boundary operations and return queued outcomes."""

    def __init__(
        self,
        *,
        outcomes: list[object] | None = None,
        fail_reset: bool = False,
        fail_step: bool = False,
    ) -> None:
        self.outcomes = list(
            outcomes
            if outcomes is not None
            else []
        )

        self.fail_reset = fail_reset
        self.fail_step = fail_step

        self.reset_calls = 0
        self.process_calls = 0

        self.initialized = False
        self.cursor_offset = 120
        self.history_event_count = 4
        self.priming_reason = "active_match"
        self.primed_event_count = 3
        self.match_active = True
        self.episode_done = False
        self.current_mode = "dm"
        self.controlled_team = "RED"

    def reset_episode(self) -> None:
        self.reset_calls += 1

        if self.fail_reset:
            raise RuntimeError(
                "event reset failed"
            )

        self.initialized = True
        self.episode_done = False

    def process_step(self) -> object:
        self.process_calls += 1

        if self.fail_step:
            raise RuntimeError(
                "event step failed"
            )

        if self.outcomes:
            outcome = self.outcomes.pop(0)
        else:
            outcome = EventStepOutcome(
                events=(),
                reward_ledger=RewardLedger(),
                terminated=False,
                truncated=False,
                termination_reason=None,
                match_active=self.match_active,
            )

        if isinstance(
            outcome,
            EventStepOutcome,
        ):
            self.match_active = outcome.match_active

            self.episode_done = (
                outcome.terminated
                or outcome.truncated
            )

        return outcome


def make_event_outcome(
    *,
    events: tuple[Event, ...] = (),
    ledger: RewardLedger | None = None,
    terminated: bool = False,
    truncated: bool = False,
    reason: str | None = None,
    match_active: bool = True,
) -> EventStepOutcome:
    return EventStepOutcome(
        events=events,
        reward_ledger=(
            ledger
            if ledger is not None
            else RewardLedger()
        ),
        terminated=terminated,
        truncated=truncated,
        termination_reason=reason,
        match_active=match_active,
    )


def test_constructor_rejects_invalid_event_processor() -> None:
    with pytest.raises(
        TypeError,
        match="event_processor must provide",
    ):
        BridgeEnvironment(
            FakeBridge(),
            event_processor=object(),
        )


def test_event_enabled_reset_establishes_boundary() -> None:
    bridge = FakeBridge()
    processor = FakeEventProcessor()

    environment = BridgeEnvironment(
        bridge,
        event_processor=processor,
    )

    observation, info = environment.reset(
        seed=456,
    )

    assert observation.tick == 0
    assert environment.event_enabled
    assert environment.ready
    assert processor.reset_calls == 1

    assert info["reward_mode"] == (
        "authoritative_server_events"
    )

    assert info["termination_mode"] == (
        "authoritative_server_events"
    )

    assert info["reset_mode"] == (
        "bridge_local_observation_state_"
        "with_event_boundary"
    )

    assert info["seed_requested"] == 456
    assert info["event_processor_initialized"]
    assert info["event_cursor_offset"] == 120
    assert info["event_history_count"] == 4
    assert info["event_priming_reason"] == (
        "active_match"
    )
    assert info["event_primed_count"] == 3
    assert info["event_match_active"]
    assert info["event_game_mode"] == "dm"
    assert info["event_controlled_team"] == "RED"


def test_repeated_reset_reinitializes_event_processor() -> None:
    bridge = FakeBridge(
        reset_observations=[
            make_observation(0),
            make_observation(10),
        ]
    )

    processor = FakeEventProcessor()

    environment = BridgeEnvironment(
        bridge,
        event_processor=processor,
    )

    environment.reset()
    observation, _ = environment.reset()

    assert observation.tick == 10
    assert processor.reset_calls == 2
    assert bridge.events.count("connect") == 1
    assert bridge.events.count("reset_match") == 2


def test_event_enabled_step_returns_reward_and_audit_info() -> None:
    outcome = make_event_outcome(
        events=(
            Event(
                type="player_kill",
                data={
                    "killer": "Noobnog",
                    "victim": "[BOT]Dominator",
                },
            ),
        ),
        ledger=RewardLedger(
            frag=1.0,
        ),
        match_active=True,
    )

    processor = FakeEventProcessor(
        outcomes=[outcome]
    )

    environment = BridgeEnvironment(
        FakeBridge(),
        event_processor=processor,
    )

    environment.reset()

    result = environment.step(
        forward_command()
    )

    assert result.reward == 1.0
    assert not result.terminated
    assert not result.truncated
    assert environment.ready
    assert environment.episode_step == 1
    assert processor.process_calls == 1

    assert result.info["event_count"] == 1

    assert result.info["event_types"] == (
        "player_kill",
    )

    assert result.info[
        "reward_components"
    ]["frag"] == 1.0

    assert result.info[
        "termination_reason"
    ] is None


def test_terminal_event_outcome_requires_environment_reset() -> None:
    processor = FakeEventProcessor(
        outcomes=[
            make_event_outcome(
                events=(
                    Event(
                        type="match_ended",
                        data={},
                    ),
                ),
                ledger=RewardLedger(
                    win=5.0,
                ),
                terminated=True,
                reason="match_ended",
                match_active=False,
            )
        ]
    )

    environment = BridgeEnvironment(
        FakeBridge(),
        event_processor=processor,
    )

    environment.reset()

    result = environment.step(
        forward_command()
    )

    assert result.reward == 5.0
    assert result.terminated
    assert not result.truncated
    assert not environment.ready

    assert result.info[
        "termination_reason"
    ] == "match_ended"

    assert result.info[
        "event_episode_done"
    ]

    with pytest.raises(
        RuntimeError,
        match="reset before step",
    ):
        environment.step(
            forward_command()
        )


def test_event_processor_reset_failure_leaves_not_ready() -> None:
    processor = FakeEventProcessor(
        fail_reset=True,
    )

    environment = BridgeEnvironment(
        FakeBridge(),
        event_processor=processor,
    )

    with pytest.raises(
        RuntimeError,
        match="event reset failed",
    ):
        environment.reset()

    assert environment.connected
    assert not environment.ready
    assert environment.episode_step == 0


def test_event_processor_step_failure_requires_reset() -> None:
    processor = FakeEventProcessor(
        fail_step=True,
    )

    environment = BridgeEnvironment(
        FakeBridge(),
        event_processor=processor,
    )

    environment.reset()

    with pytest.raises(
        RuntimeError,
        match="event step failed",
    ):
        environment.step(
            forward_command()
        )

    assert environment.episode_step == 0
    assert not environment.ready


def test_invalid_event_outcome_requires_reset() -> None:
    processor = FakeEventProcessor(
        outcomes=[
            "not-an-event-outcome",
        ]
    )

    environment = BridgeEnvironment(
        FakeBridge(),
        event_processor=processor,
    )

    environment.reset()

    with pytest.raises(
        TypeError,
        match="must return an event step outcome",
    ):
        environment.step(
            forward_command()
        )

    assert environment.episode_step == 0
    assert not environment.ready
