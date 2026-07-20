"""Tests for the strictly bounded single-episode runner."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.agents.baselines.contracts import Agent
from RL.env.core.contracts import (
    Environment,
    StepResult,
)
from RL.episodes import (
    EpisodeTransition,
    run_bounded_episode,
)
from RL.observations.contracts import Observation


def make_observation(tick: int) -> Observation:
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


def make_step_result(
    tick: int,
    *,
    reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    reason: str | None = None,
) -> StepResult:
    info: dict[str, object] = {
        "episode_step": tick,
    }

    if reason is not None:
        info["termination_reason"] = reason

    return StepResult(
        observation=make_observation(tick),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


class RecordingAgent(Agent):
    """Return one configured action and record observations."""

    def __init__(
        self,
        *,
        command: object | None = None,
        fail: bool = False,
    ) -> None:
        self.command = (
            command
            if command is not None
            else forward_command()
        )
        self.fail = fail
        self.observation_ticks: list[int] = []

    def act(
        self,
        observation: Observation,
    ) -> ActionCommand:
        self.observation_ticks.append(
            observation.tick
        )

        if self.fail:
            raise RuntimeError(
                "agent failed"
            )

        return self.command  # type: ignore[return-value]


class FakeEnvironment(Environment):
    """Return queued results and expose cleanup calls."""

    def __init__(
        self,
        results: list[StepResult] | None = None,
        *,
        fail_reset: bool = False,
        fail_step_index: int | None = None,
    ) -> None:
        self.results = list(
            results
            if results is not None
            else []
        )
        self.fail_reset = fail_reset
        self.fail_step_index = fail_step_index

        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.actions: list[ActionCommand] = []

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Observation, dict[str, object]]:
        self.reset_calls += 1

        if self.fail_reset:
            raise RuntimeError(
                "reset failed"
            )

        return make_observation(0), {
            "seed_requested": seed,
        }

    def step(
        self,
        action: ActionCommand,
    ) -> StepResult:
        current_index = self.step_calls
        self.step_calls += 1
        self.actions.append(action)

        if current_index == self.fail_step_index:
            raise RuntimeError(
                "step failed"
            )

        if not self.results:
            raise AssertionError(
                "no queued step result"
            )

        return self.results.pop(0)

    def close(self) -> None:
        self.close_calls += 1


def test_runner_rejects_invalid_max_steps() -> None:
    environment = FakeEnvironment()
    agent = RecordingAgent()

    for invalid_value in (
        0,
        -1,
        True,
    ):
        with pytest.raises(
            ValueError,
            match="positive integer",
        ):
            run_bounded_episode(
                environment,
                agent,
                max_steps=invalid_value,
            )

    assert environment.reset_calls == 0
    assert environment.close_calls == 0


def test_runner_marks_max_step_truncation() -> None:
    environment = FakeEnvironment(
        [
            make_step_result(
                1,
                reward=0.5,
            ),
            make_step_result(
                2,
                reward=0.5,
            ),
            make_step_result(
                3,
                reward=0.5,
            ),
        ]
    )
    agent = RecordingAgent()

    result = run_bounded_episode(
        environment,
        agent,
        max_steps=3,
        seed=123,
    )

    assert result.steps == 3
    assert result.total_reward == 1.5
    assert not result.terminated
    assert result.truncated
    assert result.termination_reason == (
        "max_steps_reached"
    )
    assert result.final_observation.tick == 3

    assert agent.observation_ticks == [
        0,
        1,
        2,
    ]

    assert environment.step_calls == 3
    assert environment.close_calls == 1

    final_transition = result.transitions[-1]

    assert final_transition.truncated
    assert final_transition.info[
        "runner_truncation_reason"
    ] == "max_steps_reached"

    summary = result.to_summary_record()

    assert summary["steps"] == 3
    assert summary[
        "final_observation_tick"
    ] == 3


def test_runner_stops_on_environment_termination() -> None:
    environment = FakeEnvironment(
        [
            make_step_result(
                1,
                reward=5.0,
                terminated=True,
                reason="match_ended",
            ),
            make_step_result(2),
        ]
    )
    agent = RecordingAgent()

    result = run_bounded_episode(
        environment,
        agent,
        max_steps=5,
    )

    assert result.steps == 1
    assert result.total_reward == 5.0
    assert result.terminated
    assert not result.truncated
    assert result.termination_reason == (
        "match_ended"
    )
    assert environment.step_calls == 1
    assert environment.close_calls == 1


def test_runner_stops_on_environment_truncation() -> None:
    environment = FakeEnvironment(
        [
            make_step_result(
                1,
                reward=-2.0,
                truncated=True,
                reason=(
                    "controlled_player_disconnected"
                ),
            ),
            make_step_result(2),
        ]
    )
    agent = RecordingAgent()

    result = run_bounded_episode(
        environment,
        agent,
        max_steps=5,
    )

    assert result.steps == 1
    assert not result.terminated
    assert result.truncated
    assert result.termination_reason == (
        "controlled_player_disconnected"
    )
    assert environment.step_calls == 1
    assert environment.close_calls == 1


def test_runner_closes_when_agent_raises() -> None:
    environment = FakeEnvironment(
        [
            make_step_result(1),
        ]
    )
    agent = RecordingAgent(
        fail=True,
    )

    with pytest.raises(
        RuntimeError,
        match="agent failed",
    ):
        run_bounded_episode(
            environment,
            agent,
            max_steps=3,
        )

    assert environment.step_calls == 0
    assert environment.close_calls == 1


def test_runner_closes_when_step_raises() -> None:
    environment = FakeEnvironment(
        [
            make_step_result(1),
        ],
        fail_step_index=0,
    )
    agent = RecordingAgent()

    with pytest.raises(
        RuntimeError,
        match="step failed",
    ):
        run_bounded_episode(
            environment,
            agent,
            max_steps=3,
        )

    assert environment.step_calls == 1
    assert environment.close_calls == 1


def test_runner_rejects_invalid_agent_action() -> None:
    environment = FakeEnvironment(
        [
            make_step_result(1),
        ]
    )
    agent = RecordingAgent(
        command="FORWARD",
    )

    with pytest.raises(
        TypeError,
        match="agent must return an ActionCommand",
    ):
        run_bounded_episode(
            environment,
            agent,
            max_steps=3,
        )

    assert environment.step_calls == 0
    assert environment.close_calls == 1


def test_transition_callback_receives_ordered_steps() -> None:
    environment = FakeEnvironment(
        [
            make_step_result(1),
            make_step_result(
                2,
                terminated=True,
            ),
        ]
    )
    agent = RecordingAgent()
    received: list[EpisodeTransition] = []

    callback: Callable[
        [EpisodeTransition],
        None,
    ] = received.append

    result = run_bounded_episode(
        environment,
        agent,
        max_steps=5,
        on_transition=callback,
    )

    assert result.steps == 2
    assert [
        transition.step_index
        for transition in received
    ] == [
        0,
        1,
    ]
    assert received[0].observation.tick == 0
    assert received[0].next_observation.tick == 1
    assert received[1].terminated
    assert environment.close_calls == 1


def test_runner_closes_when_reset_raises() -> None:
    environment = FakeEnvironment(
        fail_reset=True,
    )
    agent = RecordingAgent()

    with pytest.raises(
        RuntimeError,
        match="reset failed",
    ):
        run_bounded_episode(
            environment,
            agent,
            max_steps=3,
        )

    assert environment.reset_calls == 1
    assert environment.step_calls == 0
    assert environment.close_calls == 1
