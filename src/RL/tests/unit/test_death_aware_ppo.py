"""Tests for death-aware and zero-signal-gated PPO."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from RL.agents import (
    ActorCriticPolicyAgent,
    VisualActorCriticNetwork,
)
from RL.env.core.contracts import (
    Environment,
    StepResult,
)
from RL.observations.contracts import (
    Observation,
    PlayerTelemetry,
)
from RL.training.ppo import (
    DeathAwarePPOConfig,
    PPOTrainer,
    run_death_aware_ppo_step,
)


def observation(
    tick: int,
    *,
    alive: bool,
) -> Observation:
    return Observation(
        frame=np.full(
            (4, 3, 90, 160),
            fill_value=tick / 100.0,
            dtype=np.float32,
        ),
        telemetry=PlayerTelemetry(
            health=100 if alive else 0,
            armor=0,
            ammo=20,
            weapon="blaster",
            alive=alive,
            score=0,
            match_time_seconds=float(
                tick
            ),
        ),
        tick=tick,
    )


class SequenceEnvironment(Environment):
    def __init__(
        self,
        results: list[StepResult],
        *,
        initial_alive: bool = True,
    ) -> None:
        self.results = list(
            results
        )

        self.initial_alive = (
            initial_alive
        )

        self.closed = False
        self.step_calls = 0
        self.actions = []

    def reset(
        self,
        *,
        seed: int | None = None,
    ):
        return (
            observation(
                0,
                alive=self.initial_alive,
            ),
            {"seed": seed},
        )

    def step(self, action):
        self.actions.append(
            action
        )

        self.step_calls += 1

        if not self.results:
            raise AssertionError(
                "no queued step result"
            )

        return self.results.pop(0)

    def close(self) -> None:
        self.closed = True


def step_result(
    tick: int,
    *,
    reward: float = 0.0,
    alive: bool = True,
    terminated: bool = False,
    truncated: bool = False,
    event_types=(),
) -> StepResult:
    return StepResult(
        observation=observation(
            tick,
            alive=alive,
        ),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info={
            "event_types": tuple(
                event_types
            ),
        },
    )


def make_components(
    *,
    deterministic: bool = False,
):
    model = VisualActorCriticNetwork()

    agent = ActorCriticPolicyAgent(
        model,
        deterministic=deterministic,
    )

    trainer = PPOTrainer(
        model,
        torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
        ),
    )

    return model, agent, trainer


def parameter_copy(model):
    return tuple(
        parameter.detach().clone()
        for parameter
        in model.parameters()
    )


def parameters_changed(
    before,
    model,
) -> bool:
    return any(
        not torch.equal(
            old,
            new.detach(),
        )
        for old, new
        in zip(
            before,
            model.parameters(),
            strict=True,
        )
    )


def test_zero_signal_segment_skips_update() -> None:
    model, agent, trainer = (
        make_components()
    )

    environment = SequenceEnvironment(
        [
            step_result(1),
            step_result(2),
            step_result(3),
        ]
    )

    before = parameter_copy(
        model
    )

    result = run_death_aware_ppo_step(
        environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=3,
            max_respawn_wait_steps=2,
        ),
    )

    assert result.rollout.steps == 3
    assert not result.rollout.meaningful_signal
    assert result.rollout.hard_limit_reached
    assert result.rollout.signal_reason == (
        "hard_step_limit_zero_signal"
    )

    assert not result.update_performed
    assert result.optimizer_operations == 0
    assert result.update_skipped_reason == (
        "zero_signal_segment"
    )

    assert (
        trainer.optimizer_step_count
        == 0
    )

    assert not parameters_changed(
        before,
        model,
    )

    assert environment.closed


def test_death_updates_and_waits_for_respawn() -> None:
    torch.manual_seed(1101)

    model, agent, trainer = (
        make_components()
    )

    environment = SequenceEnvironment(
        [
            step_result(1),
            step_result(2),
            step_result(
                3,
                reward=-1.0,
                alive=False,
                event_types=(
                    "player_kill",
                ),
            ),
            step_result(
                4,
                alive=False,
            ),
            step_result(
                5,
                alive=True,
                event_types=(
                    "player_now_playing",
                ),
            ),
        ]
    )

    before = parameter_copy(
        model
    )

    result = run_death_aware_ppo_step(
        environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=8,
            max_respawn_wait_steps=4,
        ),
    )

    assert result.rollout.steps == 3
    assert result.rollout.total_reward == (
        pytest.approx(-1.0)
    )

    assert result.rollout.meaningful_signal
    assert result.rollout.death_detected
    assert result.rollout.signal_reason == (
        "death_detected"
    )

    assert result.rollout.respawn_detected
    assert (
        result.rollout.respawn_wait_steps
        == 2
    )

    assert (
        result.rollout
        .post_respawn_observation
        is not None
    )

    assert (
        result.rollout
        .post_respawn_observation
        .telemetry
        .alive
    )

    assert environment.step_calls == 5
    assert result.rollout.batch.terminated.tolist() == [
        False,
        False,
        True,
    ]

    assert result.update_performed
    assert result.optimizer_operations == 1
    assert (
        trainer.optimizer_step_count
        == 1
    )

    assert parameters_changed(
        before,
        model,
    )

    assert environment.closed


def test_positive_reward_stops_without_respawn_wait() -> None:
    _, agent, trainer = (
        make_components()
    )

    environment = SequenceEnvironment(
        [
            step_result(1),
            step_result(
                2,
                reward=1.0,
                alive=True,
            ),
            step_result(3),
        ]
    )

    result = run_death_aware_ppo_step(
        environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=8,
        ),
    )

    assert result.rollout.steps == 2
    assert result.rollout.meaningful_signal
    assert not result.rollout.death_detected
    assert result.rollout.signal_reason == (
        "reward_signal"
    )

    assert (
        result.rollout.respawn_wait_steps
        == 0
    )

    assert not (
        result.rollout.respawn_detected
    )

    assert environment.step_calls == 2
    assert result.update_performed
    assert environment.closed


def test_environment_termination_is_meaningful() -> None:
    _, agent, trainer = (
        make_components()
    )

    environment = SequenceEnvironment(
        [
            step_result(
                1,
                terminated=True,
                alive=True,
            ),
        ]
    )

    result = run_death_aware_ppo_step(
        environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=4,
        ),
    )

    assert result.rollout.meaningful_signal
    assert (
        result.rollout.signal_reason
        == "environment_terminated"
    )

    assert (
        result.rollout
        .environment_terminated
    )

    assert result.update_performed
    assert environment.closed


def test_zero_signal_then_death_uses_fresh_segment() -> None:
    _, agent, trainer = (
        make_components()
    )

    zero_environment = (
        SequenceEnvironment(
            [
                step_result(1),
                step_result(2),
            ]
        )
    )

    first = run_death_aware_ppo_step(
        zero_environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=2,
        ),
    )

    assert not first.update_performed
    assert (
        trainer.optimizer_step_count
        == 0
    )

    death_environment = (
        SequenceEnvironment(
            [
                step_result(
                    1,
                    reward=-1.0,
                    alive=False,
                ),
                step_result(
                    2,
                    alive=True,
                    event_types=(
                        "player_now_playing",
                    ),
                ),
            ]
        )
    )

    second = run_death_aware_ppo_step(
        death_environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=4,
            max_respawn_wait_steps=2,
        ),
    )

    assert second.rollout.steps == 1
    assert second.rollout.batch.rewards.tolist() == (
        pytest.approx([-1.0])
    )

    assert second.update_performed
    assert (
        trainer.optimizer_step_count
        == 1
    )


def test_respawn_wait_is_strictly_bounded() -> None:
    _, agent, trainer = (
        make_components()
    )

    environment = SequenceEnvironment(
        [
            step_result(
                1,
                reward=-1.0,
                alive=False,
            ),
            step_result(
                2,
                alive=False,
            ),
            step_result(
                3,
                alive=False,
            ),
            step_result(
                4,
                alive=True,
            ),
        ]
    )

    result = run_death_aware_ppo_step(
        environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=5,
            max_respawn_wait_steps=2,
        ),
    )

    assert result.rollout.death_detected
    assert not (
        result.rollout.respawn_detected
    )

    assert (
        result.rollout.respawn_wait_steps
        == 2
    )

    assert environment.step_calls == 3
    assert result.update_performed
    assert environment.closed


def test_agent_and_trainer_must_share_model() -> None:
    first_model = (
        VisualActorCriticNetwork()
    )

    second_model = (
        VisualActorCriticNetwork()
    )

    agent = ActorCriticPolicyAgent(
        first_model
    )

    trainer = PPOTrainer(
        second_model,
        torch.optim.Adam(
            second_model.parameters(),
            lr=1e-4,
        ),
    )

    environment = SequenceEnvironment(
        [
            step_result(1),
        ]
    )

    with pytest.raises(
        ValueError,
        match="share",
    ):
        run_death_aware_ppo_step(
            environment,
            agent,
            trainer,
            DeathAwarePPOConfig(
                max_steps=1,
            ),
        )


def test_config_rejects_invalid_limits() -> None:
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        DeathAwarePPOConfig(
            max_steps=0
        )

    with pytest.raises(
        ValueError,
        match="nonnegative integer",
    ):
        DeathAwarePPOConfig(
            max_steps=1,
            max_respawn_wait_steps=-1,
        )

    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        DeathAwarePPOConfig(
            max_steps=1,
            updates_per_signal=0,
        )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        DeathAwarePPOConfig(
            max_steps=1,
            death_reward_threshold=0.0,
        )


def test_nested_event_record_can_trigger_signal() -> None:
    _, agent, trainer = (
        make_components()
    )

    environment = SequenceEnvironment(
        [
            StepResult(
                observation=observation(
                    1,
                    alive=True,
                ),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "events": [
                        {
                            "event_type": (
                                "objective_completed"
                            ),
                        }
                    ],
                },
            )
        ]
    )

    result = run_death_aware_ppo_step(
        environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=3,
        ),
    )

    assert result.rollout.meaningful_signal
    assert result.rollout.signal_reason == (
        "authoritative_event"
    )

    assert result.update_performed

def test_delayed_death_reward_is_attached_to_terminal_action() -> None:
    torch.manual_seed(2201)

    model, agent, trainer = (
        make_components()
    )

    environment = SequenceEnvironment(
        [
            step_result(
                1,
                reward=0.0,
                alive=False,
            ),
            step_result(
                2,
                reward=-1.0,
                alive=False,
                event_types=(
                    "player_kill",
                ),
            ),
            step_result(
                3,
                alive=True,
                event_types=(
                    "player_now_playing",
                ),
            ),
        ]
    )

    before = parameter_copy(
        model
    )

    result = run_death_aware_ppo_step(
        environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=8,
            max_respawn_wait_steps=4,
            death_confirmation_steps=3,
            respawn_fire_interval_steps=1,
        ),
    )

    assert result.rollout.steps == 1
    assert result.rollout.death_detected
    assert result.rollout.death_reward_confirmed
    assert (
        result.rollout
        .death_confirmation_steps
        == 1
    )
    assert (
        result.rollout
        .confirmed_death_reward
        == pytest.approx(-1.0)
    )
    assert (
        result.rollout.total_reward
        == pytest.approx(-1.0)
    )
    assert (
        result.rollout.batch.rewards.tolist()
        == pytest.approx([-1.0])
    )

    assert result.rollout.respawn_detected
    assert result.rollout.respawn_wait_steps == 1
    assert result.rollout.respawn_fire_actions == 1
    assert (
        environment.actions[-1]
        .action.name
        == "FIRE"
    )

    assert result.update_performed
    assert trainer.optimizer_step_count == 1
    assert parameters_changed(
        before,
        model,
    )
    assert environment.closed


def test_unconfirmed_zero_reward_death_skips_update() -> None:
    model, agent, trainer = (
        make_components()
    )

    environment = SequenceEnvironment(
        [
            step_result(
                1,
                reward=0.0,
                alive=False,
            ),
            step_result(
                2,
                reward=0.0,
                alive=False,
            ),
            step_result(
                3,
                reward=0.0,
                alive=False,
            ),
            step_result(
                4,
                reward=0.0,
                alive=True,
            ),
        ]
    )

    before = parameter_copy(
        model
    )

    result = run_death_aware_ppo_step(
        environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=8,
            max_respawn_wait_steps=2,
            death_confirmation_steps=2,
            respawn_fire_interval_steps=1,
        ),
    )

    assert result.rollout.steps == 1
    assert result.rollout.death_detected
    assert not result.rollout.death_reward_confirmed
    assert (
        result.rollout.signal_reason
        == "unconfirmed_death_zero_reward"
    )
    assert not result.rollout.meaningful_signal
    assert (
        result.rollout.death_confirmation_steps
        == 2
    )

    assert result.rollout.respawn_detected
    assert result.rollout.respawn_fire_actions == 1

    assert not result.update_performed
    assert (
        result.update_skipped_reason
        == "zero_signal_segment"
    )
    assert trainer.optimizer_step_count == 0
    assert not parameters_changed(
        before,
        model,
    )
    assert environment.closed

def test_second_death_infers_respawn_without_entering_first_rollout() -> None:
    torch.manual_seed(3301)

    model, agent, trainer = (
        make_components()
    )

    environment = SequenceEnvironment(
        [
            step_result(
                1,
                reward=-1.0,
                alive=False,
                event_types=(
                    "player_kill",
                ),
            ),
            step_result(
                2,
                reward=0.0,
                alive=False,
            ),
            step_result(
                3,
                reward=-1.0,
                alive=False,
                event_types=(
                    "player_kill",
                ),
            ),
        ]
    )

    before = parameter_copy(
        model
    )

    result = run_death_aware_ppo_step(
        environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=8,
            max_respawn_wait_steps=4,
            respawn_fire_interval_steps=8,
        ),
    )

    assert result.rollout.steps == 1
    assert result.rollout.death_detected
    assert result.rollout.death_reward_confirmed

    assert not result.rollout.respawn_detected
    assert result.rollout.respawn_inferred
    assert (
        result.rollout.respawn_signal_reason
        == "second_death_proves_respawn"
    )

    assert result.rollout.respawn_wait_steps == 2
    assert result.rollout.respawn_fire_actions == 1
    assert (
        result.rollout.post_respawn_reward
        == pytest.approx(-1.0)
    )

    assert (
        result.rollout.post_respawn_observation
        is not None
    )
    assert (
        result.rollout
        .post_respawn_observation
        .tick
        == 3
    )

    # The second death proves respawn, but it must not be placed in
    # the first death's on-policy PPO trajectory.
    assert result.rollout.total_reward == (
        pytest.approx(-1.0)
    )
    assert (
        result.rollout.batch.rewards.tolist()
        == pytest.approx([-1.0])
    )
    assert (
        result.rollout.batch.rewards.numel()
        == 1
    )

    assert environment.step_calls == 3
    assert result.update_performed
    assert result.optimizer_operations == 1
    assert trainer.optimizer_step_count == 1
    assert parameters_changed(
        before,
        model,
    )
    assert environment.closed


def test_late_first_death_penalty_is_not_mislabeled_as_respawn() -> None:
    model, agent, trainer = (
        make_components()
    )

    environment = SequenceEnvironment(
        [
            step_result(
                1,
                reward=0.0,
                alive=False,
            ),
            step_result(
                2,
                reward=-1.0,
                alive=False,
                event_types=(
                    "player_kill",
                ),
            ),
        ]
    )

    before = parameter_copy(
        model
    )

    result = run_death_aware_ppo_step(
        environment,
        agent,
        trainer,
        DeathAwarePPOConfig(
            max_steps=4,
            max_respawn_wait_steps=2,
            death_confirmation_steps=0,
            respawn_fire_interval_steps=1,
        ),
    )

    assert result.rollout.steps == 1
    assert result.rollout.death_detected
    assert not result.rollout.death_reward_confirmed
    assert not result.rollout.meaningful_signal

    assert not result.rollout.respawn_detected
    assert not result.rollout.respawn_inferred
    assert (
        result.rollout.respawn_signal_reason
        == (
            "late_death_reward_without_"
            "confirmed_first_death"
        )
    )
    assert (
        result.rollout.post_respawn_reward
        == pytest.approx(-1.0)
    )

    assert not result.update_performed
    assert result.optimizer_operations == 0
    assert (
        result.update_skipped_reason
        == "zero_signal_segment"
    )
    assert trainer.optimizer_step_count == 0
    assert not parameters_changed(
        before,
        model,
    )
    assert environment.closed
