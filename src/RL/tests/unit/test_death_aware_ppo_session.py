"""Tests for isolated, auditable death-aware PPO sessions."""

from __future__ import annotations

import json

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
    DeathAwarePPOTrainingSessionConfig,
    PPOTrainer,
    PPOTrainingProgress,
    load_ppo_training_checkpoint,
    run_bounded_death_aware_ppo_session,
    save_ppo_training_checkpoint,
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
            match_time_seconds=float(tick),
        ),
        tick=tick,
    )


def step_result(
    tick: int,
    *,
    reward: float = 0.0,
    alive: bool = True,
    event_types=(),
) -> StepResult:
    return StepResult(
        observation=observation(
            tick,
            alive=alive,
        ),
        reward=reward,
        terminated=False,
        truncated=False,
        info={
            "event_types": tuple(
                event_types
            ),
        },
    )


class SequenceEnvironment(Environment):
    def __init__(
        self,
        results: list[StepResult],
    ) -> None:
        self.results = list(results)
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
    ):
        return (
            observation(
                0,
                alive=True,
            ),
            {"seed": seed},
        )

    def step(self, action):
        if not self.results:
            raise AssertionError(
                "no queued step result"
            )

        return self.results.pop(0)

    def close(self) -> None:
        self.closed = True


def create_source_checkpoint(
    tmp_path,
) -> tuple:
    model = VisualActorCriticNetwork()

    trainer = PPOTrainer(
        model,
        torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
        ),
        optimizer_step_count=2,
    )

    progress = PPOTrainingProgress(
        rollout_count=5,
        environment_step_count=20,
        completed_episode_count=1,
        cumulative_reward=-2.0,
    )

    path = tmp_path / "source.pt"

    save_ppo_training_checkpoint(
        path,
        trainer,
        progress=progress,
        policy_name="death-aware-test",
        policy_version="source",
    )

    return path, progress


def agent_factory(
    model,
    _: int,
) -> ActorCriticPolicyAgent:
    return ActorCriticPolicyAgent(
        model,
        deterministic=False,
    )


def session_config(
    *,
    attempt_count: int,
) -> DeathAwarePPOTrainingSessionConfig:
    return (
        DeathAwarePPOTrainingSessionConfig(
            attempt_count=attempt_count,
            step_config=(
                DeathAwarePPOConfig(
                    max_steps=4,
                    max_respawn_wait_steps=4,
                    death_confirmation_steps=2,
                    respawn_fire_interval_steps=1,
                    updates_per_signal=1,
                    death_reward_threshold=-1.0,
                    stop_on_kl=False,
                    seed=700,
                )
            ),
        )
    )


def test_rejected_update_isolated_before_direct_respawn_promotion(
    tmp_path,
) -> None:
    source, initial_progress = (
        create_source_checkpoint(tmp_path)
    )

    environments = [
        [
            step_result(
                1,
                reward=1.0,
                alive=True,
            ),
        ],
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
                alive=True,
                event_types=(
                    "player_now_playing",
                ),
            ),
        ],
    ]

    def environment_factory(
        attempt_index: int,
    ) -> Environment:
        return SequenceEnvironment(
            environments[attempt_index]
        )

    before = tmp_path / "before.pt"
    promoted = tmp_path / "promoted.pt"
    audit = tmp_path / "audit.jsonl"

    result = (
        run_bounded_death_aware_ppo_session(
            source,
            environment_factory,
            agent_factory,
            session_config(
                attempt_count=2
            ),
            before_checkpoint_path=before,
            promoted_checkpoint_path=(
                promoted
            ),
            audit_path=audit,
            policy_name="death-aware-test",
            policy_version="promoted",
        )
    )

    assert result.promoted
    assert result.accepted_attempt_index == 1
    assert result.attempts_completed == 2

    first, second = result.attempts

    assert not first.accepted
    assert "death_not_detected" in (
        first.rejection_reasons
    )

    # Both attempts began at optimizer step 2. The rejected update
    # therefore cannot make the accepted checkpoint step 4.
    assert (
        first.source_optimizer_step_count
        == 2
    )
    assert (
        first.ending_optimizer_step_count
        == 3
    )
    assert (
        second.source_optimizer_step_count
        == 2
    )
    assert (
        second.ending_optimizer_step_count
        == 3
    )

    assert second.accepted
    assert second.rollout.death_detected
    assert (
        second.rollout
        .death_reward_confirmed
    )
    assert (
        second.rollout.respawn_detected
    )
    assert not second.rollout.respawn_inferred

    source_loaded = (
        load_ppo_training_checkpoint(
            source
        )
    )
    before_loaded = (
        load_ppo_training_checkpoint(
            before
        )
    )
    promoted_loaded = (
        load_ppo_training_checkpoint(
            promoted
        )
    )

    assert (
        source_loaded.trainer
        .optimizer_step_count
        == 2
    )
    assert (
        before_loaded.trainer
        .optimizer_step_count
        == 2
    )
    assert (
        promoted_loaded.trainer
        .optimizer_step_count
        == 3
    )

    assert (
        before_loaded.progress
        == initial_progress
    )

    assert (
        promoted_loaded.progress
        == result.progress
    )

    assert result.progress == (
        PPOTrainingProgress(
            rollout_count=6,
            environment_step_count=21,
            completed_episode_count=2,
            cumulative_reward=-3.0,
        )
    )

    records = [
        json.loads(line)
        for line in audit.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    record_types = [
        record["type"]
        for record in records
    ]

    assert record_types == [
        "death_aware_session_started",
        "death_aware_attempt_completed",
        "death_aware_attempt_completed",
        "death_aware_attempt_promoted",
        "death_aware_session_completed",
    ]

    assert (
        records[-1]["data"]["promoted"]
        is True
    )


def test_second_death_respawn_inference_is_promoted(
    tmp_path,
) -> None:
    source, _ = create_source_checkpoint(
        tmp_path
    )

    def environment_factory(
        _: int,
    ) -> Environment:
        return SequenceEnvironment(
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

    promoted = tmp_path / "inferred.pt"

    result = (
        run_bounded_death_aware_ppo_session(
            source,
            environment_factory,
            agent_factory,
            session_config(
                attempt_count=1
            ),
            promoted_checkpoint_path=(
                promoted
            ),
        )
    )

    assert result.promoted

    accepted = result.accepted_attempt

    assert accepted is not None
    assert accepted.rollout.respawn_inferred
    assert not (
        accepted.rollout.respawn_detected
    )
    assert (
        accepted.rollout
        .respawn_signal_reason
        == "second_death_proves_respawn"
    )
    assert (
        accepted.rollout
        .post_respawn_reward
        == pytest.approx(-1.0)
    )

    # The second death is evidence only and remains outside the first
    # death's PPO batch and cumulative training reward.
    assert accepted.rollout.steps == 1
    assert (
        accepted.rollout.batch.rewards
        .tolist()
        == pytest.approx([-1.0])
    )
    assert (
        result.progress.cumulative_reward
        == pytest.approx(-3.0)
    )

    loaded = load_ppo_training_checkpoint(
        promoted
    )

    assert (
        loaded.trainer.optimizer_step_count
        == 3
    )


def test_no_accepted_attempt_writes_audit_without_promotion(
    tmp_path,
) -> None:
    source, initial_progress = (
        create_source_checkpoint(tmp_path)
    )

    def environment_factory(
        _: int,
    ) -> Environment:
        return SequenceEnvironment(
            [
                step_result(1),
            ]
        )

    promoted = tmp_path / "not-created.pt"
    audit = tmp_path / "rejected.jsonl"

    config = (
        DeathAwarePPOTrainingSessionConfig(
            attempt_count=2,
            step_config=(
                DeathAwarePPOConfig(
                    max_steps=1,
                    max_respawn_wait_steps=1,
                    updates_per_signal=1,
                    stop_on_kl=False,
                    seed=900,
                )
            ),
        )
    )

    result = (
        run_bounded_death_aware_ppo_session(
            source,
            environment_factory,
            agent_factory,
            config,
            promoted_checkpoint_path=(
                promoted
            ),
            audit_path=audit,
        )
    )

    assert not result.promoted
    assert result.accepted_attempt is None
    assert result.attempts_completed == 2
    assert result.progress == initial_progress
    assert not promoted.exists()
    assert audit.is_file()

    assert all(
        not attempt.accepted
        for attempt in result.attempts
    )

    records = [
        json.loads(line)
        for line in audit.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert records[-1]["type"] == (
        "death_aware_session_completed"
    )

    assert (
        records[-1]["data"]["promoted"]
        is False
    )


def test_existing_output_path_is_rejected(
    tmp_path,
) -> None:
    source, _ = create_source_checkpoint(
        tmp_path
    )

    existing = tmp_path / "existing.jsonl"
    existing.write_text(
        "existing",
        encoding="utf-8",
    )

    def environment_factory(
        _: int,
    ) -> Environment:
        return SequenceEnvironment(
            [step_result(1)]
        )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        run_bounded_death_aware_ppo_session(
            source,
            environment_factory,
            agent_factory,
            session_config(
                attempt_count=1
            ),
            audit_path=existing,
        )
