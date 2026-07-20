"""Tests for finite, auditable PPO training sessions."""

from __future__ import annotations

import json
from types import MethodType

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
)
from RL.training.ppo import (
    PPOBatchResult,
    PPOMetrics,
    PPOTrainer,
    PPOTrainingProgress,
    PPOTrainingSessionConfig,
    load_ppo_training_checkpoint,
    run_bounded_ppo_training_session,
)


def observation(
    tick: int,
) -> Observation:
    return Observation(
        frame=np.full(
            (4, 3, 90, 160),
            tick / 20.0,
            dtype=np.float32,
        ),
        telemetry=None,
        tick=tick,
    )


class SyntheticEnvironment(Environment):
    def __init__(
        self,
        rewards: tuple[float, ...],
    ) -> None:
        self.rewards = rewards
        self.index = 0
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
    ):
        self.index = 0

        return (
            observation(0),
            {"seed": seed},
        )

    def step(self, action):
        reward = self.rewards[
            self.index
        ]

        self.index += 1

        return StepResult(
            observation=observation(
                self.index
            ),
            reward=reward,
            terminated=False,
            truncated=False,
            info={},
        )

    def close(self) -> None:
        self.closed = True


def make_components(
    *,
    deterministic: bool = True,
):
    model = VisualActorCriticNetwork()

    agent = ActorCriticPolicyAgent(
        model,
        deterministic=deterministic,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    trainer = PPOTrainer(
        model,
        optimizer,
    )

    return model, agent, trainer


def parameter_copy(model):
    return tuple(
        parameter.detach().clone()
        for parameter in model.parameters()
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
        for old, new in zip(
            before,
            model.parameters(),
            strict=True,
        )
    )


def environment_factory(
    _: int,
) -> Environment:
    return SyntheticEnvironment(
        (0.0, 1.0)
    )


def test_evaluation_only_session_preserves_weights() -> None:
    model, agent, trainer = (
        make_components()
    )

    before = parameter_copy(model)

    result = (
        run_bounded_ppo_training_session(
            environment_factory,
            agent,
            trainer,
            PPOTrainingSessionConfig(
                rollout_count=2,
                max_steps_per_rollout=2,
                training_enabled=False,
                updates_per_rollout=0,
                seed=20,
            ),
        )
    )

    assert result.completed_rollouts == 2
    assert result.environment_steps == 4
    assert result.optimizer_operations == 0
    assert result.progress.rollout_count == 2
    assert (
        result.progress
        .environment_step_count
        == 4
    )

    assert not parameters_changed(
        before,
        model,
    )

    assert all(
        not operation.result.optimizer_step
        for rollout in result.rollouts
        for operation in rollout.operations
    )


def test_training_session_updates_weights() -> None:
    torch.manual_seed(800)

    model, agent, trainer = (
        make_components(
            deterministic=False
        )
    )

    before = parameter_copy(model)

    result = (
        run_bounded_ppo_training_session(
            environment_factory,
            agent,
            trainer,
            PPOTrainingSessionConfig(
                rollout_count=2,
                max_steps_per_rollout=2,
                training_enabled=True,
                updates_per_rollout=1,
                stop_on_kl=False,
            ),
        )
    )

    assert result.optimizer_operations == 2
    assert trainer.optimizer_step_count == 2

    assert parameters_changed(
        before,
        model,
    )


def test_session_writes_checkpoints_and_audit(
    tmp_path,
) -> None:
    model, agent, trainer = (
        make_components()
    )

    before_path = (
        tmp_path / "before.pt"
    )

    after_path = (
        tmp_path / "after.pt"
    )

    audit_path = (
        tmp_path / "session.jsonl"
    )

    initial_progress = (
        PPOTrainingProgress(
            rollout_count=3,
            environment_step_count=20,
            completed_episode_count=1,
            cumulative_reward=-2.0,
        )
    )

    result = (
        run_bounded_ppo_training_session(
            environment_factory,
            agent,
            trainer,
            PPOTrainingSessionConfig(
                rollout_count=1,
                max_steps_per_rollout=2,
            ),
            progress=initial_progress,
            before_checkpoint_path=(
                before_path
            ),
            after_checkpoint_path=(
                after_path
            ),
            audit_path=audit_path,
            checkpoint_metadata={
                "purpose": "unit-test",
            },
        )
    )

    assert before_path.is_file()
    assert after_path.is_file()
    assert audit_path.is_file()

    before_checkpoint = (
        load_ppo_training_checkpoint(
            before_path
        )
    )

    after_checkpoint = (
        load_ppo_training_checkpoint(
            after_path
        )
    )

    assert (
        before_checkpoint.progress
        == initial_progress
    )

    assert (
        after_checkpoint.progress
        == result.progress
    )

    records = [
        json.loads(line)
        for line in audit_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    record_types = [
        record["type"]
        for record in records
    ]

    assert record_types[0] == (
        "session_started"
    )

    assert "ppo_operation" in record_types

    assert record_types[-1] == (
        "session_completed"
    )


def test_kl_limit_stops_session_early() -> None:
    _, agent, trainer = (
        make_components()
    )

    def high_kl_train_batch(
        self,
        batch,
    ):
        return PPOBatchResult(
            metrics=PPOMetrics(
                total_loss=0.1,
                policy_loss=0.1,
                value_loss=0.0,
                entropy=1.0,
                approximate_kl=0.5,
                clip_fraction=0.0,
                explained_variance=0.0,
                sample_count=int(
                    batch.frames.shape[0]
                ),
            ),
            optimizer_step=True,
            optimizer_step_count=1,
            gradient_norm=0.1,
        )

    trainer.train_batch = MethodType(
        high_kl_train_batch,
        trainer,
    )

    result = (
        run_bounded_ppo_training_session(
            environment_factory,
            agent,
            trainer,
            PPOTrainingSessionConfig(
                rollout_count=4,
                max_steps_per_rollout=2,
                training_enabled=True,
                updates_per_rollout=3,
                max_absolute_kl=0.05,
                stop_on_kl=True,
            ),
        )
    )

    assert result.stopped_early
    assert result.stop_reason == (
        "absolute_kl_limit_exceeded"
    )
    assert result.completed_rollouts == 1
    assert len(
        result.rollouts[0].operations
    ) == 1


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

    with pytest.raises(
        ValueError,
        match="share",
    ):
        run_bounded_ppo_training_session(
            environment_factory,
            agent,
            trainer,
            PPOTrainingSessionConfig(
                rollout_count=1,
                max_steps_per_rollout=1,
            ),
        )


def test_existing_audit_path_is_rejected(
    tmp_path,
) -> None:
    _, agent, trainer = (
        make_components()
    )

    audit_path = (
        tmp_path / "existing.jsonl"
    )

    audit_path.write_text(
        "existing",
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        run_bounded_ppo_training_session(
            environment_factory,
            agent,
            trainer,
            PPOTrainingSessionConfig(
                rollout_count=1,
                max_steps_per_rollout=1,
            ),
            audit_path=audit_path,
        )


def test_invalid_training_mode_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="at least one update",
    ):
        PPOTrainingSessionConfig(
            rollout_count=1,
            max_steps_per_rollout=1,
            training_enabled=True,
            updates_per_rollout=0,
        )

    with pytest.raises(
        ValueError,
        match="cannot",
    ):
        PPOTrainingSessionConfig(
            rollout_count=1,
            max_steps_per_rollout=1,
            training_enabled=False,
            updates_per_rollout=1,
        )
