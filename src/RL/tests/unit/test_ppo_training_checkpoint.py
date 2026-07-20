"""Tests for resumable PPO actor-critic checkpoints."""

from __future__ import annotations

import math

import pytest
import torch

from RL.agents import (
    VisualActorCriticNetwork,
)
from RL.training.ppo import (
    PPOHyperparameters,
    PPOTrainer,
    PPOTrainingProgress,
    RolloutBuffer,
    RolloutTransition,
    load_ppo_training_checkpoint,
    restore_ppo_checkpoint_rng_state,
    save_ppo_training_checkpoint,
)


def make_batch(
    model: VisualActorCriticNetwork,
):
    buffer = RolloutBuffer(
        max_steps=2
    )

    for index, reward in enumerate(
        (0.25, 1.0)
    ):
        frame = torch.rand(
            4,
            3,
            90,
            160,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            decision = model.act(
                frame.unsqueeze(0),
                deterministic=False,
            )

        buffer.append(
            RolloutTransition(
                frames=frame,
                action_index=int(
                    decision.action_indices.item()
                ),
                reward=reward,
                terminated=False,
                truncated=(
                    index == 1
                ),
                old_log_prob=float(
                    decision.log_probs.item()
                ),
                old_value=float(
                    decision.values.item()
                ),
            )
        )

    return buffer.finish(
        last_value=0.5,
    )


def make_trainer():
    model = VisualActorCriticNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    trainer = PPOTrainer(
        model,
        optimizer,
        hyperparameters=PPOHyperparameters(
            clip_epsilon=0.15,
            value_coefficient=0.6,
            entropy_coefficient=0.02,
            value_clip_epsilon=0.1,
        ),
        max_gradient_norm=0.7,
    )

    return trainer


def state_copy(
    model: VisualActorCriticNetwork,
):
    return {
        key: tensor.detach()
        .cpu()
        .clone()
        for key, tensor
        in model.state_dict().items()
    }


def test_progress_validates_values() -> None:
    progress = PPOTrainingProgress(
        rollout_count=2,
        environment_step_count=16,
        completed_episode_count=1,
        cumulative_reward=-0.5,
    )

    assert progress.to_record() == {
        "rollout_count": 2,
        "environment_step_count": 16,
        "completed_episode_count": 1,
        "cumulative_reward": -0.5,
    }

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        PPOTrainingProgress(
            rollout_count=-1
        )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        PPOTrainingProgress(
            cumulative_reward=float("nan")
        )


def test_round_trip_restores_training_state(
    tmp_path,
) -> None:
    torch.manual_seed(401)

    trainer = make_trainer()
    batch = make_batch(
        trainer.model
    )

    first_update = trainer.train_batch(
        batch
    )

    assert first_update.optimizer_step_count == 1

    progress = PPOTrainingProgress(
        rollout_count=1,
        environment_step_count=2,
        completed_episode_count=0,
        cumulative_reward=1.25,
    )

    expected_state = state_copy(
        trainer.model
    )

    path = tmp_path / "checkpoint.pt"

    save_ppo_training_checkpoint(
        path,
        trainer,
        progress=progress,
        policy_name="test-ppo",
        policy_version="v7",
        metadata={
            "mode": "synthetic",
            "trained": True,
        },
        created_at=(
            "2026-07-19T22:00:00+00:00"
        ),
    )

    loaded = (
        load_ppo_training_checkpoint(
            path
        )
    )

    assert loaded.policy_name == (
        "test-ppo"
    )

    assert loaded.policy_version == "v7"
    assert loaded.progress == progress

    assert dict(loaded.metadata) == {
        "mode": "synthetic",
        "trained": True,
    }

    assert (
        loaded.trainer.optimizer_step_count
        == 1
    )

    assert (
        loaded.trainer.max_gradient_norm
        == pytest.approx(0.7)
    )

    assert (
        loaded.trainer.hyperparameters
        .clip_epsilon
        == pytest.approx(0.15)
    )

    for key, expected in (
        expected_state.items()
    ):
        assert torch.equal(
            loaded.model.state_dict()[key],
            expected,
        )

    assert loaded.optimizer.state_dict()[
        "state"
    ]


def test_resumed_update_continues_counter(
    tmp_path,
) -> None:
    torch.manual_seed(402)

    trainer = make_trainer()
    batch = make_batch(
        trainer.model
    )

    trainer.train_batch(batch)

    path = tmp_path / "resume.pt"

    save_ppo_training_checkpoint(
        path,
        trainer,
    )

    loaded = (
        load_ppo_training_checkpoint(
            path
        )
    )

    before = state_copy(
        loaded.model
    )

    result = (
        loaded.trainer.train_batch(
            batch
        )
    )

    assert result.optimizer_step
    assert result.optimizer_step_count == 2

    assert any(
        not torch.equal(
            before[key],
            tensor.detach().cpu(),
        )
        for key, tensor
        in loaded.model.state_dict().items()
    )


def test_torch_rng_state_can_be_restored(
    tmp_path,
) -> None:
    trainer = make_trainer()

    torch.manual_seed(98765)

    path = tmp_path / "rng.pt"

    save_ppo_training_checkpoint(
        path,
        trainer,
    )

    expected = torch.rand(8)

    loaded = (
        load_ppo_training_checkpoint(
            path
        )
    )

    restore_ppo_checkpoint_rng_state(
        loaded,
        include_cuda=False,
    )

    actual = torch.rand(8)

    assert torch.equal(
        actual,
        expected,
    )


def test_action_mapping_corruption_is_rejected(
    tmp_path,
) -> None:
    trainer = make_trainer()

    path = tmp_path / "bad-actions.pt"

    save_ppo_training_checkpoint(
        path,
        trainer,
    )

    payload = torch.load(
        path,
        weights_only=True,
    )

    payload["action_names"] = [
        "BROKEN"
    ]

    torch.save(
        payload,
        path,
    )

    with pytest.raises(
        ValueError,
        match="action mapping",
    ):
        load_ppo_training_checkpoint(
            path
        )


def test_unsupported_optimizer_is_rejected(
    tmp_path,
) -> None:
    model = VisualActorCriticNetwork()

    trainer = PPOTrainer(
        model,
        torch.optim.SGD(
            model.parameters(),
            lr=1e-3,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Adam",
    ):
        save_ppo_training_checkpoint(
            tmp_path / "sgd.pt",
            trainer,
        )


def test_nonfinite_metadata_is_rejected(
    tmp_path,
) -> None:
    trainer = make_trainer()

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        save_ppo_training_checkpoint(
            tmp_path / "metadata.pt",
            trainer,
            metadata={
                "loss": float("inf"),
            },
        )


def test_save_is_atomic_without_temp_files(
    tmp_path,
) -> None:
    trainer = make_trainer()

    path = tmp_path / "atomic.pt"

    returned = (
        save_ppo_training_checkpoint(
            path,
            trainer,
        )
    )

    assert returned == path
    assert path.is_file()
    assert path.stat().st_size > 0

    assert list(
        tmp_path.glob(
            ".atomic.pt.*.tmp"
        )
    ) == []


def test_constructor_accepts_restored_count() -> None:
    model = VisualActorCriticNetwork()

    trainer = PPOTrainer(
        model,
        torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
        ),
        optimizer_step_count=9,
    )

    assert trainer.optimizer_step_count == 9

    invalid_model = (
        VisualActorCriticNetwork()
    )

    invalid_optimizer = (
        torch.optim.Adam(
            invalid_model.parameters(),
            lr=1e-4,
        )
    )

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        PPOTrainer(
            invalid_model,
            invalid_optimizer,
            optimizer_step_count=-1,
        )
