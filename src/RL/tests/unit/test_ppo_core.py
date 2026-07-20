"""Tests for rollout, GAE, PPO loss, and trainer contracts."""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import Categorical

from RL.actions.contracts import DiscreteAction
from RL.agents import VisualActorCriticNetwork
from RL.training.ppo import (
    PPOHyperparameters,
    PPOTrainer,
    RolloutBuffer,
    RolloutTransition,
    compute_gae,
    ppo_loss,
)


def make_frame() -> torch.Tensor:
    return torch.rand(
        4,
        3,
        90,
        160,
        dtype=torch.float32,
    )


def make_rollout_batch(
    *,
    model: (
        VisualActorCriticNetwork | None
    ) = None,
):
    network = (
        model
        if model is not None
        else VisualActorCriticNetwork()
    )

    buffer = RolloutBuffer(
        max_steps=3
    )

    rewards = (
        0.25,
        -0.10,
        1.00,
    )

    for index, reward in enumerate(rewards):
        frame = make_frame()

        with torch.inference_mode():
            result = network.act(
                frame.unsqueeze(0),
                deterministic=False,
            )

        buffer.append(
            RolloutTransition(
                frames=frame,
                action_index=int(
                    result.action_indices.item()
                ),
                reward=reward,
                terminated=False,
                truncated=(
                    index == 2
                ),
                old_log_prob=float(
                    result.log_probs.item()
                ),
                old_value=float(
                    result.values.item()
                ),
            )
        )

    with torch.inference_mode():
        _, final_value = network(
            make_frame().unsqueeze(0)
        )

    return buffer.finish(
        last_value=float(
            final_value.item()
        ),
    )


def clone_parameters(
    model: VisualActorCriticNetwork,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter.detach().clone()
        for parameter in model.parameters()
    )


def parameters_changed(
    before: tuple[torch.Tensor, ...],
    model: VisualActorCriticNetwork,
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


def test_gae_termination_disables_bootstrap() -> None:
    advantages, returns = compute_gae(
        torch.tensor(
            [1.0],
            dtype=torch.float32,
        ),
        torch.tensor(
            [0.25],
            dtype=torch.float32,
        ),
        torch.tensor(
            [True],
            dtype=torch.bool,
        ),
        last_value=100.0,
        gamma=0.9,
        gae_lambda=1.0,
    )

    assert advantages.tolist() == (
        pytest.approx([0.75])
    )

    assert returns.tolist() == (
        pytest.approx([1.0])
    )


def test_gae_truncation_can_bootstrap() -> None:
    advantages, returns = compute_gae(
        torch.tensor(
            [1.0],
            dtype=torch.float32,
        ),
        torch.tensor(
            [0.25],
            dtype=torch.float32,
        ),
        torch.tensor(
            [False],
            dtype=torch.bool,
        ),
        last_value=2.0,
        gamma=0.9,
        gae_lambda=1.0,
    )

    assert advantages.tolist() == (
        pytest.approx([2.55])
    )

    assert returns.tolist() == (
        pytest.approx([2.8])
    )


def test_rollout_buffer_builds_ordered_batch() -> None:
    buffer = RolloutBuffer(
        max_steps=2
    )

    buffer.append(
        RolloutTransition(
            frames=make_frame(),
            action_index=int(
                DiscreteAction.FORWARD
            ),
            reward=0.5,
            terminated=False,
            truncated=False,
            old_log_prob=-1.0,
            old_value=0.1,
        )
    )

    buffer.append(
        RolloutTransition(
            frames=make_frame(),
            action_index=int(
                DiscreteAction.FIRE
            ),
            reward=1.0,
            terminated=True,
            truncated=False,
            old_log_prob=-0.5,
            old_value=0.2,
        )
    )

    batch = buffer.finish(
        last_value=50.0,
    )

    assert batch.frames.shape == (
        2,
        4,
        3,
        90,
        160,
    )

    assert batch.action_indices.tolist() == [
        int(DiscreteAction.FORWARD),
        int(DiscreteAction.FIRE),
    ]

    assert batch.terminated.tolist() == [
        False,
        True,
    ]

    assert batch.truncated.tolist() == [
        False,
        False,
    ]

    assert torch.isfinite(
        batch.advantages
    ).all()

    assert torch.isfinite(
        batch.returns
    ).all()


def test_rollout_rejects_append_after_terminal() -> None:
    buffer = RolloutBuffer(
        max_steps=4
    )

    buffer.append(
        RolloutTransition(
            frames=make_frame(),
            action_index=int(
                DiscreteAction.NO_OP
            ),
            reward=0.0,
            terminated=True,
            truncated=False,
            old_log_prob=-1.0,
            old_value=0.0,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="already closed",
    ):
        buffer.append(
            RolloutTransition(
                frames=make_frame(),
                action_index=int(
                    DiscreteAction.FORWARD
                ),
                reward=0.0,
                terminated=False,
                truncated=False,
                old_log_prob=-1.0,
                old_value=0.0,
            )
        )


def test_rollout_cannot_finish_twice() -> None:
    buffer = RolloutBuffer(
        max_steps=1
    )

    buffer.append(
        RolloutTransition(
            frames=make_frame(),
            action_index=int(
                DiscreteAction.JUMP
            ),
            reward=0.0,
            terminated=False,
            truncated=True,
            old_log_prob=-1.0,
            old_value=0.0,
        )
    )

    buffer.finish(
        last_value=0.0
    )

    with pytest.raises(
        RuntimeError,
        match="already finished",
    ):
        buffer.finish(
            last_value=0.0
        )


def test_ppo_loss_is_finite() -> None:
    model = VisualActorCriticNetwork()
    batch = make_rollout_batch(
        model=model
    )

    logits, values = model(
        batch.frames
    )

    losses = ppo_loss(
        logits,
        values,
        batch,
    )

    assert torch.isfinite(
        losses.total_loss
    )

    assert losses.value_loss >= 0.0
    assert losses.entropy >= 0.0
    assert 0.0 <= float(
        losses.clip_fraction
    ) <= 1.0


def test_ppo_ratio_is_one_before_update() -> None:
    model = VisualActorCriticNetwork()

    frames = torch.stack(
        [
            make_frame(),
            make_frame(),
        ]
    )

    with torch.inference_mode():
        logits, values = model(frames)

        distribution = Categorical(
            logits=logits
        )

        actions = distribution.sample()

        old_log_probs = (
            distribution.log_prob(actions)
        )

    buffer = RolloutBuffer(
        max_steps=2
    )

    for index in range(2):
        buffer.append(
            RolloutTransition(
                frames=frames[index],
                action_index=int(
                    actions[index].item()
                ),
                reward=float(index),
                terminated=False,
                truncated=(
                    index == 1
                ),
                old_log_prob=float(
                    old_log_probs[index].item()
                ),
                old_value=float(
                    values[index].item()
                ),
            )
        )

    batch = buffer.finish(
        last_value=0.0,
    )

    current_logits, current_values = model(
        batch.frames
    )

    losses = ppo_loss(
        current_logits,
        current_values,
        batch,
    )

    assert float(
        losses.approximate_kl
        .detach()
        .item()
    ) == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert float(
        losses.clip_fraction
        .detach()
        .item()
    ) == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_ppo_trainer_updates_parameters() -> None:
    torch.manual_seed(77)

    model = VisualActorCriticNetwork()

    batch = make_rollout_batch(
        model=model
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    trainer = PPOTrainer(
        model,
        optimizer,
        max_gradient_norm=0.5,
    )

    before = clone_parameters(model)

    result = trainer.train_batch(
        batch
    )

    assert result.optimizer_step
    assert result.optimizer_step_count == 1
    assert result.gradient_norm is not None
    assert math.isfinite(
        result.gradient_norm
    )
    assert parameters_changed(
        before,
        model,
    )


def test_ppo_evaluation_does_not_update() -> None:
    model = VisualActorCriticNetwork()

    batch = make_rollout_batch(
        model=model
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    trainer = PPOTrainer(
        model,
        optimizer,
    )

    before = clone_parameters(model)

    result = trainer.evaluate_batch(
        batch
    )

    assert not result.optimizer_step
    assert result.optimizer_step_count == 0
    assert result.gradient_norm is None

    assert not parameters_changed(
        before,
        model,
    )


def test_ppo_trainer_rejects_foreign_optimizer() -> None:
    model = VisualActorCriticNetwork()

    other_model = (
        VisualActorCriticNetwork()
    )

    optimizer = torch.optim.Adam(
        other_model.parameters(),
        lr=1e-4,
    )

    with pytest.raises(
        ValueError,
        match="another model",
    ):
        PPOTrainer(
            model,
            optimizer,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0.0,
        -1.0,
        float("inf"),
        True,
    ],
)
def test_ppo_rejects_invalid_gradient_limit(
    invalid_value: object,
) -> None:
    model = VisualActorCriticNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        PPOTrainer(
            model,
            optimizer,
            max_gradient_norm=(
                invalid_value
            ),
        )


def test_ppo_hyperparameters_validate_ranges() -> None:
    with pytest.raises(
        ValueError,
        match="clip_epsilon",
    ):
        PPOHyperparameters(
            clip_epsilon=0.0
        )

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        PPOHyperparameters(
            entropy_coefficient=-0.1
        )
