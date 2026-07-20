"""Tests for behavior-cloning objectives and batch trainer."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from RL.actions.contracts import DiscreteAction
from RL.agents import VisualPolicyNetwork
from RL.training.imitation import (
    BehaviorCloningTrainer,
    DemonstrationBatch,
    behavior_cloning_loss,
    behavior_cloning_metrics,
)


FRAME_SHAPE = (
    4,
    3,
    90,
    160,
)


def make_batch(
    actions: list[DiscreteAction],
) -> DemonstrationBatch:
    batch_size = len(actions)

    return DemonstrationBatch(
        frames=torch.rand(
            batch_size,
            *FRAME_SHAPE,
            dtype=torch.float32,
        ),
        action_indices=torch.tensor(
            [
                int(action)
                for action in actions
            ],
            dtype=torch.long,
        ),
        duration_ticks=torch.ones(
            batch_size,
            dtype=torch.long,
        ),
        episode_ids=tuple(
            f"episode-{index}"
            for index in range(batch_size)
        ),
        step_indices=torch.arange(
            batch_size,
            dtype=torch.long,
        ),
        source_episode_paths=tuple(
            Path(f"/tmp/episode-{index}.jsonl")
            for index in range(batch_size)
        ),
        source_frame_paths=tuple(
            Path(f"/tmp/frame-{index}.npy")
            for index in range(batch_size)
        ),
    )


def clone_parameters(
    model: VisualPolicyNetwork,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter.detach().clone()
        for parameter in model.parameters()
    )


def parameters_changed(
    before: tuple[torch.Tensor, ...],
    model: VisualPolicyNetwork,
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


def test_uniform_logits_have_expected_loss() -> None:
    logits = torch.zeros(
        2,
        len(tuple(DiscreteAction)),
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            int(DiscreteAction.NO_OP),
            int(DiscreteAction.FIRE),
        ],
        dtype=torch.long,
    )

    loss = behavior_cloning_loss(
        logits,
        targets,
    )

    metrics = behavior_cloning_metrics(
        logits,
        targets,
        loss=loss,
    )

    assert float(loss) == pytest.approx(
        math.log(
            len(tuple(DiscreteAction))
        )
    )

    assert metrics.sample_count == 2
    assert metrics.correct_count == 1
    assert metrics.accuracy == 0.5
    assert metrics.mean_confidence == (
        pytest.approx(
            1.0
            / len(tuple(DiscreteAction))
        )
    )


def test_loss_accepts_class_weights() -> None:
    logits = torch.zeros(
        2,
        len(tuple(DiscreteAction)),
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            int(DiscreteAction.FORWARD),
            int(DiscreteAction.FIRE),
        ],
        dtype=torch.long,
    )

    weights = torch.ones(
        len(tuple(DiscreteAction)),
        dtype=torch.float32,
    )

    weights[
        int(DiscreteAction.FIRE)
    ] = 3.0

    loss = behavior_cloning_loss(
        logits,
        targets,
        class_weights=weights,
    )

    assert torch.isfinite(loss)


def test_loss_rejects_unknown_action_index() -> None:
    logits = torch.zeros(
        1,
        len(tuple(DiscreteAction)),
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [len(tuple(DiscreteAction))],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="unknown action",
    ):
        behavior_cloning_loss(
            logits,
            targets,
        )


def test_train_batch_updates_parameters() -> None:
    torch.manual_seed(101)

    model = VisualPolicyNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    trainer = BehaviorCloningTrainer(
        model,
        optimizer,
        device="cpu",
        max_gradient_norm=1.0,
    )

    before = clone_parameters(model)

    result = trainer.train_batch(
        make_batch(
            [
                DiscreteAction.FORWARD,
                DiscreteAction.FIRE,
            ]
        )
    )

    assert result.optimizer_step
    assert result.optimizer_step_count == 1
    assert result.gradient_norm is not None
    assert result.gradient_norm >= 0.0
    assert parameters_changed(
        before,
        model,
    )


def test_evaluate_batch_does_not_update_parameters() -> None:
    torch.manual_seed(202)

    model = VisualPolicyNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    trainer = BehaviorCloningTrainer(
        model,
        optimizer,
        device="cpu",
    )

    before = clone_parameters(model)

    result = trainer.evaluate_batch(
        make_batch(
            [
                DiscreteAction.TURN_LEFT,
                DiscreteAction.TURN_RIGHT,
            ]
        )
    )

    assert not result.optimizer_step
    assert result.optimizer_step_count == 0
    assert result.gradient_norm is None
    assert not parameters_changed(
        before,
        model,
    )


def test_trainer_counts_explicit_steps() -> None:
    model = VisualPolicyNetwork()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-4,
    )

    trainer = BehaviorCloningTrainer(
        model,
        optimizer,
    )

    batch = make_batch(
        [DiscreteAction.JUMP]
    )

    first = trainer.train_batch(batch)
    second = trainer.train_batch(batch)

    assert first.optimizer_step_count == 1
    assert second.optimizer_step_count == 2
    assert trainer.optimizer_step_count == 2


def test_trainer_rejects_other_model_optimizer() -> None:
    model = VisualPolicyNetwork()
    other_model = VisualPolicyNetwork()

    optimizer = torch.optim.Adam(
        other_model.parameters(),
        lr=1e-3,
    )

    with pytest.raises(
        ValueError,
        match="another model",
    ):
        BehaviorCloningTrainer(
            model,
            optimizer,
        )


def test_trainer_rejects_invalid_gradient_limit() -> None:
    model = VisualPolicyNetwork()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    for invalid_value in (
        0.0,
        -1.0,
        float("inf"),
        True,
    ):
        with pytest.raises(
            ValueError,
            match="finite and positive",
        ):
            BehaviorCloningTrainer(
                model,
                optimizer,
                max_gradient_norm=(
                    invalid_value
                ),
            )
