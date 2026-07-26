"""Tests for hierarchical recurrent behavior cloning."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from RL.actions.contracts import DiscreteAction
from RL.agents.policies.hierarchical_recurrent import (
    HierarchicalPolicyOutput,
    HierarchicalRecurrentPolicy,
    TacticalIntent,
)
from RL.training.imitation.composite_sequence_dataset import (
    CompositeSequenceBatch,
)
from RL.training.imitation.hierarchical_behavior_cloning import (
    HierarchicalBehaviorCloningTrainer,
    HierarchicalClassWeights,
    hierarchical_behavior_cloning_loss,
    hierarchical_behavior_cloning_metrics,
)


def make_batch(
    *,
    batch_size: int = 1,
    sequence_length: int = 2,
) -> CompositeSequenceBatch:
    return CompositeSequenceBatch(
        frames=torch.rand(
            batch_size,
            sequence_length,
            4,
            3,
            90,
            160,
            dtype=torch.float32,
        ),
        previous_action_features=torch.rand(
            batch_size,
            sequence_length,
            8,
            dtype=torch.float32,
        ),
        forward_classes=torch.tensor(
            [
                [
                    index % 3
                    for index in range(
                        sequence_length
                    )
                ]
                for _ in range(batch_size)
            ],
            dtype=torch.long,
        ),
        strafe_classes=torch.tensor(
            [
                [
                    (index + 1) % 3
                    for index in range(
                        sequence_length
                    )
                ]
                for _ in range(batch_size)
            ],
            dtype=torch.long,
        ),
        mouse_deltas=torch.zeros(
            batch_size,
            sequence_length,
            2,
            dtype=torch.float32,
        ),
        fire_targets=torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.float32,
        ),
        jump_targets=torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.float32,
        ),
        weapon_classes=torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.long,
        ),
        duration_ticks=torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.long,
        ),
        legacy_action_indices=torch.full(
            (
                batch_size,
                sequence_length,
            ),
            int(DiscreteAction.NO_OP),
            dtype=torch.long,
        ),
        valid_mask=torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.bool,
        ),
        timestamps_ns=torch.arange(
            sequence_length,
            dtype=torch.long,
        ).unsqueeze(0).repeat(
            batch_size,
            1,
        ),
        mode_indices=torch.zeros(
            batch_size,
            dtype=torch.long,
        ),
        episode_ids=tuple(
            f"episode-{index}"
            for index in range(batch_size)
        ),
        start_step_indices=torch.zeros(
            batch_size,
            dtype=torch.long,
        ),
        source_episode_paths=tuple(
            Path(f"/tmp/episode-{index}.jsonl")
            for index in range(batch_size)
        ),
    )


def make_zero_output(
    batch: CompositeSequenceBatch,
) -> HierarchicalPolicyOutput:
    batch_size = int(batch.frames.shape[0])
    sequence_length = int(
        batch.frames.shape[1]
    )

    return HierarchicalPolicyOutput(
        intent_logits=torch.zeros(
            batch_size,
            sequence_length,
            len(tuple(TacticalIntent)),
        ),
        forward_logits=torch.zeros(
            batch_size,
            sequence_length,
            3,
        ),
        strafe_logits=torch.zeros(
            batch_size,
            sequence_length,
            3,
        ),
        mouse_deltas=torch.zeros(
            batch_size,
            sequence_length,
            2,
        ),
        fire_logits=torch.zeros(
            batch_size,
            sequence_length,
        ),
        jump_logits=torch.zeros(
            batch_size,
            sequence_length,
        ),
        weapon_logits=torch.zeros(
            batch_size,
            sequence_length,
            3,
        ),
        legacy_action_logits=torch.zeros(
            batch_size,
            sequence_length,
            len(tuple(DiscreteAction)),
        ),
        hidden_state=torch.zeros(
            2,
            batch_size,
            256,
        ),
    )


def clone_parameters(
    model: HierarchicalRecurrentPolicy,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter.detach().clone()
        for parameter in model.parameters()
    )


def parameters_changed(
    before: tuple[torch.Tensor, ...],
    model: HierarchicalRecurrentPolicy,
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


def test_multitask_loss_and_metrics_are_finite() -> None:
    batch = make_batch(
        batch_size=2,
        sequence_length=3,
    )
    output = make_zero_output(batch)

    losses = hierarchical_behavior_cloning_loss(
        output,
        batch,
    )
    metrics = (
        hierarchical_behavior_cloning_metrics(
            output,
            batch,
            losses=losses,
        )
    )

    assert torch.isfinite(losses.total)
    assert metrics.valid_step_count == 6
    assert metrics.mouse_mean_absolute_error == 0.0


def test_loss_rejects_batch_without_valid_steps() -> None:
    batch = make_batch()
    object.__setattr__(
        batch,
        "valid_mask",
        torch.zeros_like(
            batch.valid_mask
        ),
    )

    with pytest.raises(
        ValueError,
        match="at least one valid step",
    ):
        hierarchical_behavior_cloning_loss(
            make_zero_output(batch),
            batch,
        )


def test_trainer_updates_model_parameters() -> None:
    torch.manual_seed(101)

    model = HierarchicalRecurrentPolicy(
        visual_feature_dim=32,
        previous_action_dim=8,
        mode_embedding_dim=4,
        recurrent_input_dim=32,
        recurrent_hidden_dim=32,
        recurrent_layers=1,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    trainer = HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
    )

    before = clone_parameters(model)

    result = trainer.train_batch(
        make_batch()
    )

    assert result.optimizer_step
    assert result.optimizer_step_count == 1
    assert result.gradient_norm is not None
    assert result.gradient_norm >= 0.0
    assert parameters_changed(
        before,
        model,
    )


def test_evaluation_does_not_update_parameters() -> None:
    torch.manual_seed(202)

    model = HierarchicalRecurrentPolicy(
        visual_feature_dim=32,
        previous_action_dim=8,
        mode_embedding_dim=4,
        recurrent_input_dim=32,
        recurrent_hidden_dim=32,
        recurrent_layers=1,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    trainer = HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
    )

    before = clone_parameters(model)

    result = trainer.evaluate_batch(
        make_batch()
    )

    assert not result.optimizer_step
    assert result.gradient_norm is None
    assert not parameters_changed(
        before,
        model,
    )


def test_trainer_rejects_other_model_optimizer() -> None:
    model = HierarchicalRecurrentPolicy(
        visual_feature_dim=32,
        previous_action_dim=8,
        mode_embedding_dim=4,
        recurrent_input_dim=32,
        recurrent_hidden_dim=32,
        recurrent_layers=1,
    )
    other_model = HierarchicalRecurrentPolicy(
        visual_feature_dim=32,
        previous_action_dim=8,
        mode_embedding_dim=4,
        recurrent_input_dim=32,
        recurrent_hidden_dim=32,
        recurrent_layers=1,
    )

    optimizer = torch.optim.Adam(
        other_model.parameters(),
        lr=1e-3,
    )

    with pytest.raises(
        ValueError,
        match="exactly the model parameters",
    ):
        HierarchicalBehaviorCloningTrainer(
            model,
            optimizer,
        )


def test_trainer_accepts_restored_optimizer_count() -> None:
    model = HierarchicalRecurrentPolicy(
        visual_feature_dim=32,
        previous_action_dim=8,
        mode_embedding_dim=4,
        recurrent_input_dim=32,
        recurrent_hidden_dim=32,
        recurrent_layers=1,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    trainer = HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
        optimizer_step_count=9,
    )

    assert trainer.optimizer_step_count == 9

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        HierarchicalBehaviorCloningTrainer(
            model,
            torch.optim.Adam(
                model.parameters(),
                lr=1e-3,
            ),
            optimizer_step_count=-1,
        )


def test_hierarchical_public_exports() -> None:
    from RL.agents import (
        HierarchicalPolicyOutput as ExportedOutput,
    )
    from RL.agents import (
        HierarchicalRecurrentPolicy as ExportedPolicy,
    )
    from RL.agents import (
        TacticalIntent as ExportedIntent,
    )
    from RL.training.imitation import (
        HierarchicalBehaviorCloningTrainer
        as ExportedTrainer,
    )
    from RL.training.imitation import (
        HierarchicalLossWeights
        as ExportedWeights,
    )

    assert ExportedPolicy is (
        HierarchicalRecurrentPolicy
    )
    assert ExportedOutput is (
        HierarchicalPolicyOutput
    )
    assert ExportedIntent is TacticalIntent
    assert ExportedTrainer is (
        HierarchicalBehaviorCloningTrainer
    )
    assert ExportedWeights.__name__ == (
        "HierarchicalLossWeights"
    )



@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        float("inf"),
        float("nan"),
        True,
    ],
)
def test_class_weights_reject_invalid_values(
    value: object,
) -> None:
    with pytest.raises(ValueError):
        HierarchicalClassWeights(
            fire_positive=value,
        )


def test_class_weights_change_imbalanced_losses() -> None:
    batch = make_batch(
        batch_size=1,
        sequence_length=2,
    )

    batch.fire_targets[0, 0] = 1.0
    batch.fire_targets[0, 1] = 0.0
    batch.jump_targets[0, 0] = 1.0
    batch.jump_targets[0, 1] = 0.0

    output = make_zero_output(batch)

    baseline = hierarchical_behavior_cloning_loss(
        output,
        batch,
    )

    weighted = hierarchical_behavior_cloning_loss(
        output,
        batch,
        class_weights=HierarchicalClassWeights(
            fire_positive=4.0,
            jump_positive=5.0,
        ),
    )

    assert weighted.fire > baseline.fire
    assert weighted.jump > baseline.jump
    assert weighted.total > baseline.total


def test_weapon_class_weight_emphasizes_previous() -> None:
    batch = make_batch(
        batch_size=1,
        sequence_length=2,
    )

    batch.weapon_classes[0, 0] = 0
    batch.weapon_classes[0, 1] = 1

    output = make_zero_output(batch)

    output.weapon_logits[0, 0, 1] = 5.0
    output.weapon_logits[0, 1, 1] = 5.0

    baseline = hierarchical_behavior_cloning_loss(
        output,
        batch,
    )

    weighted = hierarchical_behavior_cloning_loss(
        output,
        batch,
        class_weights=HierarchicalClassWeights(
            weapon_previous=8.0,
            weapon_neutral=1.0,
            weapon_next=1.0,
        ),
    )

    assert weighted.weapon > baseline.weapon
    assert weighted.total > baseline.total


def test_trainer_stores_class_weights() -> None:
    model = HierarchicalRecurrentPolicy(
        visual_feature_dim=32,
        previous_action_dim=8,
        mode_embedding_dim=4,
        recurrent_input_dim=32,
        recurrent_hidden_dim=32,
        recurrent_layers=1,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    class_weights = HierarchicalClassWeights(
        fire_positive=2.0,
        jump_positive=3.0,
        weapon_previous=4.0,
        weapon_neutral=1.0,
        weapon_next=5.0,
    )

    trainer = HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
        class_weights=class_weights,
    )

    assert trainer.class_weights == class_weights
