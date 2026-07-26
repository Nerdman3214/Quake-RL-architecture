"""Tests for hierarchical behavior-cloning checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from RL.actions.contracts import DiscreteAction
from RL.agents import HierarchicalRecurrentPolicy
from RL.training.imitation import (
    CompositeSequenceBatch,
    HierarchicalBehaviorCloningTrainer,
    HierarchicalClassWeights,
    HierarchicalLossWeights,
)
from RL.training.imitation.hierarchical_checkpoint import (
    HIERARCHICAL_CHECKPOINT_FORMAT_VERSION,
    LoadedHierarchicalCheckpoint,
    load_hierarchical_checkpoint,
    save_hierarchical_checkpoint,
)


def make_model() -> HierarchicalRecurrentPolicy:
    return HierarchicalRecurrentPolicy(
        visual_feature_dim=32,
        previous_action_dim=8,
        mode_embedding_dim=4,
        recurrent_input_dim=32,
        recurrent_hidden_dim=32,
        recurrent_layers=1,
    )


def make_batch() -> CompositeSequenceBatch:
    return CompositeSequenceBatch(
        frames=torch.rand(
            1,
            1,
            4,
            3,
            90,
            160,
            dtype=torch.float32,
        ),
        previous_action_features=torch.zeros(
            1,
            1,
            8,
            dtype=torch.float32,
        ),
        forward_classes=torch.ones(
            1,
            1,
            dtype=torch.long,
        ),
        strafe_classes=torch.ones(
            1,
            1,
            dtype=torch.long,
        ),
        mouse_deltas=torch.zeros(
            1,
            1,
            2,
            dtype=torch.float32,
        ),
        fire_targets=torch.zeros(
            1,
            1,
            dtype=torch.float32,
        ),
        jump_targets=torch.zeros(
            1,
            1,
            dtype=torch.float32,
        ),
        weapon_classes=torch.ones(
            1,
            1,
            dtype=torch.long,
        ),
        duration_ticks=torch.ones(
            1,
            1,
            dtype=torch.long,
        ),
        legacy_action_indices=torch.full(
            (1, 1),
            int(DiscreteAction.NO_OP),
            dtype=torch.long,
        ),
        valid_mask=torch.ones(
            1,
            1,
            dtype=torch.bool,
        ),
        timestamps_ns=torch.zeros(
            1,
            1,
            dtype=torch.long,
        ),
        mode_indices=torch.zeros(
            1,
            dtype=torch.long,
        ),
        episode_ids=("checkpoint-episode",),
        start_step_indices=torch.zeros(
            1,
            dtype=torch.long,
        ),
        source_episode_paths=(
            Path("/tmp/checkpoint-episode.jsonl"),
        ),
    )


def assert_nested_equal(
    left: object,
    right: object,
) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return

    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left) == set(right)

        for key in left:
            assert_nested_equal(
                left[key],
                right[key],
            )

        return

    if isinstance(left, (list, tuple)):
        assert isinstance(
            right,
            type(left),
        )
        assert len(left) == len(right)

        for left_item, right_item in zip(
            left,
            right,
            strict=True,
        ):
            assert_nested_equal(
                left_item,
                right_item,
            )

        return

    assert left == right


def test_checkpoint_round_trip_and_resume(
    tmp_path: Path,
) -> None:
    torch.manual_seed(101)

    model = make_model()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )
    weights = HierarchicalLossWeights(
        mouse=0.02,
        legacy_action=0.1,
    )
    trainer = HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
        weights=weights,
        max_gradient_norm=0.75,
        optimizer_step_count=7,
    )

    # Initialize Adam state before saving.
    penalty = sum(
        parameter.square().mean()
        for parameter in model.parameters()
    )
    penalty.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    original_model_state = {
        name: tensor.detach().clone()
        for name, tensor
        in model.state_dict().items()
    }
    original_optimizer_state = (
        optimizer.state_dict()
    )

    checkpoint_path = (
        tmp_path / "hierarchical.pt"
    )

    saved_path = save_hierarchical_checkpoint(
        checkpoint_path,
        trainer=trainer,
        policy_name="hierarchical-bc",
        policy_version="v1",
        dataset_metadata={
            "sequence_length": 8,
            "stride": 4,
            "episodes": ["match-7"],
        },
        metadata={
            "purpose": "unit-test",
            "candidate_playable": False,
        },
    )

    assert saved_path == checkpoint_path
    assert checkpoint_path.is_file()
    assert not list(
        tmp_path.glob(
            ".hierarchical.pt.*.tmp"
        )
    )

    loaded = load_hierarchical_checkpoint(
        checkpoint_path
    )

    assert isinstance(
        loaded,
        LoadedHierarchicalCheckpoint,
    )
    assert loaded.policy_name == (
        "hierarchical-bc"
    )
    assert loaded.policy_version == "v1"
    assert loaded.trainer.optimizer_step_count == 7
    assert loaded.trainer.max_gradient_norm == 0.75
    assert loaded.trainer.weights == weights
    assert loaded.dataset_metadata[
        "sequence_length"
    ] == 8
    assert loaded.metadata[
        "candidate_playable"
    ] is False

    for name, tensor in (
        loaded.model.state_dict().items()
    ):
        assert torch.equal(
            tensor,
            original_model_state[name],
        )

    assert_nested_equal(
        loaded.optimizer.state_dict(),
        original_optimizer_state,
    )

    result = loaded.trainer.train_batch(
        make_batch()
    )

    assert result.optimizer_step
    assert result.optimizer_step_count == 8


def test_checkpoint_rejects_unsupported_version(
    tmp_path: Path,
) -> None:
    model = make_model()
    optimizer = torch.optim.Adam(
        model.parameters()
    )
    trainer = HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
    )
    path = tmp_path / "version.pt"

    save_hierarchical_checkpoint(
        path,
        trainer=trainer,
        policy_name="policy",
        policy_version="v1",
        dataset_metadata={},
    )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    payload["format_version"] = (
        HIERARCHICAL_CHECKPOINT_FORMAT_VERSION
        + 1
    )
    torch.save(payload, path)

    with pytest.raises(
        ValueError,
        match="unsupported",
    ):
        load_hierarchical_checkpoint(path)


def test_checkpoint_rejects_non_adam_optimizer(
    tmp_path: Path,
) -> None:
    model = make_model()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
    )
    trainer = HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
    )

    with pytest.raises(
        ValueError,
        match="only torch.optim.Adam",
    ):
        save_hierarchical_checkpoint(
            tmp_path / "sgd.pt",
            trainer=trainer,
            policy_name="policy",
            policy_version="v1",
            dataset_metadata={},
        )


def test_checkpoint_rejects_non_json_metadata(
    tmp_path: Path,
) -> None:
    model = make_model()
    optimizer = torch.optim.Adam(
        model.parameters()
    )
    trainer = HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
    )

    with pytest.raises(
        ValueError,
        match="unsupported value",
    ):
        save_hierarchical_checkpoint(
            tmp_path / "metadata.pt",
            trainer=trainer,
            policy_name="policy",
            policy_version="v1",
            dataset_metadata={
                "bad": object(),
            },
        )


def test_checkpoint_public_exports_and_missing_file(
    tmp_path: Path,
) -> None:
    from RL.training.imitation import (
        LoadedHierarchicalCheckpoint
        as ExportedLoadedCheckpoint,
    )
    from RL.training.imitation import (
        load_hierarchical_checkpoint
        as exported_load,
    )
    from RL.training.imitation import (
        save_hierarchical_checkpoint
        as exported_save,
    )

    assert ExportedLoadedCheckpoint is (
        LoadedHierarchicalCheckpoint
    )
    assert exported_load is (
        load_hierarchical_checkpoint
    )
    assert exported_save is (
        save_hierarchical_checkpoint
    )

    with pytest.raises(FileNotFoundError):
        load_hierarchical_checkpoint(
            tmp_path / "missing.pt"
        )



def test_checkpoint_round_trips_class_weights(
    tmp_path: Path,
) -> None:
    model = make_model()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    class_weights = HierarchicalClassWeights(
        fire_positive=2.5,
        jump_positive=4.0,
        weapon_previous=6.0,
        weapon_neutral=1.0,
        weapon_next=5.0,
    )

    trainer = HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
        class_weights=class_weights,
    )

    path = tmp_path / "class-weights-v2.pt"

    save_hierarchical_checkpoint(
        path,
        trainer=trainer,
        policy_name="weighted-policy",
        policy_version="v2",
        dataset_metadata={},
    )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    assert payload["format_version"] == (
        HIERARCHICAL_CHECKPOINT_FORMAT_VERSION
    )
    assert payload["trainer_config"][
        "class_weights"
    ] == {
        "fire_positive": 2.5,
        "jump_positive": 4.0,
        "weapon_previous": 6.0,
        "weapon_neutral": 1.0,
        "weapon_next": 5.0,
    }

    loaded = load_hierarchical_checkpoint(path)

    assert loaded.trainer.class_weights == (
        class_weights
    )


def test_checkpoint_loads_legacy_v1_with_defaults(
    tmp_path: Path,
) -> None:
    model = make_model()
    optimizer = torch.optim.Adam(
        model.parameters(),
    )

    trainer = HierarchicalBehaviorCloningTrainer(
        model,
        optimizer,
        class_weights=HierarchicalClassWeights(
            fire_positive=3.0,
            jump_positive=4.0,
            weapon_previous=5.0,
            weapon_neutral=1.0,
            weapon_next=6.0,
        ),
    )

    path = tmp_path / "legacy-v1.pt"

    save_hierarchical_checkpoint(
        path,
        trainer=trainer,
        policy_name="legacy-policy",
        policy_version="v1",
        dataset_metadata={},
    )

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    payload["format_version"] = 1
    payload["trainer_config"].pop(
        "class_weights"
    )

    torch.save(payload, path)

    loaded = load_hierarchical_checkpoint(path)

    assert loaded.trainer.class_weights == (
        HierarchicalClassWeights()
    )



def test_checkpoint_round_trips_telemetry_model(
    tmp_path: Path,
) -> None:
    from RL.training.imitation.hierarchical_checkpoint import (
        load_hierarchical_checkpoint,
        save_hierarchical_checkpoint,
    )

    model = HierarchicalRecurrentPolicy(
        visual_feature_dim=32,
        previous_action_dim=8,
        mode_embedding_dim=4,
        telemetry_feature_count=8,
        telemetry_feature_dim=8,
        telemetry_weapon_count=64,
        telemetry_weapon_embedding_dim=4,
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

    path = save_hierarchical_checkpoint(
        tmp_path / "telemetry.pt",
        trainer=trainer,
        policy_name="telemetry-test",
        policy_version="v1",
        dataset_metadata={
            "telemetry_schema_version": 1,
            "telemetry_feature_count": 8,
            "telemetry_weapon_count": 64,
        },
    )

    loaded = load_hierarchical_checkpoint(
        path
    )

    assert loaded.model.telemetry_enabled
    assert (
        loaded.model.telemetry_feature_count
        == 8
    )
    assert (
        loaded.model.telemetry_weapon_count
        == 64
    )
