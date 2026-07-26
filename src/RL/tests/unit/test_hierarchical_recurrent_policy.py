"""Tests for the hierarchical recurrent composite policy."""

from __future__ import annotations

import pytest
import torch

from RL.actions.contracts import DiscreteAction
from RL.agents.policies.hierarchical_recurrent import (
    HierarchicalRecurrentPolicy,
    TacticalIntent,
)


def make_inputs(
    *,
    batch_size: int = 2,
    sequence_length: int = 3,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    frames = torch.rand(
        batch_size,
        sequence_length,
        4,
        3,
        90,
        160,
        dtype=torch.float32,
    )

    previous_actions = torch.rand(
        batch_size,
        sequence_length,
        8,
        dtype=torch.float32,
    )

    mode_indices = torch.tensor(
        [
            index % 5
            for index in range(batch_size)
        ],
        dtype=torch.long,
    )

    return (
        frames,
        previous_actions,
        mode_indices,
    )


def test_policy_returns_all_composite_heads() -> None:
    torch.manual_seed(101)

    model = HierarchicalRecurrentPolicy()
    model.eval()

    frames, previous, modes = make_inputs()

    with torch.no_grad():
        output = model(
            frames,
            previous,
            modes,
        )

    assert output.intent_logits.shape == (
        2,
        3,
        len(tuple(TacticalIntent)),
    )
    assert output.forward_logits.shape == (
        2,
        3,
        3,
    )
    assert output.strafe_logits.shape == (
        2,
        3,
        3,
    )
    assert output.mouse_deltas.shape == (
        2,
        3,
        2,
    )
    assert output.fire_logits.shape == (
        2,
        3,
    )
    assert output.jump_logits.shape == (
        2,
        3,
    )
    assert output.weapon_logits.shape == (
        2,
        3,
        3,
    )
    assert output.legacy_action_logits.shape == (
        2,
        3,
        len(tuple(DiscreteAction)),
    )
    assert output.hidden_state.shape == (
        2,
        2,
        256,
    )


def test_policy_can_continue_from_hidden_state() -> None:
    torch.manual_seed(202)

    model = HierarchicalRecurrentPolicy()
    model.eval()

    frames, previous, modes = make_inputs(
        sequence_length=2
    )

    with torch.no_grad():
        first = model(
            frames,
            previous,
            modes,
        )

        second = model(
            frames[:, :1],
            previous[:, :1],
            modes,
            first.hidden_state,
        )

    assert second.hidden_state.shape == (
        2,
        2,
        256,
    )
    assert second.forward_logits.shape == (
        2,
        1,
        3,
    )


def test_policy_backward_reaches_parameters() -> None:
    torch.manual_seed(303)

    model = HierarchicalRecurrentPolicy()

    frames, previous, modes = make_inputs(
        batch_size=1,
        sequence_length=2,
    )

    output = model(
        frames,
        previous,
        modes,
    )

    loss = (
        output.forward_logits.square().mean()
        + output.strafe_logits.square().mean()
        + output.mouse_deltas.square().mean()
        + output.fire_logits.square().mean()
        + output.jump_logits.square().mean()
        + output.weapon_logits.square().mean()
        + output.legacy_action_logits.square().mean()
    )

    loss.backward()

    assert any(
        parameter.grad is not None
        and bool(
            torch.isfinite(
                parameter.grad
            ).all().item()
        )
        for parameter in model.parameters()
    )


def test_policy_rejects_wrong_frame_shape() -> None:
    model = HierarchicalRecurrentPolicy()

    frames, previous, modes = make_inputs()

    with pytest.raises(
        ValueError,
        match="unexpected policy frame shape",
    ):
        model(
            frames[:, :, :, :, :, :-1],
            previous,
            modes,
        )


def test_policy_rejects_unknown_mode() -> None:
    model = HierarchicalRecurrentPolicy()

    frames, previous, modes = make_inputs()
    modes[0] = 5

    with pytest.raises(
        ValueError,
        match="unknown mode",
    ):
        model(
            frames,
            previous,
            modes,
        )



def test_policy_fuses_telemetry_inputs() -> None:
    frames, previous, modes = make_inputs()

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

    telemetry = torch.rand(
        frames.shape[0],
        frames.shape[1],
        8,
        dtype=torch.float32,
    )
    weapons = torch.zeros(
        frames.shape[0],
        frames.shape[1],
        dtype=torch.long,
    )

    output = model(
        frames,
        previous,
        modes,
        telemetry_features=telemetry,
        telemetry_weapon_indices=weapons,
    )

    assert output.forward_logits.shape[:2] == (
        frames.shape[0],
        frames.shape[1],
    )

    output.forward_logits.sum().backward()

    assert (
        model.telemetry_encoder[1]
        .weight.grad
        is not None
    )
    assert (
        model.telemetry_weapon_embedding
        .weight.grad
        is not None
    )
