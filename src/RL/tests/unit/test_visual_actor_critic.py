"""Tests for the visual actor-critic network."""

from __future__ import annotations

import pytest
import torch

from RL.actions.contracts import DiscreteAction
from RL.agents import VisualActorCriticNetwork


def make_frames(
    batch_size: int = 2,
) -> torch.Tensor:
    return torch.rand(
        batch_size,
        4,
        3,
        90,
        160,
        dtype=torch.float32,
    )


def test_actor_critic_output_shapes() -> None:
    model = VisualActorCriticNetwork()

    logits, values = model(
        make_frames(2)
    )

    assert logits.shape == (
        2,
        len(tuple(DiscreteAction)),
    )

    assert values.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()
    assert model.frame_shape == (
        4,
        3,
        90,
        160,
    )


def test_actor_critic_backward_is_finite() -> None:
    model = VisualActorCriticNetwork()

    logits, values = model(
        make_frames(2)
    )

    loss = (
        logits.mean()
        + values.pow(2).mean()
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(
        torch.isfinite(
            gradient
        ).all()
        for gradient in gradients
    )


def test_deterministic_action_uses_policy_argmax() -> None:
    model = VisualActorCriticNetwork()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        model.policy_head.bias[
            int(DiscreteAction.FIRE)
        ] = 5.0

        model.value_head.bias[0] = 2.5

    result = model.act(
        make_frames(1),
        deterministic=True,
    )

    assert result.action_indices.tolist() == [
        int(DiscreteAction.FIRE)
    ]

    assert result.values.tolist() == (
        pytest.approx([2.5])
    )

    assert result.log_probs.shape == (1,)
    assert result.entropies.shape == (1,)


def test_stochastic_action_is_valid() -> None:
    torch.manual_seed(55)

    model = VisualActorCriticNetwork()

    result = model.act(
        make_frames(4),
        deterministic=False,
    )

    assert result.action_indices.shape == (
        4,
    )

    assert bool(
        (
            result.action_indices >= 0
        ).all()
    )

    assert bool(
        (
            result.action_indices
            < len(tuple(DiscreteAction))
        ).all()
    )

    assert torch.isfinite(
        result.log_probs
    ).all()


def test_actor_critic_rejects_wrong_shape() -> None:
    model = VisualActorCriticNetwork()

    with pytest.raises(
        ValueError,
        match="unexpected shape",
    ):
        model(
            torch.zeros(
                1,
                4,
                3,
                84,
                84,
                dtype=torch.float32,
            )
        )


def test_actor_critic_rejects_nonfinite_frames() -> None:
    model = VisualActorCriticNetwork()

    frames = make_frames(1)
    frames[0, 0, 0, 0, 0] = float("nan")

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        model(frames)
