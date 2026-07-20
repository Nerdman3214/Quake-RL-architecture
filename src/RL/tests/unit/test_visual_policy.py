"""Tests for the visual policy network and inference agent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from RL.actions.contracts import (
    DiscreteAction,
)
from RL.agents import (
    NeuralPolicyAgent,
    VisualPolicyNetwork,
)
from RL.observations.contracts import Observation


FRAME_SHAPE = (
    4,
    3,
    90,
    160,
)


def make_frame() -> np.ndarray:
    return np.zeros(
        FRAME_SHAPE,
        dtype=np.float32,
    )


def make_observation(
    *,
    frame: object | None = None,
    tick: int = 0,
) -> Observation:
    return Observation(
        frame=(
            make_frame()
            if frame is None
            else frame
        ),
        telemetry=None,
        tick=tick,
    )


def make_biased_model(
    action: DiscreteAction,
) -> VisualPolicyNetwork:
    model = VisualPolicyNetwork()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        model.action_head.bias[
            int(action)
        ] = 5.0

    return model


def test_visual_policy_forward_shape() -> None:
    model = VisualPolicyNetwork()

    frames = torch.zeros(
        2,
        *FRAME_SHAPE,
        dtype=torch.float32,
    )

    logits = model(frames)

    assert logits.shape == (
        2,
        len(tuple(DiscreteAction)),
    )

    assert torch.isfinite(logits).all()
    assert model.frame_shape == FRAME_SHAPE
    assert model.encoded_shape == (
        64,
        7,
        16,
    )


def test_visual_policy_rejects_wrong_rank() -> None:
    model = VisualPolicyNetwork()

    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        model(
            torch.zeros(
                *FRAME_SHAPE,
                dtype=torch.float32,
            )
        )


def test_visual_policy_rejects_wrong_shape() -> None:
    model = VisualPolicyNetwork()

    with pytest.raises(
        ValueError,
        match="unexpected policy frame shape",
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


def test_visual_policy_rejects_integer_frames() -> None:
    model = VisualPolicyNetwork()

    with pytest.raises(
        TypeError,
        match="floating-point",
    ):
        model(
            torch.zeros(
                1,
                *FRAME_SHAPE,
                dtype=torch.uint8,
            )
        )


def test_visual_policy_rejects_nonfinite_frames() -> None:
    model = VisualPolicyNetwork()

    frames = torch.zeros(
        1,
        *FRAME_SHAPE,
        dtype=torch.float32,
    )

    frames[0, 0, 0, 0, 0] = float("nan")

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        model(frames)


def test_neural_agent_selects_highest_logit_action() -> None:
    model = make_biased_model(
        DiscreteAction.FIRE
    )

    agent = NeuralPolicyAgent(
        model,
        device="cpu",
        duration_ticks=2,
    )

    action = agent.act(
        make_observation()
    )

    assert action.action is DiscreteAction.FIRE
    assert action.duration_ticks == 2
    assert not model.training


def test_neural_agent_exposes_latest_scores() -> None:
    model = make_biased_model(
        DiscreteAction.TURN_RIGHT
    )

    agent = NeuralPolicyAgent(
        model,
        device="cpu",
    )

    agent.act(
        make_observation(tick=5)
    )

    assert agent.last_logits is not None
    assert len(agent.last_logits) == len(
        tuple(DiscreteAction)
    )

    assert agent.last_action_scores[
        "TURN_RIGHT"
    ] == 5.0

    assert max(
        agent.last_action_scores,
        key=agent.last_action_scores.get,
    ) == "TURN_RIGHT"


def test_neural_agent_rejects_missing_frame() -> None:
    agent = NeuralPolicyAgent(
        VisualPolicyNetwork()
    )

    observation = Observation(
        frame=None,
        telemetry=None,
        tick=0,
    )

    with pytest.raises(
        ValueError,
        match="must not be None",
    ):
        agent.act(observation)


def test_neural_agent_rejects_wrong_frame_shape() -> None:
    agent = NeuralPolicyAgent(
        VisualPolicyNetwork()
    )

    with pytest.raises(
        ValueError,
        match="unexpected observation frame shape",
    ):
        agent.act(
            make_observation(
                frame=np.zeros(
                    (4, 3, 84, 84),
                    dtype=np.float32,
                )
            )
        )


def test_neural_agent_rejects_nonfinite_frame() -> None:
    agent = NeuralPolicyAgent(
        VisualPolicyNetwork()
    )

    frame = make_frame()
    frame[0, 0, 0, 0] = np.inf

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        agent.act(
            make_observation(
                frame=frame
            )
        )


def test_neural_agent_rejects_invalid_duration() -> None:
    for invalid_value in (
        0,
        -1,
        True,
    ):
        with pytest.raises(
            ValueError,
            match="positive integer",
        ):
            NeuralPolicyAgent(
                VisualPolicyNetwork(),
                duration_ticks=invalid_value,
            )
