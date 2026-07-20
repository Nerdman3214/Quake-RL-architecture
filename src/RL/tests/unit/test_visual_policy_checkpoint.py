"""Tests for safe visual-policy checkpoint storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from RL.actions.contracts import (
    DiscreteAction,
)
from RL.agents import (
    NeuralPolicyAgent,
    VisualPolicyNetwork,
    load_visual_policy_checkpoint,
    save_visual_policy_checkpoint,
)
from RL.observations.contracts import Observation


def make_observation(
    model: VisualPolicyNetwork,
) -> Observation:
    return Observation(
        frame=np.zeros(
            model.frame_shape,
            dtype=np.float32,
        ),
        telemetry=None,
        tick=0,
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
        ] = 7.0

    return model


def read_payload(
    path: Path,
) -> dict[str, object]:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    assert isinstance(payload, dict)

    return payload


def test_checkpoint_round_trip_preserves_model(
    tmp_path: Path,
) -> None:
    torch.manual_seed(1234)

    model = VisualPolicyNetwork()

    frames = torch.rand(
        2,
        *model.frame_shape,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        expected_logits = model(frames)

    path = tmp_path / "policy.pt"

    returned_path = (
        save_visual_policy_checkpoint(
            model,
            path,
            policy_name="test-policy",
            policy_version="v3",
            created_at=(
                "2026-07-19T20:15:00+00:00"
            ),
            metadata={
                "curriculum_stage": (
                    "open_arena_1v1"
                ),
                "tags": [
                    "unit-test",
                    "inference",
                ],
                "evaluation_score": 1.25,
            },
        )
    )

    loaded = load_visual_policy_checkpoint(
        path,
        device="cpu",
    )

    with torch.inference_mode():
        actual_logits = loaded.model(frames)

    assert returned_path == path
    assert loaded.path == path
    assert loaded.policy_name == "test-policy"
    assert loaded.policy_version == "v3"
    assert loaded.created_at == (
        "2026-07-19T20:15:00+00:00"
    )
    assert loaded.metadata[
        "curriculum_stage"
    ] == "open_arena_1v1"
    assert loaded.metadata["tags"] == [
        "unit-test",
        "inference",
    ]
    assert not loaded.model.training

    torch.testing.assert_close(
        actual_logits,
        expected_logits,
    )

    assert list(
        tmp_path.glob(".policy.pt.*.tmp")
    ) == []


def test_loaded_checkpoint_runs_neural_agent(
    tmp_path: Path,
) -> None:
    model = make_biased_model(
        DiscreteAction.FIRE
    )

    path = tmp_path / "fire-policy.pt"

    save_visual_policy_checkpoint(
        model,
        path,
        policy_name="fire-policy",
        policy_version="v1",
    )

    loaded = load_visual_policy_checkpoint(
        path
    )

    agent = NeuralPolicyAgent(
        loaded.model,
        device="cpu",
        policy_name=loaded.policy_name,
        policy_version=loaded.policy_version,
    )

    action = agent.act(
        make_observation(
            loaded.model
        )
    )

    assert action.action is DiscreteAction.FIRE
    assert agent.policy_name == "fire-policy"
    assert agent.policy_version == "v1"


def test_checkpoint_rejects_non_json_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.pt"

    with pytest.raises(
        TypeError,
        match="JSON-compatible",
    ):
        save_visual_policy_checkpoint(
            VisualPolicyNetwork(),
            path,
            policy_name="policy",
            policy_version="v1",
            metadata={
                "invalid": object(),
            },
        )

    assert not path.exists()


def test_checkpoint_rejects_unknown_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "version.pt"

    save_visual_policy_checkpoint(
        VisualPolicyNetwork(),
        path,
        policy_name="policy",
        policy_version="v1",
    )

    payload = read_payload(path)
    payload["format_version"] = 999
    torch.save(payload, path)

    with pytest.raises(
        ValueError,
        match="format version",
    ):
        load_visual_policy_checkpoint(path)


def test_checkpoint_rejects_action_mapping_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "actions.pt"

    save_visual_policy_checkpoint(
        VisualPolicyNetwork(),
        path,
        policy_name="policy",
        policy_version="v1",
    )

    payload = read_payload(path)

    action_names = list(
        payload["action_names"]
    )

    action_names[0] = "UNEXPECTED_ACTION"
    payload["action_names"] = action_names

    torch.save(payload, path)

    with pytest.raises(
        ValueError,
        match="action mapping",
    ):
        load_visual_policy_checkpoint(path)


def test_checkpoint_rejects_incompatible_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.pt"

    save_visual_policy_checkpoint(
        VisualPolicyNetwork(),
        path,
        policy_name="policy",
        policy_version="v1",
    )

    payload = read_payload(path)

    state_dict = dict(
        payload["state_dict"]
    )

    state_dict.pop("action_head.bias")
    payload["state_dict"] = state_dict

    torch.save(payload, path)

    with pytest.raises(
        ValueError,
        match="state_dict is incompatible",
    ):
        load_visual_policy_checkpoint(path)


def test_checkpoint_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
    ):
        load_visual_policy_checkpoint(
            tmp_path / "missing.pt"
        )
