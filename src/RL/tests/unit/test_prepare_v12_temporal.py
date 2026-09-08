"""Checks for frozen-alignment labels before temporal artifact creation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from RL.visual_bc.prepare_v12_temporal import (
    action_from_interval,
    build_labeled_records,
)


def _telemetry(
    sample_time: float,
    *,
    forward: float = 0.0,
    strafe: float = 0.0,
    jump: int = 0,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> dict:
    return {
        "sample_time": sample_time,
        "input_available": 1,
        "input_move_forward": forward,
        "input_move_strafe": strafe,
        "input_jump": jump,
        "view_yaw": yaw,
        "view_pitch": pitch,
    }


def test_action_preserves_raw_axes_thresholds_and_wrapped_view_delta() -> None:
    action = action_from_interval(
        [
            _telemetry(10.02, forward=360, strafe=1, jump=1, yaw=179, pitch=5),
            _telemetry(10.07, forward=-360, strafe=-1, jump=0, yaw=-179, pitch=2),
        ]
    )

    assert action == {
        "forward_axis_mean": 0.0,
        "strafe_axis_mean": 0.0,
        "forward_positive_fraction": 0.5,
        "forward_negative_fraction": 0.5,
        "strafe_positive_fraction": 0.0,
        "strafe_negative_fraction": 0.0,
        "jump_fraction": 0.5,
        "attack_fraction": None,
        "secondary_attack_fraction": None,
        "crouch_fraction": None,
        "use_fraction": None,
        "view_yaw_delta_degrees": 2.0,
        "view_pitch_delta_degrees": -3.0,
    }


def test_single_sample_matches_historical_zero_angle_labels() -> None:
    action = action_from_interval(
        [_telemetry(10.02, forward=360, strafe=-360, jump=-1, yaw=123, pitch=-12)]
    )

    assert action["forward_axis_mean"] == 360.0
    assert action["strafe_axis_mean"] == -360.0
    assert action["forward_positive_fraction"] == 1.0
    assert action["strafe_negative_fraction"] == 1.0
    assert action["jump_fraction"] == 1.0
    assert action["view_yaw_delta_degrees"] == 0.0
    assert action["view_pitch_delta_degrees"] == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_available", 0),
        ("input_move_forward", float("nan")),
        ("input_move_strafe", float("inf")),
        ("view_yaw", None),
        ("view_pitch", "invalid"),
        ("input_jump", 0.5),
    ],
)
def test_action_refuses_invalid_required_input(field: str, value: object) -> None:
    row = _telemetry(10.02)
    row[field] = value

    with pytest.raises(ValueError):
        action_from_interval([row])


@pytest.mark.parametrize(
    "field",
    [
        "input_available",
        "input_move_forward",
        "input_move_strafe",
        "input_jump",
        "view_yaw",
        "view_pitch",
    ],
)
def test_action_refuses_missing_required_input(field: str) -> None:
    row = _telemetry(10.02)
    del row[field]

    with pytest.raises(ValueError):
        action_from_interval([row])


@pytest.mark.parametrize(
    "field",
    [
        "input_attack",
        "input_fire",
        "input_primary",
        "input_attack2",
        "input_secondary",
        "input_crouch",
        "input_use",
    ],
)
def test_action_refuses_silent_changes_to_uncaptured_controls(field: str) -> None:
    row = _telemetry(10.02)
    row[field] = 0

    with pytest.raises(ValueError):
        action_from_interval([row])


def test_empty_action_interval_cannot_borrow_a_future_sample() -> None:
    with pytest.raises(ValueError):
        action_from_interval([])
