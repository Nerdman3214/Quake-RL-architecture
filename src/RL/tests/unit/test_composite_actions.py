"""Tests for simultaneous FPS control contracts."""

from __future__ import annotations

import pytest

from RL.actions import CompositeActionCommand


def test_default_composite_action_is_no_op() -> None:
    command = CompositeActionCommand()

    assert command.is_no_op
    assert command.to_record() == {
        "forward_axis": 0,
        "strafe_axis": 0,
        "turn_delta_x": 0.0,
        "look_delta_y": 0.0,
        "fire": False,
        "jump": False,
        "weapon_delta": 0,
        "duration_ticks": 1,
    }


def test_composite_action_preserves_combinations() -> None:
    command = CompositeActionCommand(
        forward_axis=1,
        strafe_axis=-1,
        turn_delta_x=4.5,
        look_delta_y=-2.0,
        fire=True,
        jump=True,
        weapon_delta=1,
        duration_ticks=2,
    )

    assert not command.is_no_op
    assert command.forward_axis == 1
    assert command.strafe_axis == -1
    assert command.fire
    assert command.jump


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("forward_axis", 2),
        ("strafe_axis", -2),
        ("turn_delta_x", float("inf")),
        ("look_delta_y", float("nan")),
        ("duration_ticks", 0),
    ],
)
def test_composite_action_rejects_invalid_values(
    field_name: str,
    value: object,
) -> None:
    arguments = {
        field_name: value,
    }

    with pytest.raises(
        (TypeError, ValueError),
    ):
        CompositeActionCommand(
            **arguments,
        )
