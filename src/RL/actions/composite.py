"""Composite FPS control contracts.

This module supplements the existing single DiscreteAction contract.
It does not replace ActionCommand or change the current environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _axis_value(
    value: object,
    *,
    field_name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in (-1, 0, 1)
    ):
        raise ValueError(
            f"{field_name} must be -1, 0, or 1"
        )

    return value


def _finite_float(
    value: object,
    *,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"{field_name} must be finite"
        )

    return float(value)


@dataclass(frozen=True)
class CompositeActionCommand:
    """One tick of simultaneous FPS controls.

    Axis conventions:
    - forward_axis: -1 backward, 0 neutral, 1 forward
    - strafe_axis: -1 left, 0 neutral, 1 right
    - turn_delta_x: negative left, positive right
    - look_delta_y: raw vertical relative-motion delta
    - weapon_delta: positive next, negative previous
    """

    forward_axis: int = 0
    strafe_axis: int = 0
    turn_delta_x: float = 0.0
    look_delta_y: float = 0.0
    fire: bool = False
    jump: bool = False
    weapon_delta: int = 0
    duration_ticks: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "forward_axis",
            _axis_value(
                self.forward_axis,
                field_name="forward_axis",
            ),
        )

        object.__setattr__(
            self,
            "strafe_axis",
            _axis_value(
                self.strafe_axis,
                field_name="strafe_axis",
            ),
        )

        object.__setattr__(
            self,
            "turn_delta_x",
            _finite_float(
                self.turn_delta_x,
                field_name="turn_delta_x",
            ),
        )

        object.__setattr__(
            self,
            "look_delta_y",
            _finite_float(
                self.look_delta_y,
                field_name="look_delta_y",
            ),
        )

        if not isinstance(self.fire, bool):
            raise TypeError(
                "fire must be bool"
            )

        if not isinstance(self.jump, bool):
            raise TypeError(
                "jump must be bool"
            )

        if (
            isinstance(self.weapon_delta, bool)
            or not isinstance(
                self.weapon_delta,
                int,
            )
        ):
            raise TypeError(
                "weapon_delta must be an integer"
            )

        if (
            isinstance(self.duration_ticks, bool)
            or not isinstance(
                self.duration_ticks,
                int,
            )
            or self.duration_ticks <= 0
        ):
            raise ValueError(
                "duration_ticks must be a positive integer"
            )

    @property
    def is_no_op(self) -> bool:
        """Return whether this command contains no active control."""

        return (
            self.forward_axis == 0
            and self.strafe_axis == 0
            and self.turn_delta_x == 0.0
            and self.look_delta_y == 0.0
            and not self.fire
            and not self.jump
            and self.weapon_delta == 0
        )

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible control record."""

        return {
            "forward_axis": self.forward_axis,
            "strafe_axis": self.strafe_axis,
            "turn_delta_x": self.turn_delta_x,
            "look_delta_y": self.look_delta_y,
            "fire": self.fire,
            "jump": self.jump,
            "weapon_delta": self.weapon_delta,
            "duration_ticks": self.duration_ticks,
        }
