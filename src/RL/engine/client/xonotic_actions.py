"""Map engine-neutral actions to verified Xonotic X11 input."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Protocol

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.engine.client.x11_window import X11Window


class ActionInputController(Protocol):
    """Input operations required by the Xonotic action mapper."""

    def hold_key(
        self,
        window: X11Window,
        key: str,
        *,
        duration_seconds: float,
    ) -> None:
        """Hold and release a keyboard key."""

    def hold_mouse_button(
        self,
        window: X11Window,
        button: int,
        *,
        duration_seconds: float,
    ) -> None:
        """Hold and release a mouse button."""

    def move_mouse_relative(
        self,
        window: X11Window,
        *,
        delta_x: int,
        delta_y: int = 0,
    ) -> None:
        """Move the mouse by a bounded relative amount."""

    def scroll(
        self,
        window: X11Window,
        *,
        steps: int,
    ) -> None:
        """Send bounded mouse-wheel steps."""

    def release_all(
        self,
        window: X11Window | None = None,
    ) -> None:
        """Release any input still marked as held."""


class XonoticActionExecutor:
    """Execute one bounded ActionCommand for Xonotic."""

    _MOVEMENT_KEYS = {
        DiscreteAction.FORWARD: "w",
        DiscreteAction.BACKWARD: "s",
        DiscreteAction.STRAFE_LEFT: "a",
        DiscreteAction.STRAFE_RIGHT: "d",
    }

    def __init__(
        self,
        input_controller: ActionInputController,
        *,
        tick_seconds: float = 0.05,
        turn_pixels_per_tick: int = 1,
        max_duration_ticks: int = 20,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(tick_seconds):
            raise ValueError(
                "tick duration must be finite"
            )

        if tick_seconds <= 0:
            raise ValueError(
                "tick duration must be greater than zero"
            )

        if (
            isinstance(turn_pixels_per_tick, bool)
            or not isinstance(
                turn_pixels_per_tick,
                int,
            )
            or turn_pixels_per_tick <= 0
        ):
            raise ValueError(
                "turn pixels per tick must be a positive integer"
            )

        if (
            isinstance(max_duration_ticks, bool)
            or not isinstance(
                max_duration_ticks,
                int,
            )
            or max_duration_ticks <= 0
        ):
            raise ValueError(
                "maximum duration ticks must be a positive integer"
            )

        if not callable(sleeper):
            raise TypeError(
                "sleeper must be callable"
            )

        self.input_controller = input_controller
        self.tick_seconds = tick_seconds
        self.turn_pixels_per_tick = (
            turn_pixels_per_tick
        )
        self.max_duration_ticks = (
            max_duration_ticks
        )
        self.sleeper = sleeper

    def _validate_command(
        self,
        command: ActionCommand,
    ) -> None:
        if not isinstance(command, ActionCommand):
            raise TypeError(
                "command must be an ActionCommand"
            )

        if not isinstance(
            command.action,
            DiscreteAction,
        ):
            raise TypeError(
                "command action must be a DiscreteAction"
            )

        duration_ticks = command.duration_ticks

        if (
            isinstance(duration_ticks, bool)
            or not isinstance(duration_ticks, int)
        ):
            raise TypeError(
                "duration_ticks must be an integer"
            )

        if duration_ticks <= 0:
            raise ValueError(
                "duration_ticks must be greater than zero"
            )

        if duration_ticks > self.max_duration_ticks:
            raise ValueError(
                "duration_ticks exceeds the configured maximum"
            )

    def _duration_seconds(
        self,
        command: ActionCommand,
    ) -> float:
        return (
            command.duration_ticks
            * self.tick_seconds
        )

    def release_all(
        self,
        window: X11Window | None = None,
    ) -> None:
        """Release any input still held by the controller."""

        self.input_controller.release_all(window)

    def execute(
        self,
        window: X11Window,
        command: ActionCommand,
    ) -> None:
        """Execute exactly one validated action command."""

        if not isinstance(window, X11Window):
            raise TypeError(
                "window must be an X11Window instance"
            )

        self._validate_command(command)

        action = command.action
        duration_seconds = self._duration_seconds(
            command
        )

        if action == DiscreteAction.NO_OP:
            self.sleeper(duration_seconds)
            return

        movement_key = self._MOVEMENT_KEYS.get(
            action
        )

        if movement_key is not None:
            self.input_controller.hold_key(
                window,
                movement_key,
                duration_seconds=duration_seconds,
            )
            return

        if action == DiscreteAction.JUMP:
            self.input_controller.hold_key(
                window,
                "space",
                duration_seconds=duration_seconds,
            )
            return

        if action == DiscreteAction.FIRE:
            self.input_controller.hold_mouse_button(
                window,
                1,
                duration_seconds=duration_seconds,
            )
            return

        if action in (
            DiscreteAction.TURN_LEFT,
            DiscreteAction.TURN_RIGHT,
        ):
            direction = (
                -1
                if action
                == DiscreteAction.TURN_LEFT
                else 1
            )

            delta_x = (
                direction
                * self.turn_pixels_per_tick
                * command.duration_ticks
            )

            self.input_controller.move_mouse_relative(
                window,
                delta_x=delta_x,
                delta_y=0,
            )
            return

        if action in (
            DiscreteAction.NEXT_WEAPON,
            DiscreteAction.PREVIOUS_WEAPON,
        ):
            if command.duration_ticks != 1:
                raise ValueError(
                    "weapon-switch actions require "
                    "duration_ticks=1"
                )

            steps = (
                1
                if action
                == DiscreteAction.NEXT_WEAPON
                else -1
            )

            self.input_controller.scroll(
                window,
                steps=steps,
            )
            return

        raise ValueError(
            f"unsupported Xonotic action: {action!r}"
        )
