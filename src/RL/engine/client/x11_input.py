"""Fail-safe X11 input primitives for a focused game window."""

from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Callable

from RL.engine.client.x11_window import X11Window


class X11InputController:
    """Send tightly bounded input to a verified X11 window."""

    def __init__(
        self,
        *,
        xdotool_command: str = "xdotool",
        sleeper: Callable[[float], None] = time.sleep,
        max_hold_seconds: float = 1.0,
        max_mouse_delta: int = 64,
        max_scroll_steps: int = 5,
    ) -> None:
        if not xdotool_command:
            raise ValueError(
                "xdotool command must not be empty"
            )

        if (
            not math.isfinite(max_hold_seconds)
            or max_hold_seconds <= 0
        ):
            raise ValueError(
                "maximum hold duration must be positive"
            )

        if max_mouse_delta <= 0:
            raise ValueError(
                "maximum mouse delta must be positive"
            )

        if max_scroll_steps <= 0:
            raise ValueError(
                "maximum scroll steps must be positive"
            )

        self.xdotool_command = xdotool_command
        self.sleeper = sleeper
        self.max_hold_seconds = max_hold_seconds
        self.max_mouse_delta = max_mouse_delta
        self.max_scroll_steps = max_scroll_steps

        self._pressed_keys: set[str] = set()
        self._pressed_buttons: set[int] = set()

    @property
    def pressed_keys(self) -> tuple[str, ...]:
        """Return keys still awaiting a successful release."""

        return tuple(sorted(self._pressed_keys))

    @property
    def pressed_buttons(self) -> tuple[int, ...]:
        """Return mouse buttons still awaiting release."""

        return tuple(sorted(self._pressed_buttons))

    @staticmethod
    def _command_error(
        command: list[str],
        error: subprocess.CalledProcessError,
    ) -> RuntimeError:
        detail = str(error.stderr or "").strip()

        message = (
            f"command failed: {' '.join(command)}"
        )

        if detail:
            message += f": {detail}"

        return RuntimeError(message)

    def _run(
        self,
        *arguments: str,
    ) -> str:
        command = [
            self.xdotool_command,
            *arguments,
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "required command is unavailable: "
                f"{self.xdotool_command}"
            ) from error
        except subprocess.CalledProcessError as error:
            raise self._command_error(
                command,
                error,
            ) from error

        return result.stdout.strip()

    @staticmethod
    def _validate_window(
        window: X11Window,
    ) -> None:
        if not isinstance(window, X11Window):
            raise TypeError(
                "window must be an X11Window instance"
            )

    def _validate_duration(
        self,
        duration_seconds: float,
    ) -> None:
        if (
            not math.isfinite(duration_seconds)
            or duration_seconds < 0
        ):
            raise ValueError(
                "input duration must be finite and nonnegative"
            )

        if duration_seconds > self.max_hold_seconds:
            raise ValueError(
                "input duration exceeds the configured maximum"
            )

    def focus(
        self,
        window: X11Window,
    ) -> None:
        """Activate the target and verify it owns keyboard focus."""

        self._validate_window(window)

        self._run(
            "windowactivate",
            "--sync",
            str(window.window_id),
        )

        active_output = self._run(
            "getactivewindow"
        )

        try:
            active_window_id = int(active_output)
        except ValueError as error:
            raise RuntimeError(
                "xdotool returned an invalid active window id: "
                f"{active_output!r}"
            ) from error

        if active_window_id != window.window_id:
            raise RuntimeError(
                "Xonotic did not receive input focus"
            )

    def hold_key(
        self,
        window: X11Window,
        key: str,
        *,
        duration_seconds: float,
    ) -> None:
        """Hold one key and guarantee a release attempt."""

        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                "key must be a nonempty string"
            )

        self._validate_duration(duration_seconds)
        self.focus(window)

        normalized_key = key.strip()
        pressed = False

        try:
            self._run(
                "keydown",
                normalized_key,
            )

            self._pressed_keys.add(
                normalized_key
            )
            pressed = True

            self.sleeper(duration_seconds)
        finally:
            if pressed:
                self._run(
                    "keyup",
                    normalized_key,
                )

                self._pressed_keys.discard(
                    normalized_key
                )

    def hold_mouse_button(
        self,
        window: X11Window,
        button: int,
        *,
        duration_seconds: float,
    ) -> None:
        """Hold one mouse button and guarantee release."""

        if button <= 0:
            raise ValueError(
                "mouse button must be positive"
            )

        self._validate_duration(duration_seconds)
        self.focus(window)

        pressed = False

        try:
            self._run(
                "mousedown",
                str(button),
            )

            self._pressed_buttons.add(button)
            pressed = True

            self.sleeper(duration_seconds)
        finally:
            if pressed:
                self._run(
                    "mouseup",
                    str(button),
                )

                self._pressed_buttons.discard(
                    button
                )

    def move_mouse_relative(
        self,
        window: X11Window,
        *,
        delta_x: int,
        delta_y: int = 0,
    ) -> None:
        """Move the mouse by a bounded relative amount."""

        if (
            not isinstance(delta_x, int)
            or not isinstance(delta_y, int)
        ):
            raise TypeError(
                "mouse deltas must be integers"
            )

        if (
            abs(delta_x) > self.max_mouse_delta
            or abs(delta_y) > self.max_mouse_delta
        ):
            raise ValueError(
                "mouse delta exceeds the configured maximum"
            )

        self.focus(window)

        if delta_x == 0 and delta_y == 0:
            return

        self._run(
            "mousemove_relative",
            "--sync",
            "--",
            str(delta_x),
            str(delta_y),
        )

    def scroll(
        self,
        window: X11Window,
        *,
        steps: int,
    ) -> None:
        """Scroll using X11 buttons 4 and 5."""

        if not isinstance(steps, int):
            raise TypeError(
                "scroll steps must be an integer"
            )

        if steps == 0:
            return

        if abs(steps) > self.max_scroll_steps:
            raise ValueError(
                "scroll steps exceed the configured maximum"
            )

        self.focus(window)

        button = 4 if steps > 0 else 5

        for _ in range(abs(steps)):
            self._run(
                "click",
                str(button),
            )

    def release_all(
        self,
        window: X11Window | None = None,
    ) -> None:
        """Retry releases for any input still marked as held."""

        if window is not None:
            self.focus(window)

        errors: list[str] = []

        for key in tuple(self._pressed_keys):
            try:
                self._run(
                    "keyup",
                    key,
                )
            except RuntimeError as error:
                errors.append(str(error))
            else:
                self._pressed_keys.discard(key)

        for button in tuple(
            self._pressed_buttons
        ):
            try:
                self._run(
                    "mouseup",
                    str(button),
                )
            except RuntimeError as error:
                errors.append(str(error))
            else:
                self._pressed_buttons.discard(
                    button
                )

        if errors:
            raise RuntimeError(
                "one or more inputs could not be released: "
                + "; ".join(errors)
            )
