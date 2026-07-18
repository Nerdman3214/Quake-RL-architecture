"""Read-only Xonotic observation bridge."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from RL.actions.contracts import ActionCommand
from RL.engine.bridge.contracts import EngineBridge
from RL.engine.client import (
    X11Window,
    X11WindowCapture,
    preprocess_rgb_frame,
)
from RL.observations.contracts import Observation


class WindowCapture(Protocol):
    """Capture operations required by the observation bridge."""

    def find_window(self) -> X11Window:
        """Find the Xonotic game window."""

    def capture_rgb(
        self,
        window: X11Window,
    ) -> np.ndarray:
        """Capture one RGB frame from the game window."""


class XonoticObservationBridge(EngineBridge):
    """Read Xonotic frames without injecting controls."""

    def __init__(
        self,
        *,
        frame_width: int = 160,
        frame_height: int = 90,
        frame_stack: int = 4,
        capture_client: WindowCapture | None = None,
    ) -> None:
        if frame_width <= 0:
            raise ValueError(
                "frame width must be greater than zero"
            )

        if frame_height <= 0:
            raise ValueError(
                "frame height must be greater than zero"
            )

        if frame_stack <= 0:
            raise ValueError(
                "frame stack must be greater than zero"
            )

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_stack = frame_stack

        self._capture_client: WindowCapture = (
            capture_client
            if capture_client is not None
            else X11WindowCapture()
        )

        self._window: X11Window | None = None
        self._frames: np.ndarray | None = None
        self._next_tick = 0

    @property
    def connected(self) -> bool:
        """Return whether a game window has been discovered."""

        return self._window is not None

    @property
    def frame_shape(
        self,
    ) -> tuple[int, int, int, int]:
        """Return the stacked policy-frame shape."""

        return (
            self.frame_stack,
            3,
            self.frame_height,
            self.frame_width,
        )

    def connect(self) -> None:
        """Discover and retain the visible Xonotic window."""

        if self._window is None:
            self._window = (
                self._capture_client.find_window()
            )

    def _require_window(self) -> X11Window:
        if self._window is None:
            raise RuntimeError(
                "observation bridge is not connected"
            )

        return self._window

    def reset_match(self) -> Observation:
        """Reset local frame history and read a fresh frame.

        This method does not restart or otherwise control Xonotic.
        """

        self._require_window()

        self._frames = None
        self._next_tick = 0

        return self.read_observation()

    def send_action(
        self,
        action: ActionCommand,
    ) -> None:
        """Reject actions until a separate control bridge exists."""

        if not isinstance(action, ActionCommand):
            raise TypeError(
                "action must be an ActionCommand"
            )

        raise NotImplementedError(
            "action injection is not available in the "
            "read-only observation bridge"
        )

    def read_observation(self) -> Observation:
        """Capture, preprocess, stack, and return one observation."""

        window = self._require_window()

        raw_frame = self._capture_client.capture_rgb(
            window
        )

        policy_frame = preprocess_rgb_frame(
            raw_frame,
            width=self.frame_width,
            height=self.frame_height,
        )

        expected_single_shape = (
            3,
            self.frame_height,
            self.frame_width,
        )

        if policy_frame.shape != expected_single_shape:
            raise RuntimeError(
                "preprocessed frame has unexpected shape: "
                f"{policy_frame.shape!r}"
            )

        if self._frames is None:
            self._frames = np.repeat(
                policy_frame[np.newaxis, ...],
                self.frame_stack,
                axis=0,
            )
        else:
            self._frames = np.concatenate(
                (
                    self._frames[1:],
                    policy_frame[np.newaxis, ...],
                ),
                axis=0,
            )

        self._frames = np.ascontiguousarray(
            self._frames,
            dtype=np.float32,
        )

        observation = Observation(
            frame=self._frames.copy(),
            telemetry=None,
            tick=self._next_tick,
        )

        self._next_tick += 1

        return observation

    def close(self) -> None:
        """Release the window and local observation state."""

        self._window = None
        self._frames = None
        self._next_tick = 0
