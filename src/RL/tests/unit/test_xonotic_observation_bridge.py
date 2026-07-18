"""Tests for the read-only Xonotic observation bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.config.schema import load_config
from RL.engine.bridge import XonoticObservationBridge
from RL.engine.client import X11Window


class FakeCapture:
    """Deterministic capture client for bridge tests."""

    def __init__(
        self,
        frames: list[np.ndarray],
    ) -> None:
        self.frames = frames
        self.find_calls = 0
        self.capture_calls = 0
        self.window = X11Window(
            window_id=100,
            title="Xonotic",
        )

    def find_window(self) -> X11Window:
        self.find_calls += 1
        return self.window

    def capture_rgb(
        self,
        window: X11Window,
    ) -> np.ndarray:
        if window != self.window:
            raise AssertionError(
                "bridge supplied an unexpected window"
            )

        if self.capture_calls >= len(self.frames):
            raise AssertionError(
                "test requested more frames than provided"
            )

        frame = self.frames[
            self.capture_calls
        ].copy()

        self.capture_calls += 1

        return frame


def solid_frame(value: int) -> np.ndarray:
    return np.full(
        (9, 16, 3),
        value,
        dtype=np.uint8,
    )


def test_constructor_rejects_invalid_dimensions() -> None:
    with pytest.raises(
        ValueError,
        match="height",
    ):
        XonoticObservationBridge(
            frame_height=0,
        )

    with pytest.raises(
        ValueError,
        match="stack",
    ):
        XonoticObservationBridge(
            frame_stack=0,
        )


def test_connect_discovers_window_once() -> None:
    capture = FakeCapture(
        [solid_frame(10)]
    )

    bridge = XonoticObservationBridge(
        capture_client=capture,
    )

    assert not bridge.connected

    bridge.connect()
    bridge.connect()

    assert bridge.connected
    assert capture.find_calls == 1


def test_read_requires_connection() -> None:
    bridge = XonoticObservationBridge(
        capture_client=FakeCapture(
            [solid_frame(10)]
        )
    )

    with pytest.raises(
        RuntimeError,
        match="not connected",
    ):
        bridge.read_observation()


def test_first_observation_repeats_initial_frame() -> None:
    capture = FakeCapture(
        [solid_frame(64)]
    )

    bridge = XonoticObservationBridge(
        frame_width=160,
        frame_height=90,
        frame_stack=4,
        capture_client=capture,
    )

    bridge.connect()
    observation = bridge.read_observation()

    assert isinstance(
        observation.frame,
        np.ndarray,
    )

    assert observation.frame.shape == (
        4,
        3,
        90,
        160,
    )

    assert observation.frame.dtype == np.float32
    assert observation.tick == 0
    assert observation.telemetry is None

    expected = 64.0 / 255.0

    assert np.allclose(
        observation.frame,
        expected,
    )

    assert observation.frame.flags[
        "C_CONTIGUOUS"
    ]


def test_later_observation_shifts_frame_stack() -> None:
    capture = FakeCapture(
        [
            solid_frame(10),
            solid_frame(200),
        ]
    )

    bridge = XonoticObservationBridge(
        capture_client=capture,
    )

    bridge.connect()

    first = bridge.read_observation()
    second = bridge.read_observation()

    assert first.tick == 0
    assert second.tick == 1

    assert np.allclose(
        second.frame[:3],
        10.0 / 255.0,
    )

    assert np.allclose(
        second.frame[3],
        200.0 / 255.0,
    )


def test_returned_frame_does_not_mutate_internal_stack() -> None:
    capture = FakeCapture(
        [
            solid_frame(20),
            solid_frame(40),
        ]
    )

    bridge = XonoticObservationBridge(
        capture_client=capture,
    )

    bridge.connect()

    first = bridge.read_observation()
    first.frame.fill(1.0)

    second = bridge.read_observation()

    assert np.allclose(
        second.frame[:3],
        20.0 / 255.0,
    )

    assert np.allclose(
        second.frame[3],
        40.0 / 255.0,
    )


def test_reset_match_resets_tick_and_history() -> None:
    capture = FakeCapture(
        [
            solid_frame(10),
            solid_frame(20),
            solid_frame(30),
        ]
    )

    bridge = XonoticObservationBridge(
        capture_client=capture,
    )

    bridge.connect()

    bridge.read_observation()
    second = bridge.read_observation()
    reset = bridge.reset_match()

    assert second.tick == 1
    assert reset.tick == 0

    assert np.allclose(
        reset.frame,
        30.0 / 255.0,
    )


def test_send_action_is_explicitly_unavailable() -> None:
    bridge = XonoticObservationBridge(
        capture_client=FakeCapture(
            [solid_frame(10)]
        )
    )

    command = ActionCommand(
        action=DiscreteAction.NO_OP,
    )

    with pytest.raises(
        NotImplementedError,
        match="not available",
    ):
        bridge.send_action(command)


def test_close_releases_connection_and_history() -> None:
    capture = FakeCapture(
        [solid_frame(10)]
    )

    bridge = XonoticObservationBridge(
        capture_client=capture,
    )

    bridge.connect()
    bridge.read_observation()
    bridge.close()

    assert not bridge.connected

    with pytest.raises(
        RuntimeError,
        match="not connected",
    ):
        bridge.read_observation()


def test_default_config_uses_native_rgb_shape() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "default.json"
    )

    config = load_config(config_path)

    assert (
        config.environment.observation_version
        == "v2"
    )

    assert config.environment.frame_width == 160
    assert config.environment.frame_height == 90
    assert config.environment.frame_stack == 4
