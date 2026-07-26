"""Tests for direct X11 window capture and preprocessing."""

from __future__ import annotations

import subprocess
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from RL.engine.client import (
    X11Window,
    X11WindowCapture,
    preprocess_rgb_frame,
)


def make_png_bytes() -> bytes:
    image = Image.new(
        "RGB",
        (8, 4),
        color=(12, 34, 56),
    )

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_window_selector_uses_hexadecimal() -> None:
    window = X11Window(
        window_id=60817410,
        title="Xonotic",
    )

    assert window.selector == "0x3a00002"


def test_find_window_uses_last_visible_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)

        if command[1] == "search":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="100\n200\n",
                stderr="",
            )

        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Xonotic\n",
            stderr="",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    capture = X11WindowCapture()
    window = capture.find_window()

    assert window.window_id == 200
    assert window.title == "Xonotic"
    assert calls[1][-1] == "200"


class FakeMSSCapture:
    def __init__(self) -> None:
        self.monitors: list[
            dict[str, int]
        ] = []
        self.closed = False

    def grab(
        self,
        monitor: dict[str, int],
    ) -> np.ndarray:
        self.monitors.append(
            dict(monitor)
        )

        frame = np.empty(
            (2, 3, 4),
            dtype=np.uint8,
        )
        frame[:, :, 0] = 10
        frame[:, :, 1] = 20
        frame[:, :, 2] = 30
        frame[:, :, 3] = 255
        return frame

    def close(self) -> None:
        self.closed = True


def test_capture_rgb_uses_persistent_mss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    instances: list[FakeMSSCapture] = []

    def factory() -> FakeMSSCapture:
        capture = FakeMSSCapture()
        instances.append(capture)
        return capture

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)

        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "WINDOW=200\n"
                "X=10\n"
                "Y=20\n"
                "WIDTH=3\n"
                "HEIGHT=2\n"
                "SCREEN=0\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    capture = X11WindowCapture(
        mss_factory=factory,
    )
    window = X11Window(
        window_id=200,
        title="Xonotic",
    )

    first = capture.capture_rgb(window)
    second = capture.capture_rgb(window)

    assert len(instances) == 1
    assert commands == [
        [
            "xdotool",
            "getwindowgeometry",
            "--shell",
            "200",
        ]
    ]
    assert instances[0].monitors == [
        {
            "left": 10,
            "top": 20,
            "width": 3,
            "height": 2,
        },
        {
            "left": 10,
            "top": 20,
            "width": 3,
            "height": 2,
        },
    ]

    for frame in (first, second):
        assert frame.shape == (2, 3, 3)
        assert frame.dtype == np.uint8
        assert frame.flags["C_CONTIGUOUS"]
        assert frame[0, 0].tolist() == [
            30,
            20,
            10,
        ]

    capture.close()
    capture.close()

    assert instances[0].closed

    reopened = capture.capture_rgb(window)

    assert reopened.shape == (2, 3, 3)
    assert len(instances) == 2
    assert len(commands) == 2

    capture.close()

    assert instances[1].closed


def test_imagemagick_backend_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_png_bytes()
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)

        return subprocess.CompletedProcess(
            command,
            0,
            stdout=payload,
            stderr=b"",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    capture = X11WindowCapture(
        backend="imagemagick",
    )

    frame = capture.capture_rgb(
        X11Window(
            window_id=200,
            title="Xonotic",
        )
    )

    assert frame.shape == (4, 8, 3)
    assert frame.dtype == np.uint8
    assert frame[0, 0].tolist() == [
        12,
        34,
        56,
    ]
    assert commands[0] == [
        "import",
        "-silent",
        "-window",
        "0xc8",
        "png:-",
    ]


def test_capture_rejects_unknown_backend() -> None:
    with pytest.raises(
        ValueError,
        match="capture backend",
    ):
        X11WindowCapture(
            backend="unknown",
        )


def test_preprocess_rgb_frame_returns_chw_float32() -> None:
    frame = np.zeros(
        (90, 160, 3),
        dtype=np.uint8,
    )
    frame[:, :, 0] = 255
    frame[:, :, 1] = 128

    policy = preprocess_rgb_frame(
        frame,
        width=160,
        height=90,
    )

    assert policy.shape == (3, 90, 160)
    assert policy.dtype == np.float32
    assert policy.flags["C_CONTIGUOUS"]
    assert policy.min() >= 0.0
    assert policy.max() <= 1.0
    assert policy[0, 0, 0] == pytest.approx(1.0)
    assert policy[1, 0, 0] == pytest.approx(
        128.0 / 255.0
    )
    assert policy[2, 0, 0] == pytest.approx(0.0)


def test_preprocess_rgb_frame_resizes_native_ratio() -> None:
    frame = np.full(
        (720, 1280, 3),
        255,
        dtype=np.uint8,
    )

    policy = preprocess_rgb_frame(
        frame,
        width=160,
        height=90,
    )

    assert policy.shape == (3, 90, 160)
    assert np.all(policy == 1.0)


def test_preprocess_rejects_non_rgb_shape() -> None:
    with pytest.raises(
        ValueError,
        match="H×W×3",
    ):
        preprocess_rgb_frame(
            np.zeros(
                (90, 160),
                dtype=np.uint8,
            ),
            width=160,
            height=90,
        )


def test_preprocess_rejects_non_uint8() -> None:
    with pytest.raises(
        TypeError,
        match="uint8",
    ):
        preprocess_rgb_frame(
            np.zeros(
                (90, 160, 3),
                dtype=np.float32,
            ),
            width=160,
            height=90,
        )
