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


def test_capture_rgb_uses_direct_window_output(
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

    capture = X11WindowCapture()

    frame = capture.capture_rgb(
        X11Window(
            window_id=200,
            title="Xonotic",
        )
    )

    assert frame.shape == (4, 8, 3)
    assert frame.dtype == np.uint8
    assert frame[0, 0].tolist() == [12, 34, 56]
    assert commands[0] == [
        "import",
        "-silent",
        "-window",
        "0xc8",
        "png:-",
    ]


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
