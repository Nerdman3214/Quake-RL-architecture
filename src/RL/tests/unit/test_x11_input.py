"""Tests for fail-safe X11 input primitives."""

from __future__ import annotations

import subprocess

import pytest

from RL.engine.client import (
    X11InputController,
    X11Window,
)


WINDOW = X11Window(
    window_id=200,
    title="Xonotic",
)


def install_runner(
    monkeypatch: pytest.MonkeyPatch,
    commands: list[list[str]],
    *,
    active_window: str = "200",
) -> None:
    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)

        output = (
            active_window
            if command[1] == "getactivewindow"
            else ""
        )

        return subprocess.CompletedProcess(
            command,
            0,
            stdout=output,
            stderr="",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )


def test_focus_activates_and_verifies_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    install_runner(monkeypatch, commands)

    controller = X11InputController()
    controller.focus(WINDOW)

    assert commands == [
        [
            "xdotool",
            "windowactivate",
            "--sync",
            "200",
        ],
        [
            "xdotool",
            "getactivewindow",
        ],
    ]


def test_focus_rejects_wrong_active_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    install_runner(
        monkeypatch,
        commands,
        active_window="201",
    )

    controller = X11InputController()

    with pytest.raises(
        RuntimeError,
        match="did not receive",
    ):
        controller.focus(WINDOW)


def test_hold_key_releases_after_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    sleeps: list[float] = []

    install_runner(monkeypatch, commands)

    controller = X11InputController(
        sleeper=sleeps.append,
    )

    controller.hold_key(
        WINDOW,
        "w",
        duration_seconds=0.05,
    )

    assert sleeps == [0.05]
    assert commands[-2:] == [
        ["xdotool", "keydown", "w"],
        ["xdotool", "keyup", "w"],
    ]
    assert controller.pressed_keys == ()


def test_hold_key_releases_when_sleep_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    install_runner(monkeypatch, commands)

    def failing_sleep(
        duration_seconds: float,
    ) -> None:
        raise RuntimeError("sleep failed")

    controller = X11InputController(
        sleeper=failing_sleep,
    )

    with pytest.raises(
        RuntimeError,
        match="sleep failed",
    ):
        controller.hold_key(
            WINDOW,
            "a",
            duration_seconds=0.05,
        )

    assert commands[-1] == [
        "xdotool",
        "keyup",
        "a",
    ]
    assert controller.pressed_keys == ()


def test_mouse_button_releases_when_sleep_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    install_runner(monkeypatch, commands)

    def failing_sleep(
        duration_seconds: float,
    ) -> None:
        raise RuntimeError("sleep failed")

    controller = X11InputController(
        sleeper=failing_sleep,
    )

    with pytest.raises(
        RuntimeError,
        match="sleep failed",
    ):
        controller.hold_mouse_button(
            WINDOW,
            1,
            duration_seconds=0.02,
        )

    assert commands[-1] == [
        "xdotool",
        "mouseup",
        "1",
    ]
    assert controller.pressed_buttons == ()


def test_relative_mouse_uses_negative_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    install_runner(monkeypatch, commands)

    controller = X11InputController()

    controller.move_mouse_relative(
        WINDOW,
        delta_x=-2,
        delta_y=0,
    )

    assert commands[-1] == [
        "xdotool",
        "mousemove_relative",
        "--sync",
        "--",
        "-2",
        "0",
    ]


def test_scroll_uses_verified_wheel_buttons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    install_runner(monkeypatch, commands)

    controller = X11InputController()

    controller.scroll(
        WINDOW,
        steps=2,
    )

    assert commands[-2:] == [
        ["xdotool", "click", "4"],
        ["xdotool", "click", "4"],
    ]

    controller.scroll(
        WINDOW,
        steps=-1,
    )

    assert commands[-1] == [
        "xdotool",
        "click",
        "5",
    ]


def test_rejects_inputs_outside_safety_bounds() -> None:
    controller = X11InputController(
        max_hold_seconds=0.25,
        max_mouse_delta=8,
        max_scroll_steps=2,
    )

    with pytest.raises(
        ValueError,
        match="duration exceeds",
    ):
        controller.hold_key(
            WINDOW,
            "w",
            duration_seconds=0.5,
        )

    with pytest.raises(
        ValueError,
        match="mouse delta exceeds",
    ):
        controller.move_mouse_relative(
            WINDOW,
            delta_x=9,
        )

    with pytest.raises(
        ValueError,
        match="scroll steps exceed",
    ):
        controller.scroll(
            WINDOW,
            steps=3,
        )
