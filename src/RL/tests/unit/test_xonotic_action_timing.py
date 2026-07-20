"""Tests for fixed-duration Xonotic turn and wheel actions."""

from __future__ import annotations

import pytest

from RL.actions.contracts import (
    ActionCommand,
    DiscreteAction,
)
from RL.engine.client import (
    X11Window,
    XonoticActionExecutor,
)


WINDOW = X11Window(
    window_id=200,
    title="Xonotic",
)


class FakeController:
    def __init__(self) -> None:
        self.calls = []

    def hold_key(
        self,
        window,
        key,
        *,
        duration_seconds,
    ):
        self.calls.append(
            (
                "hold_key",
                key,
                duration_seconds,
            )
        )

    def hold_mouse_button(
        self,
        window,
        button,
        *,
        duration_seconds,
    ):
        self.calls.append(
            (
                "hold_mouse_button",
                button,
                duration_seconds,
            )
        )

    def move_mouse_relative(
        self,
        window,
        *,
        delta_x,
        delta_y=0,
    ):
        self.calls.append(
            (
                "move_mouse_relative",
                delta_x,
                delta_y,
            )
        )

    def scroll(
        self,
        window,
        *,
        steps,
    ):
        self.calls.append(
            (
                "scroll",
                steps,
            )
        )

    def release_all(
        self,
        window=None,
    ):
        self.calls.append(
            (
                "release_all",
                window,
            )
        )


def test_turn_action_honors_tick_duration() -> None:
    sleeps = []
    controller = FakeController()

    executor = XonoticActionExecutor(
        controller,
        tick_seconds=0.05,
        turn_pixels_per_tick=2,
        sleeper=sleeps.append,
    )

    executor.execute(
        WINDOW,
        ActionCommand(
            action=DiscreteAction.TURN_RIGHT,
            duration_ticks=3,
        ),
    )

    assert controller.calls == [
        (
            "move_mouse_relative",
            6,
            0,
        )
    ]

    assert sleeps == [
        pytest.approx(0.15)
    ]


def test_weapon_action_honors_one_tick_duration() -> None:
    sleeps = []
    controller = FakeController()

    executor = XonoticActionExecutor(
        controller,
        tick_seconds=0.05,
        sleeper=sleeps.append,
    )

    executor.execute(
        WINDOW,
        ActionCommand(
            action=DiscreteAction.NEXT_WEAPON,
            duration_ticks=1,
        ),
    )

    assert controller.calls == [
        (
            "scroll",
            1,
        )
    ]

    assert sleeps == [
        pytest.approx(0.05)
    ]
