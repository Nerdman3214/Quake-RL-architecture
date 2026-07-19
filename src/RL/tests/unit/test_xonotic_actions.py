"""Tests for Xonotic ActionCommand execution."""

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


class FakeInputController:
    """Record mapped input without sending real X11 commands."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def hold_key(
        self,
        window: X11Window,
        key: str,
        *,
        duration_seconds: float,
    ) -> None:
        self.calls.append(
            (
                "hold_key",
                window,
                key,
                duration_seconds,
            )
        )

    def hold_mouse_button(
        self,
        window: X11Window,
        button: int,
        *,
        duration_seconds: float,
    ) -> None:
        self.calls.append(
            (
                "hold_mouse_button",
                window,
                button,
                duration_seconds,
            )
        )

    def move_mouse_relative(
        self,
        window: X11Window,
        *,
        delta_x: int,
        delta_y: int = 0,
    ) -> None:
        self.calls.append(
            (
                "move_mouse_relative",
                window,
                delta_x,
                delta_y,
            )
        )

    def scroll(
        self,
        window: X11Window,
        *,
        steps: int,
    ) -> None:
        self.calls.append(
            (
                "scroll",
                window,
                steps,
            )
        )

    def release_all(
        self,
        window: X11Window | None = None,
    ) -> None:
        self.calls.append(
            (
                "release_all",
                window,
            )
        )


def make_executor(
    controller: FakeInputController,
    *,
    sleeper=lambda _: None,
) -> XonoticActionExecutor:
    return XonoticActionExecutor(
        controller,
        tick_seconds=0.05,
        turn_pixels_per_tick=2,
        max_duration_ticks=20,
        sleeper=sleeper,
    )


def test_constructor_validates_safety_values() -> None:
    controller = FakeInputController()

    with pytest.raises(
        ValueError,
        match="tick duration",
    ):
        XonoticActionExecutor(
            controller,
            tick_seconds=0,
        )

    with pytest.raises(
        ValueError,
        match="turn pixels",
    ):
        XonoticActionExecutor(
            controller,
            turn_pixels_per_tick=0,
        )

    with pytest.raises(
        ValueError,
        match="maximum duration",
    ):
        XonoticActionExecutor(
            controller,
            max_duration_ticks=0,
        )


def test_no_op_only_sleeps() -> None:
    controller = FakeInputController()
    sleeps: list[float] = []

    executor = make_executor(
        controller,
        sleeper=sleeps.append,
    )

    executor.execute(
        WINDOW,
        ActionCommand(
            action=DiscreteAction.NO_OP,
            duration_ticks=4,
        ),
    )

    assert sleeps == [pytest.approx(0.2)]
    assert controller.calls == []


@pytest.mark.parametrize(
    ("action", "key"),
    [
        (DiscreteAction.FORWARD, "w"),
        (DiscreteAction.BACKWARD, "s"),
        (DiscreteAction.STRAFE_LEFT, "a"),
        (DiscreteAction.STRAFE_RIGHT, "d"),
    ],
)
def test_maps_movement_keys(
    action: DiscreteAction,
    key: str,
) -> None:
    controller = FakeInputController()
    executor = make_executor(controller)

    executor.execute(
        WINDOW,
        ActionCommand(
            action=action,
            duration_ticks=2,
        ),
    )

    assert controller.calls == [
        (
            "hold_key",
            WINDOW,
            key,
            pytest.approx(0.1),
        )
    ]


def test_maps_jump_to_space() -> None:
    controller = FakeInputController()
    executor = make_executor(controller)

    executor.execute(
        WINDOW,
        ActionCommand(
            action=DiscreteAction.JUMP,
            duration_ticks=1,
        ),
    )

    assert controller.calls == [
        (
            "hold_key",
            WINDOW,
            "space",
            pytest.approx(0.05),
        )
    ]


def test_maps_fire_to_primary_mouse_button() -> None:
    controller = FakeInputController()
    executor = make_executor(controller)

    executor.execute(
        WINDOW,
        ActionCommand(
            action=DiscreteAction.FIRE,
            duration_ticks=3,
        ),
    )

    assert controller.calls == [
        (
            "hold_mouse_button",
            WINDOW,
            1,
            pytest.approx(0.15),
        )
    ]


@pytest.mark.parametrize(
    ("action", "expected_delta"),
    [
        (DiscreteAction.TURN_LEFT, -6),
        (DiscreteAction.TURN_RIGHT, 6),
    ],
)
def test_maps_turning_to_relative_mouse(
    action: DiscreteAction,
    expected_delta: int,
) -> None:
    controller = FakeInputController()
    executor = make_executor(controller)

    executor.execute(
        WINDOW,
        ActionCommand(
            action=action,
            duration_ticks=3,
        ),
    )

    assert controller.calls == [
        (
            "move_mouse_relative",
            WINDOW,
            expected_delta,
            0,
        )
    ]


@pytest.mark.parametrize(
    ("action", "expected_steps"),
    [
        (DiscreteAction.NEXT_WEAPON, 1),
        (DiscreteAction.PREVIOUS_WEAPON, -1),
    ],
)
def test_maps_weapon_switching_to_wheel(
    action: DiscreteAction,
    expected_steps: int,
) -> None:
    controller = FakeInputController()
    executor = make_executor(controller)

    executor.execute(
        WINDOW,
        ActionCommand(
            action=action,
            duration_ticks=1,
        ),
    )

    assert controller.calls == [
        (
            "scroll",
            WINDOW,
            expected_steps,
        )
    ]


def test_rejects_repeated_weapon_switch_duration() -> None:
    controller = FakeInputController()
    executor = make_executor(controller)

    with pytest.raises(
        ValueError,
        match="require duration_ticks=1",
    ):
        executor.execute(
            WINDOW,
            ActionCommand(
                action=DiscreteAction.NEXT_WEAPON,
                duration_ticks=2,
            ),
        )

    assert controller.calls == []


def test_rejects_invalid_duration_ticks() -> None:
    controller = FakeInputController()
    executor = make_executor(controller)

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        executor.execute(
            WINDOW,
            ActionCommand(
                action=DiscreteAction.FORWARD,
                duration_ticks=0,
            ),
        )

    with pytest.raises(
        ValueError,
        match="configured maximum",
    ):
        executor.execute(
            WINDOW,
            ActionCommand(
                action=DiscreteAction.FORWARD,
                duration_ticks=21,
            ),
        )


def test_rejects_non_discrete_action() -> None:
    controller = FakeInputController()
    executor = make_executor(controller)

    command = ActionCommand(
        action=999,  # type: ignore[arg-type]
        duration_ticks=1,
    )

    with pytest.raises(
        TypeError,
        match="DiscreteAction",
    ):
        executor.execute(
            WINDOW,
            command,
        )

    assert controller.calls == []


def test_release_all_delegates_to_input_controller() -> None:
    controller = FakeInputController()
    executor = make_executor(controller)

    executor.release_all(WINDOW)

    assert controller.calls == [
        (
            "release_all",
            WINDOW,
        )
    ]
