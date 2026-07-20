"""Tests for XInput2 human-control parsing."""

from __future__ import annotations

from RL.recording import (
    HumanInputAccumulator,
    XInput2EventStreamParser,
    XInput2RawEvent,
    parse_xinput2_raw_event,
)


KEY_PRESS = """EVENT type 13 (RawKeyPress)
    device: 3 (13)
    time:   286587137
    detail: 25
    valuators:
"""

KEY_RELEASE = """EVENT type 14 (RawKeyRelease)
    device: 3 (13)
    time:   286587393
    detail: 25
    valuators:
"""

MOTION = """EVENT type 17 (RawMotion)
    device: 2 (10)
    time:   286585319
    detail: 0
    flags:
    valuators:
          0: -1.00 (-1.00)
          1: 2.50 (2.50)
"""

BUTTON_PRESS = """EVENT type 15 (RawButtonPress)
    device: 2 (10)
    time:   286586244
    detail: 1
    flags:
    valuators:
"""

BUTTON_RELEASE = """EVENT type 16 (RawButtonRelease)
    device: 2 (10)
    time:   286586324
    detail: 1
    flags:
    valuators:
"""


def event(
    event_type: str,
    *,
    source: int,
    detail: int,
    valuators: dict[int, float] | None = None,
) -> XInput2RawEvent:
    return XInput2RawEvent(
        event_code={
            "RawKeyPress": 13,
            "RawKeyRelease": 14,
            "RawButtonPress": 15,
            "RawButtonRelease": 16,
            "RawMotion": 17,
        }[event_type],
        event_type=event_type,
        master_device_id=(
            3
            if event_type.startswith("RawKey")
            else 2
        ),
        source_device_id=source,
        time_millis=100,
        detail=detail,
        valuators=valuators or {},
    )


def test_parses_verified_raw_key_format() -> None:
    parsed = parse_xinput2_raw_event(
        KEY_PRESS
    )

    assert parsed.event_type == "RawKeyPress"
    assert parsed.master_device_id == 3
    assert parsed.source_device_id == 13
    assert parsed.time_millis == 286587137
    assert parsed.detail == 25
    assert dict(parsed.valuators) == {}


def test_parses_verified_raw_motion_format() -> None:
    parsed = parse_xinput2_raw_event(
        MOTION
    )

    assert parsed.source_device_id == 10
    assert parsed.valuators[0] == -1.0
    assert parsed.valuators[1] == 2.5


def test_stream_parser_ignores_nonraw_events() -> None:
    text = (
        KEY_PRESS
        + "EVENT type 2 (KeyPress)\n"
        + "    device: 3 (13)\n"
        + "    time: 286587138\n"
        + "    detail: 25\n"
        + MOTION
    )

    parser = XInput2EventStreamParser()

    parsed = list(
        parser.feed_lines(
            text.splitlines(
                keepends=True
            )
        )
    )

    parsed.extend(
        parser.flush()
    )

    assert [
        item.event_type
        for item in parsed
    ] == [
        "RawKeyPress",
        "RawMotion",
    ]


def test_accumulator_combines_simultaneous_controls() -> None:
    accumulator = HumanInputAccumulator(
        keyboard_source_ids={13},
        mouse_source_ids={10},
    )

    for raw_event in (
        event(
            "RawKeyPress",
            source=13,
            detail=25,
        ),
        event(
            "RawKeyPress",
            source=13,
            detail=38,
        ),
        event(
            "RawKeyPress",
            source=13,
            detail=65,
        ),
        event(
            "RawButtonPress",
            source=10,
            detail=1,
        ),
        event(
            "RawMotion",
            source=10,
            detail=0,
            valuators={
                0: 6.0,
                1: -3.0,
            },
        ),
    ):
        accumulator.apply(raw_event)

    command = accumulator.snapshot()

    assert command.forward_axis == 1
    assert command.strafe_axis == -1
    assert command.turn_delta_x == 6.0
    assert command.look_delta_y == -3.0
    assert command.fire
    assert command.jump


def test_snapshot_clears_only_transient_controls() -> None:
    accumulator = HumanInputAccumulator()

    accumulator.apply(
        event(
            "RawKeyPress",
            source=13,
            detail=25,
        )
    )

    accumulator.apply(
        event(
            "RawButtonPress",
            source=10,
            detail=1,
        )
    )

    accumulator.apply(
        event(
            "RawMotion",
            source=10,
            detail=0,
            valuators={
                0: 5.0,
                1: 2.0,
            },
        )
    )

    accumulator.apply(
        event(
            "RawButtonPress",
            source=10,
            detail=4,
        )
    )

    first = accumulator.snapshot()
    second = accumulator.snapshot()

    assert first.forward_axis == 1
    assert first.fire
    assert first.turn_delta_x == 5.0
    assert first.weapon_delta == 1

    assert second.forward_axis == 1
    assert second.fire
    assert second.turn_delta_x == 0.0
    assert second.look_delta_y == 0.0
    assert second.weapon_delta == 0


def test_release_events_clear_held_controls() -> None:
    accumulator = HumanInputAccumulator()

    accumulator.apply(
        parse_xinput2_raw_event(
            KEY_PRESS
        )
    )

    accumulator.apply(
        parse_xinput2_raw_event(
            BUTTON_PRESS
        )
    )

    accumulator.apply(
        parse_xinput2_raw_event(
            KEY_RELEASE
        )
    )

    accumulator.apply(
        parse_xinput2_raw_event(
            BUTTON_RELEASE
        )
    )

    command = accumulator.snapshot()

    assert command.forward_axis == 0
    assert not command.fire


def test_wheel_events_accumulate_and_reset() -> None:
    accumulator = HumanInputAccumulator()

    accumulator.apply(
        event(
            "RawButtonPress",
            source=10,
            detail=4,
        )
    )

    accumulator.apply(
        event(
            "RawButtonPress",
            source=10,
            detail=4,
        )
    )

    accumulator.apply(
        event(
            "RawButtonPress",
            source=10,
            detail=5,
        )
    )

    assert (
        accumulator.snapshot().weapon_delta
        == 1
    )

    assert (
        accumulator.snapshot().weapon_delta
        == 0
    )


def test_source_filters_ignore_other_devices() -> None:
    accumulator = HumanInputAccumulator(
        keyboard_source_ids={13},
        mouse_source_ids={10},
    )

    accumulator.apply(
        event(
            "RawKeyPress",
            source=99,
            detail=25,
        )
    )

    accumulator.apply(
        event(
            "RawMotion",
            source=98,
            detail=0,
            valuators={
                0: 9.0,
            },
        )
    )

    assert accumulator.snapshot().is_no_op


def test_reset_clears_unbalanced_capture_state() -> None:
    accumulator = HumanInputAccumulator()

    accumulator.apply(
        event(
            "RawKeyPress",
            source=13,
            detail=25,
        )
    )

    accumulator.apply(
        event(
            "RawButtonPress",
            source=10,
            detail=1,
        )
    )

    assert accumulator.pressed_keycodes == (
        25,
    )

    assert accumulator.pressed_buttons == (
        1,
    )

    accumulator.reset()

    assert accumulator.pressed_keycodes == ()
    assert accumulator.pressed_buttons == ()
    assert accumulator.snapshot().is_no_op
