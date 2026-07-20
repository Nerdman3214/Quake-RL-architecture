"""Parse XInput2 raw events and accumulate human FPS controls."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from RL.actions.composite import (
    CompositeActionCommand,
)


RAW_EVENT_TYPES = frozenset(
    {
        "RawKeyPress",
        "RawKeyRelease",
        "RawButtonPress",
        "RawButtonRelease",
        "RawMotion",
    }
)

XONOTIC_KEYCODES = MappingProxyType(
    {
        "forward": 25,
        "strafe_left": 38,
        "backward": 39,
        "strafe_right": 40,
        "jump": 65,
    }
)

_HEADER_PATTERN = re.compile(
    r"^EVENT type "
    r"(?P<event_code>\d+) "
    r"\((?P<event_type>[^)]+)\)"
)

_DEVICE_PATTERN = re.compile(
    r"^\s*device:\s*"
    r"(?P<master>\d+)"
    r"(?:\s+\((?P<source>\d+)\))?"
    r"\s*$"
)

_TIME_PATTERN = re.compile(
    r"^\s*time:\s*(?P<time>\d+)\s*$"
)

_DETAIL_PATTERN = re.compile(
    r"^\s*detail:\s*(?P<detail>-?\d+)\s*$"
)

_NUMBER = (
    r"[+-]?"
    r"(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
)

_VALUATOR_PATTERN = re.compile(
    r"^\s*(?P<axis>\d+):\s*"
    rf"(?P<value>{_NUMBER})"
    r"(?:\s+\([^)]+\))?"
    r"\s*$"
)


@dataclass(frozen=True)
class XInput2RawEvent:
    """One parsed XInput2 raw event."""

    event_code: int
    event_type: str
    master_device_id: int
    source_device_id: int | None
    time_millis: int
    detail: int
    valuators: Mapping[int, float]

    def __post_init__(self) -> None:
        if self.event_type not in RAW_EVENT_TYPES:
            raise ValueError(
                "event_type must be a supported raw event"
            )

        for field_name, value in (
            ("event_code", self.event_code),
            (
                "master_device_id",
                self.master_device_id,
            ),
            ("time_millis", self.time_millis),
            ("detail", self.detail),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise TypeError(
                    f"{field_name} must be an integer"
                )

        if (
            self.source_device_id is not None
            and (
                isinstance(
                    self.source_device_id,
                    bool,
                )
                or not isinstance(
                    self.source_device_id,
                    int,
                )
            )
        ):
            raise TypeError(
                "source_device_id must be an integer or None"
            )

        normalized: dict[int, float] = {}

        for axis, raw_value in self.valuators.items():
            if (
                isinstance(axis, bool)
                or not isinstance(axis, int)
                or axis < 0
            ):
                raise ValueError(
                    "valuator axes must be nonnegative integers"
                )

            value = float(raw_value)

            if not math.isfinite(value):
                raise ValueError(
                    "valuator values must be finite"
                )

            normalized[axis] = value

        object.__setattr__(
            self,
            "valuators",
            MappingProxyType(normalized),
        )


def parse_xinput2_raw_event(
    block: str,
) -> XInput2RawEvent:
    """Parse one complete XInput2 raw-event text block."""

    if not isinstance(block, str) or not block.strip():
        raise ValueError(
            "event block must be nonempty text"
        )

    lines = block.strip().splitlines()

    header = _HEADER_PATTERN.match(
        lines[0].strip()
    )

    if header is None:
        raise ValueError(
            "invalid XInput2 event header"
        )

    event_type = header.group(
        "event_type"
    )

    if event_type not in RAW_EVENT_TYPES:
        raise ValueError(
            "event block is not a supported raw event"
        )

    master_device_id: int | None = None
    source_device_id: int | None = None
    time_millis: int | None = None
    detail: int | None = None
    valuators: dict[int, float] = {}

    for line in lines[1:]:
        device_match = _DEVICE_PATTERN.match(line)

        if device_match is not None:
            master_device_id = int(
                device_match.group("master")
            )

            source_text = device_match.group(
                "source"
            )

            source_device_id = (
                int(source_text)
                if source_text is not None
                else None
            )
            continue

        time_match = _TIME_PATTERN.match(line)

        if time_match is not None:
            time_millis = int(
                time_match.group("time")
            )
            continue

        detail_match = _DETAIL_PATTERN.match(
            line
        )

        if detail_match is not None:
            detail = int(
                detail_match.group("detail")
            )
            continue

        valuator_match = (
            _VALUATOR_PATTERN.match(line)
        )

        if valuator_match is not None:
            valuators[
                int(
                    valuator_match.group("axis")
                )
            ] = float(
                valuator_match.group("value")
            )

    if master_device_id is None:
        raise ValueError(
            "raw event has no device identifier"
        )

    if time_millis is None:
        raise ValueError(
            "raw event has no timestamp"
        )

    if detail is None:
        raise ValueError(
            "raw event has no detail value"
        )

    return XInput2RawEvent(
        event_code=int(
            header.group("event_code")
        ),
        event_type=event_type,
        master_device_id=master_device_id,
        source_device_id=source_device_id,
        time_millis=time_millis,
        detail=detail,
        valuators=valuators,
    )


class XInput2EventStreamParser:
    """Incrementally parse the text stream from xinput test-xi2."""

    def __init__(self) -> None:
        self._current_lines: list[str] = []

    def _finish_current(
        self,
    ) -> tuple[XInput2RawEvent, ...]:
        if not self._current_lines:
            return ()

        block = "\n".join(
            self._current_lines
        )

        self._current_lines = []

        header = _HEADER_PATTERN.match(
            block.splitlines()[0].strip()
        )

        if header is None:
            raise ValueError(
                "invalid XInput2 event stream"
            )

        if (
            header.group("event_type")
            not in RAW_EVENT_TYPES
        ):
            return ()

        return (
            parse_xinput2_raw_event(block),
        )

    def feed_line(
        self,
        line: str,
    ) -> tuple[XInput2RawEvent, ...]:
        """Consume one line and return any completed raw event."""

        if not isinstance(line, str):
            raise TypeError(
                "line must be text"
            )

        completed: tuple[
            XInput2RawEvent,
            ...
        ] = ()

        if line.startswith("EVENT type "):
            completed = self._finish_current()
            self._current_lines = [
                line.rstrip("\n")
            ]
        elif self._current_lines:
            self._current_lines.append(
                line.rstrip("\n")
            )

        return completed

    def feed_lines(
        self,
        lines: Iterable[str],
    ) -> tuple[XInput2RawEvent, ...]:
        """Consume multiple lines and return completed events."""

        events: list[XInput2RawEvent] = []

        for line in lines:
            events.extend(
                self.feed_line(line)
            )

        return tuple(events)

    def flush(
        self,
    ) -> tuple[XInput2RawEvent, ...]:
        """Finish the final buffered event."""

        return self._finish_current()


class HumanInputAccumulator:
    """Maintain held controls and per-tick transient input."""

    def __init__(
        self,
        *,
        keyboard_source_ids: (
            Iterable[int] | None
        ) = None,
        mouse_source_ids: (
            Iterable[int] | None
        ) = None,
    ) -> None:
        self._keyboard_source_ids = (
            frozenset(keyboard_source_ids)
            if keyboard_source_ids is not None
            else None
        )

        self._mouse_source_ids = (
            frozenset(mouse_source_ids)
            if mouse_source_ids is not None
            else None
        )

        self._pressed_keycodes: set[int] = set()
        self._pressed_buttons: set[int] = set()

        self._turn_delta_x = 0.0
        self._look_delta_y = 0.0
        self._weapon_delta = 0

    @property
    def pressed_keycodes(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(self._pressed_keycodes)
        )

    @property
    def pressed_buttons(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(self._pressed_buttons)
        )

    def _source_allowed(
        self,
        event: XInput2RawEvent,
        allowed_sources: (
            frozenset[int] | None
        ),
    ) -> bool:
        return (
            allowed_sources is None
            or event.source_device_id
            in allowed_sources
        )

    def apply(
        self,
        event: XInput2RawEvent,
    ) -> None:
        """Apply one raw event to the current control state."""

        if not isinstance(
            event,
            XInput2RawEvent,
        ):
            raise TypeError(
                "event must be an XInput2RawEvent"
            )

        if event.event_type.startswith(
            "RawKey"
        ):
            if not self._source_allowed(
                event,
                self._keyboard_source_ids,
            ):
                return

            if event.event_type == "RawKeyPress":
                self._pressed_keycodes.add(
                    event.detail
                )
            else:
                self._pressed_keycodes.discard(
                    event.detail
                )

            return

        if not self._source_allowed(
            event,
            self._mouse_source_ids,
        ):
            return

        if event.event_type == "RawMotion":
            self._turn_delta_x += (
                event.valuators.get(
                    0,
                    0.0,
                )
            )

            self._look_delta_y += (
                event.valuators.get(
                    1,
                    0.0,
                )
            )
            return

        if event.event_type == "RawButtonPress":
            if event.detail == 4:
                self._weapon_delta += 1
            elif event.detail == 5:
                self._weapon_delta -= 1
            else:
                self._pressed_buttons.add(
                    event.detail
                )
            return

        if event.event_type == "RawButtonRelease":
            if event.detail not in (4, 5):
                self._pressed_buttons.discard(
                    event.detail
                )

    def snapshot(
        self,
        *,
        duration_ticks: int = 1,
    ) -> CompositeActionCommand:
        """Return one tick and clear motion/wheel transients."""

        forward_axis = (
            int(
                XONOTIC_KEYCODES["forward"]
                in self._pressed_keycodes
            )
            - int(
                XONOTIC_KEYCODES["backward"]
                in self._pressed_keycodes
            )
        )

        strafe_axis = (
            int(
                XONOTIC_KEYCODES[
                    "strafe_right"
                ]
                in self._pressed_keycodes
            )
            - int(
                XONOTIC_KEYCODES[
                    "strafe_left"
                ]
                in self._pressed_keycodes
            )
        )

        command = CompositeActionCommand(
            forward_axis=forward_axis,
            strafe_axis=strafe_axis,
            turn_delta_x=(
                self._turn_delta_x
            ),
            look_delta_y=(
                self._look_delta_y
            ),
            fire=1 in self._pressed_buttons,
            jump=(
                XONOTIC_KEYCODES["jump"]
                in self._pressed_keycodes
            ),
            weapon_delta=self._weapon_delta,
            duration_ticks=duration_ticks,
        )

        self._turn_delta_x = 0.0
        self._look_delta_y = 0.0
        self._weapon_delta = 0

        return command

    def reset(self) -> None:
        """Clear held and transient state at recorder shutdown."""

        self._pressed_keycodes.clear()
        self._pressed_buttons.clear()

        self._turn_delta_x = 0.0
        self._look_delta_y = 0.0
        self._weapon_delta = 0
