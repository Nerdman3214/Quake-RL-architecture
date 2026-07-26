"""Live XInput2 and X11 adapter for personal match recording."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from RL.actions.composite import (
    CompositeActionCommand,
)
from RL.engine.client.x11_window import (
    X11Window,
)
from RL.events.contracts import Event
from RL.recording.match_recorder import (
    AutomaticMatchRecorder,
    RecordedMatchResult,
)
from RL.recording.xonotic_telemetry_source import (
    XonoticTelemetrySnapshot,
)
from RL.recording.xinput2 import (
    HumanInputAccumulator,
    XInput2EventStreamParser,
)


_TRANSITION_IGNORED_EVENT_TYPES = frozenset(
    {
        "session_started",
        "session_ended",
        "match_started",
        "match_ended",
        "suppressed_human_event",
    }
)


class LatestTelemetrySource(Protocol):
    """Latest telemetry cache used by the live recorder."""

    def read_latest(
        self,
    ) -> XonoticTelemetrySnapshot | None:
        """Return the newest available telemetry snapshot."""


@dataclass(frozen=True)
class HumanInputSnapshot:
    """One synchronized snapshot of accumulated human controls."""

    command: CompositeActionCommand
    pressed_keycodes: tuple[int, ...]
    pressed_buttons: tuple[int, ...]
    timestamp_ns: int
    raw_event_count: int
    weapon_previous_event_count: int = 0
    weapon_next_event_count: int = 0


class IncrementalEventCursor(Protocol):
    """Append-only event cursor used by the live runtime."""

    def read_new_events(
        self,
    ) -> tuple[Event, ...]:
        """Return newly appended events."""


class WindowFrameCapture(Protocol):
    """Visible-window RGB capture operations."""

    def capture_rgb(
        self,
        window: X11Window,
    ) -> np.ndarray:
        """Capture one RGB frame."""


class HumanInputCapture(Protocol):
    """Human input operations used by the runtime."""

    def poll(
        self,
        *,
        quiet_period_seconds: float = 0.002,
    ) -> int:
        """Consume currently available raw events."""

    def snapshot(
        self,
        *,
        duration_ticks: int = 1,
    ) -> HumanInputSnapshot:
        """Return one synchronized input snapshot."""

    def reset(self) -> None:
        """Clear accumulated input state."""

    def close(self) -> None:
        """Stop the capture source."""


def build_xinput2_command(
    *,
    xinput_command: str,
    xi2_device: str,
) -> tuple[str, ...]:
    """Build the command used for global XInput2 raw events."""

    if not xinput_command.strip():
        raise ValueError(
            "xinput_command must not be blank"
        )

    if not xi2_device.strip():
        raise ValueError(
            "xi2_device must not be blank"
        )

    return (
        xinput_command,
        "test-xi2",
        "--root",
        xi2_device,
    )


class BufferedHumanInputState:
    """Parse raw-event lines and maintain synchronized controls."""

    def __init__(
        self,
        *,
        keyboard_source_ids: (
            Iterable[int] | None
        ) = None,
        mouse_source_ids: (
            Iterable[int] | None
        ) = None,
        time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not callable(time_ns):
            raise TypeError(
                "time_ns must be callable"
            )

        self._parser = (
            XInput2EventStreamParser()
        )

        self._accumulator = (
            HumanInputAccumulator(
                keyboard_source_ids=(
                    keyboard_source_ids
                ),
                mouse_source_ids=(
                    mouse_source_ids
                ),
            )
        )

        self._time_ns = time_ns
        self._raw_event_count = 0

    @property
    def raw_event_count(self) -> int:
        return self._raw_event_count

    def _apply(
        self,
        events: Iterable[object],
    ) -> int:
        count = 0

        for event in events:
            self._accumulator.apply(event)
            count += 1

        self._raw_event_count += count
        return count

    def feed_line(
        self,
        line: str,
    ) -> int:
        """Consume one textual XInput2 output line."""

        return self._apply(
            self._parser.feed_line(line)
        )

    def flush_if_complete(self) -> int:
        """Flush one completed event after an output quiet period."""

        return self._apply(
            self._parser.flush_if_complete()
        )

    def snapshot(
        self,
        *,
        duration_ticks: int = 1,
    ) -> HumanInputSnapshot:
        """Return held and transient input in one snapshot."""

        keycodes = (
            self._accumulator.pressed_keycodes
        )

        buttons = (
            self._accumulator.pressed_buttons
        )

        weapon_previous_event_count = (
            self._accumulator
            .weapon_previous_event_count
        )
        weapon_next_event_count = (
            self._accumulator
            .weapon_next_event_count
        )

        command = self._accumulator.snapshot(
            duration_ticks=duration_ticks
        )

        return HumanInputSnapshot(
            command=command,
            pressed_keycodes=keycodes,
            pressed_buttons=buttons,
            timestamp_ns=int(
                self._time_ns()
            ),
            raw_event_count=(
                self._raw_event_count
            ),
            weapon_previous_event_count=(
                weapon_previous_event_count
            ),
            weapon_next_event_count=(
                weapon_next_event_count
            ),
        )

    def reset(self) -> None:
        self._accumulator.reset()


class XInput2ProcessCapture:
    """Read ``xinput test-xi2`` without generating input."""

    _END = object()

    def __init__(
        self,
        *,
        xi2_device: str,
        keyboard_source_ids: (
            Iterable[int] | None
        ) = None,
        mouse_source_ids: (
            Iterable[int] | None
        ) = None,
        xinput_command: str = "xinput",
        process_factory: Callable[..., object] = (
            subprocess.Popen
        ),
        time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.command = build_xinput2_command(
            xinput_command=xinput_command,
            xi2_device=xi2_device,
        )

        if not callable(process_factory):
            raise TypeError(
                "process_factory must be callable"
            )

        self._process_factory = (
            process_factory
        )

        self._state = BufferedHumanInputState(
            keyboard_source_ids=(
                keyboard_source_ids
            ),
            mouse_source_ids=(
                mouse_source_ids
            ),
            time_ns=time_ns,
        )

        self._queue: queue.Queue[
            object
        ] = queue.Queue()

        self._process: object | None = None
        self._thread: (
            threading.Thread | None
        ) = None

        self._closing = False
        self._stream_ended = False

    @property
    def running(self) -> bool:
        process = self._process

        return (
            process is not None
            and getattr(
                process,
                "poll",
            )() is None
        )

    @property
    def raw_event_count(self) -> int:
        return self._state.raw_event_count

    def _read_stdout(self) -> None:
        process = self._process

        if process is None:
            self._queue.put(self._END)
            return

        stdout = getattr(
            process,
            "stdout",
            None,
        )

        try:
            if stdout is not None:
                for line in stdout:
                    self._queue.put(line)
        finally:
            self._queue.put(self._END)

    def start(self) -> None:
        """Start the passive XInput2 monitor."""

        if self._process is not None:
            raise RuntimeError(
                "XInput2 capture was already started"
            )

        process = self._process_factory(
            list(self.command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if getattr(
            process,
            "stdout",
            None,
        ) is None:
            raise RuntimeError(
                "XInput2 process has no stdout"
            )

        self._process = process

        self._thread = threading.Thread(
            target=self._read_stdout,
            name="xinput2-raw-reader",
            daemon=True,
        )

        self._thread.start()

    def _check_process(self) -> None:
        process = self._process

        if (
            process is None
            or self._closing
        ):
            return

        returncode = getattr(
            process,
            "poll",
        )()

        if (
            returncode is not None
            and self._stream_ended
        ):
            raise RuntimeError(
                "xinput test-xi2 exited "
                f"with status {returncode}"
            )

    def poll(
        self,
        *,
        quiet_period_seconds: float = 0.002,
    ) -> int:
        """Drain available lines and flush a complete final event."""

        if quiet_period_seconds < 0.0:
            raise ValueError(
                "quiet_period_seconds must not be negative"
            )

        if self._process is None:
            raise RuntimeError(
                "XInput2 capture must be started"
            )

        applied = 0
        received_line = False

        while True:
            try:
                if received_line:
                    item = self._queue.get(
                        timeout=(
                            quiet_period_seconds
                        )
                    )
                else:
                    item = self._queue.get_nowait()
            except queue.Empty:
                break

            if item is self._END:
                self._stream_ended = True
                break

            if not isinstance(item, str):
                raise RuntimeError(
                    "XInput2 reader returned "
                    "non-text output"
                )

            applied += self._state.feed_line(
                item
            )

            received_line = True

        applied += (
            self._state.flush_if_complete()
        )

        self._check_process()
        return applied

    def snapshot(
        self,
        *,
        duration_ticks: int = 1,
    ) -> HumanInputSnapshot:
        """Poll and return one synchronized input snapshot."""

        self.poll()

        return self._state.snapshot(
            duration_ticks=duration_ticks
        )

    def reset(self) -> None:
        self._state.reset()

    def close(self) -> None:
        """Stop the passive monitor and release resources."""

        process = self._process

        if process is None:
            return

        self._closing = True

        if getattr(
            process,
            "poll",
        )() is None:
            getattr(
                process,
                "terminate",
            )()

            try:
                getattr(
                    process,
                    "wait",
                )(timeout=1.0)
            except subprocess.TimeoutExpired:
                getattr(
                    process,
                    "kill",
                )()

                getattr(
                    process,
                    "wait",
                )(timeout=1.0)

        thread = self._thread

        if thread is not None:
            thread.join(timeout=1.0)

        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break

            if item is self._END:
                self._stream_ended = True
                continue

            if isinstance(item, str):
                self._state.feed_line(item)

        self._state.flush_if_complete()
        self._state.reset()

    def __enter__(
        self,
    ) -> "XInput2ProcessCapture":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


class CombinedHumanInputCapture:
    """Merge keyboard and pointer XInput2 monitors."""

    def __init__(
        self,
        *,
        keyboard_capture: object,
        mouse_capture: object,
    ) -> None:
        required_methods = (
            "start",
            "poll",
            "snapshot",
            "reset",
            "close",
        )

        for capture_name, capture in (
            (
                "keyboard_capture",
                keyboard_capture,
            ),
            (
                "mouse_capture",
                mouse_capture,
            ),
        ):
            missing_methods = [
                method_name
                for method_name in required_methods
                if not callable(
                    getattr(
                        capture,
                        method_name,
                        None,
                    )
                )
            ]

            if missing_methods:
                raise TypeError(
                    f"{capture_name} is missing "
                    "required methods: "
                    + ", ".join(
                        missing_methods
                    )
                )

        if keyboard_capture is mouse_capture:
            raise ValueError(
                "keyboard and mouse captures "
                "must be different instances"
            )

        self.keyboard_capture = (
            keyboard_capture
        )
        self.mouse_capture = mouse_capture
        self._started = False

    def start(self) -> None:
        """Start both passive XInput2 monitors."""

        if self._started:
            raise RuntimeError(
                "combined input capture was "
                "already started"
            )

        self.keyboard_capture.start()

        try:
            self.mouse_capture.start()
        except Exception:
            self.keyboard_capture.close()
            raise

        self._started = True

    def poll(
        self,
        *,
        quiet_period_seconds: float = 0.002,
    ) -> int:
        """Consume available keyboard and mouse events."""

        keyboard_count = (
            self.keyboard_capture.poll(
                quiet_period_seconds=(
                    quiet_period_seconds
                ),
            )
        )

        mouse_count = self.mouse_capture.poll(
            quiet_period_seconds=(
                quiet_period_seconds
            ),
        )

        return keyboard_count + mouse_count

    def snapshot(
        self,
        *,
        duration_ticks: int = 1,
    ) -> HumanInputSnapshot:
        """Merge keyboard controls with pointer controls."""

        keyboard = (
            self.keyboard_capture.snapshot(
                duration_ticks=duration_ticks
            )
        )

        mouse = self.mouse_capture.snapshot(
            duration_ticks=duration_ticks
        )

        keyboard_command = keyboard.command
        mouse_command = mouse.command

        weapon_delta = (
            mouse_command.weapon_delta
            if mouse_command.weapon_delta != 0
            else keyboard_command.weapon_delta
        )

        command = CompositeActionCommand(
            forward_axis=(
                keyboard_command.forward_axis
            ),
            strafe_axis=(
                keyboard_command.strafe_axis
            ),
            turn_delta_x=(
                mouse_command.turn_delta_x
            ),
            look_delta_y=(
                mouse_command.look_delta_y
            ),
            fire=(
                keyboard_command.fire
                or mouse_command.fire
            ),
            jump=(
                keyboard_command.jump
                or mouse_command.jump
            ),
            weapon_delta=weapon_delta,
            duration_ticks=duration_ticks,
        )

        return HumanInputSnapshot(
            command=command,
            pressed_keycodes=tuple(
                sorted(
                    set(
                        keyboard.pressed_keycodes
                    )
                    | set(
                        mouse.pressed_keycodes
                    )
                )
            ),
            pressed_buttons=tuple(
                sorted(
                    set(
                        keyboard.pressed_buttons
                    )
                    | set(
                        mouse.pressed_buttons
                    )
                )
            ),
            timestamp_ns=max(
                keyboard.timestamp_ns,
                mouse.timestamp_ns,
            ),
            raw_event_count=(
                keyboard.raw_event_count
                + mouse.raw_event_count
            ),
            weapon_previous_event_count=(
                keyboard.weapon_previous_event_count
                + mouse.weapon_previous_event_count
            ),
            weapon_next_event_count=(
                keyboard.weapon_next_event_count
                + mouse.weapon_next_event_count
            ),
        )

    def reset(self) -> None:
        self.keyboard_capture.reset()
        self.mouse_capture.reset()

    def close(self) -> None:
        self.mouse_capture.close()
        self.keyboard_capture.close()
        self._started = False


class LivePersonalMatchRecorder:
    """Connect events, human input, and X11 frames."""

    def __init__(
        self,
        *,
        recorder: AutomaticMatchRecorder,
        event_cursor: IncrementalEventCursor,
        frame_capture: WindowFrameCapture,
        window: X11Window,
        input_capture: HumanInputCapture,
        telemetry_source: LatestTelemetrySource | None = None,
        frame_rate: float,
        max_frames_per_match: int = 0,
        progress_every_frames: int = 0,
        progress_callback: (
            Callable[[int, int], None] | None
        ) = None,
        poll_interval_seconds: float = 0.01,
        monotonic: Callable[[], float] = (
            time.monotonic
        ),
        sleeper: Callable[[float], None] = (
            time.sleep
        ),
    ) -> None:
        if frame_rate <= 0.0:
            raise ValueError(
                "frame_rate must be positive"
            )

        if max_frames_per_match < 0:
            raise ValueError(
                "max_frames_per_match must not "
                "be negative"
            )

        if progress_every_frames < 0:
            raise ValueError(
                "progress_every_frames must not be negative"
            )

        if poll_interval_seconds <= 0.0:
            raise ValueError(
                "poll_interval_seconds must be positive"
            )

        self.recorder = recorder
        self.event_cursor = event_cursor
        self.frame_capture = frame_capture
        self.window = window
        self.input_capture = input_capture
        self.telemetry_source = telemetry_source

        self.frame_rate = float(
            frame_rate
        )

        self.frame_period_seconds = (
            1.0 / self.frame_rate
        )

        self.max_frames_per_match = int(
            max_frames_per_match
        )

        self.progress_every_frames = int(
            progress_every_frames
        )
        self.progress_callback = (
            progress_callback
        )

        self.poll_interval_seconds = float(
            poll_interval_seconds
        )

        self._monotonic = monotonic
        self._sleeper = sleeper

        self._frames_this_match = 0
        self._next_frame_at = (
            self._monotonic()
        )

        self._event_session_closed = False
        self._pending_game_events: list[Event] = []

    @property
    def event_session_closed(self) -> bool:
        """Return whether the event source has closed."""

        return self._event_session_closed

    @property
    def frames_this_match(self) -> int:
        return self._frames_this_match

    def _new_results_since(
        self,
        previous_count: int,
    ) -> list[RecordedMatchResult]:
        return list(
            self.recorder.completed_results[
                previous_count:
            ]
        )

    def step(
        self,
    ) -> tuple[RecordedMatchResult, ...]:
        """Process events and capture at most one frame."""

        self.input_capture.poll()

        previous_active = (
            self.recorder.active
        )

        previous_result_count = len(
            self.recorder.completed_results
        )

        events = (
            self.event_cursor.read_new_events()
        )

        if any(
            event.type == "session_ended"
            for event in events
        ):
            self._event_session_closed = True

        self.recorder.process_events(
            events
        )

        results = self._new_results_since(
            previous_result_count
        )

        now = self._monotonic()

        if (
            self.recorder.active
            and not previous_active
        ):
            self._frames_this_match = 0
            self._next_frame_at = now
            self._pending_game_events.clear()

        if self.recorder.active:
            self._pending_game_events.extend(
                event
                for event in events
                if event.type
                not in _TRANSITION_IGNORED_EVENT_TYPES
            )

        if (
            results
            and not self.recorder.active
        ):
            self._frames_this_match = 0
            self.input_capture.reset()
            self._pending_game_events.clear()

        if self.recorder.active:
            frame_limit_reached = (
                self.max_frames_per_match > 0
                and self._frames_this_match
                >= self.max_frames_per_match
            )

            if frame_limit_reached:
                result = self.recorder.finalize(
                    status="interrupted",
                    outcome="frame_limit_reached",
                )

                if result is not None:
                    results.append(result)

                self._frames_this_match = 0
                self.input_capture.reset()
                self._pending_game_events.clear()

            elif now >= self._next_frame_at:
                human_input = (
                    self.input_capture.snapshot()
                )

                frame = (
                    self.frame_capture.capture_rgb(
                        self.window
                    )
                )

                telemetry_snapshot = (
                    self.telemetry_source.read_latest()
                    if self.telemetry_source is not None
                    else None
                )

                self.recorder.record_frame(
                    frame,
                    human_input.command,
                    pressed_keycodes=(
                        human_input.pressed_keycodes
                    ),
                    pressed_buttons=(
                        human_input.pressed_buttons
                    ),
                    timestamp_ns=(
                        human_input.timestamp_ns
                    ),
                    raw_event_count=(
                        human_input.raw_event_count
                    ),
                    weapon_previous_event_count=(
                        human_input
                        .weapon_previous_event_count
                    ),
                    weapon_next_event_count=(
                        human_input
                        .weapon_next_event_count
                    ),
                    telemetry=(
                        telemetry_snapshot.telemetry
                        if telemetry_snapshot is not None
                        else None
                    ),
                    telemetry_sync=(
                        telemetry_snapshot.to_sync_record()
                        if telemetry_snapshot is not None
                        else None
                    ),
                    game_events=tuple(
                        self._pending_game_events
                    ),
                )

                self._pending_game_events.clear()
                self._frames_this_match += 1

                progress_due = (
                    self.progress_every_frames > 0
                    and (
                        self._frames_this_match
                        % self.progress_every_frames
                        == 0
                        or (
                            self.max_frames_per_match
                            > 0
                            and self._frames_this_match
                            == self.max_frames_per_match
                        )
                    )
                )

                if (
                    progress_due
                    and self.progress_callback
                    is not None
                ):
                    self.progress_callback(
                        self._frames_this_match,
                        self.max_frames_per_match,
                    )

                capture_completed_at = (
                    self._monotonic()
                )

                self._next_frame_at = max(
                    self._next_frame_at
                    + self.frame_period_seconds,
                    capture_completed_at,
                )

        return tuple(results)

    def _next_sleep_seconds(self) -> float:
        """Return a bounded sleep until the next useful poll."""

        if not self.recorder.active:
            return self.poll_interval_seconds

        remaining = (
            self._next_frame_at
            - self._monotonic()
        )

        if remaining <= 0.0:
            return 0.0

        return min(
            self.poll_interval_seconds,
            remaining,
        )

    def run(
        self,
        *,
        max_completed_matches: int = 0,
    ) -> tuple[RecordedMatchResult, ...]:
        """Run until interrupted or the match limit is reached."""

        if max_completed_matches < 0:
            raise ValueError(
                "max_completed_matches must not "
                "be negative"
            )

        results: list[
            RecordedMatchResult
        ] = []

        while (
            max_completed_matches == 0
            or len(results)
            < max_completed_matches
        ):
            results.extend(
                self.step()
            )

            if (
                max_completed_matches > 0
                and len(results)
                >= max_completed_matches
            ):
                break

            if self._event_session_closed:
                break

            sleep_seconds = (
                self._next_sleep_seconds()
            )

            if sleep_seconds > 0.0:
                self._sleeper(
                    sleep_seconds
                )

        return tuple(results)
