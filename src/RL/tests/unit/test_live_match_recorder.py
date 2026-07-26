"""Tests for the live personal match-recorder adapter."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from RL.actions.composite import (
    CompositeActionCommand,
)
from RL.engine.client.x11_window import (
    X11Window,
)
from RL.events.contracts import Event
from RL.recording.live_match_recorder import (
    BufferedHumanInputState,
    CombinedHumanInputCapture,
    HumanInputSnapshot,
    LivePersonalMatchRecorder,
    build_xinput2_command,
)
from RL.recording.match_recorder import (
    AutomaticMatchRecorder,
    MatchRecorderConfig,
)
from RL.tools.recording import (
    record_personal_matches as cli,
)


KEY_PRESS = """EVENT type 13 (RawKeyPress)
    device: 3 (13)
    time:   286587137
    detail: 25
    valuators:
"""


class FakeCursor:
    def __init__(
        self,
        batches: list[tuple[Event, ...]],
    ) -> None:
        self.batches = deque(batches)

    def read_new_events(
        self,
    ) -> tuple[Event, ...]:
        if not self.batches:
            return ()

        return self.batches.popleft()


class FakeCapture:
    def __init__(self) -> None:
        self.calls = 0

    def capture_rgb(
        self,
        window: X11Window,
    ) -> np.ndarray:
        assert window.window_id == 123
        self.calls += 1

        return np.full(
            (120, 200, 3),
            20,
            dtype=np.uint8,
        )


class FakeInputCapture:
    def __init__(self) -> None:
        self.poll_calls = 0
        self.snapshot_calls = 0
        self.reset_calls = 0
        self.closed = False

    def poll(
        self,
        *,
        quiet_period_seconds: float = 0.002,
    ) -> int:
        self.poll_calls += 1
        return 0

    def snapshot(
        self,
        *,
        duration_ticks: int = 1,
    ) -> HumanInputSnapshot:
        self.snapshot_calls += 1

        return HumanInputSnapshot(
            command=CompositeActionCommand(
                forward_axis=1,
                fire=True,
                duration_ticks=duration_ticks,
            ),
            pressed_keycodes=(25,),
            pressed_buttons=(1,),
            timestamp_ns=123456789,
            raw_event_count=5,
        )

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.value += seconds


def started_event() -> Event:
    return Event(
        type="match_started",
        data={
            "match_id": "live-match-1",
            "game_mode": "tdm",
            "map_name": "fuse",
            "event_channel": (
                "structured_eventlog"
            ),
            "authority_tier": "primary",
        },
    )


def ended_event() -> Event:
    return Event(
        type="match_ended",
        data={
            "raw_line": ":gameover",
            "event_channel": (
                "structured_eventlog"
            ),
            "authority_tier": "primary",
        },
    )


def test_build_xinput2_command() -> None:
    assert build_xinput2_command(
        xinput_command="xinput",
        xi2_device="2",
    ) == (
        "xinput",
        "test-xi2",
        "--root",
        "2",
    )


def test_buffered_state_flushes_final_event() -> None:
    state = BufferedHumanInputState(
        keyboard_source_ids={13},
        mouse_source_ids={10},
        time_ns=lambda: 999,
    )

    for line in KEY_PRESS.splitlines(
        keepends=True
    ):
        state.feed_line(line)

    assert state.raw_event_count == 0

    assert state.flush_if_complete() == 1
    assert state.raw_event_count == 1

    snapshot = state.snapshot()

    assert snapshot.command.forward_axis == 1
    assert snapshot.pressed_keycodes == (25,)
    assert snapshot.timestamp_ns == 999


def test_runtime_records_one_complete_match(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
        )
    )

    cursor = FakeCursor(
        [
            (started_event(),),
            (ended_event(),),
        ]
    )

    capture = FakeCapture()
    human_input = FakeInputCapture()
    clock = FakeClock()

    runtime = LivePersonalMatchRecorder(
        recorder=recorder,
        event_cursor=cursor,
        frame_capture=capture,
        window=X11Window(
            window_id=123,
            title="Xonotic",
        ),
        input_capture=human_input,
        frame_rate=5.0,
        max_frames_per_match=10,
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
    )

    assert runtime.step() == ()
    assert recorder.active
    assert recorder.step_count == 1
    assert capture.calls == 1
    assert human_input.snapshot_calls == 1

    results = runtime.step()

    assert len(results) == 1
    assert results[0].status == "complete"
    assert results[0].step_count == 1
    assert not recorder.active
    assert human_input.reset_calls == 1


def test_runtime_frame_limit_preserves_partial_match(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
        )
    )

    cursor = FakeCursor(
        [
            (started_event(),),
            (),
        ]
    )

    capture = FakeCapture()
    human_input = FakeInputCapture()
    clock = FakeClock()

    runtime = LivePersonalMatchRecorder(
        recorder=recorder,
        event_cursor=cursor,
        frame_capture=capture,
        window=X11Window(
            window_id=123,
            title="Xonotic",
        ),
        input_capture=human_input,
        frame_rate=5.0,
        max_frames_per_match=1,
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
    )

    assert runtime.step() == ()
    assert recorder.step_count == 1

    clock.value = 1.0

    results = runtime.step()

    assert len(results) == 1
    assert results[0].status == (
        "interrupted"
    )
    assert results[0].outcome == (
        "frame_limit_reached"
    )
    assert not recorder.active


def test_cli_defaults_are_calibration_bounded() -> None:
    args = cli.parse_args(
        [
            "--event-path",
            "events.jsonl",
            "--controlled-player",
            "Noobnog",
        ]
    )

    assert args.frame_rate == 5.0
    assert args.max_matches == 1
    assert args.max_frames_per_match == 300
    assert args.start_current_match is False
    assert args.progress_every_frames == 10
    assert args.xi2_device == "2"
    assert args.keyboard_xi2_device == "3"
    assert args.save_raw_frames is False


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-1",
    ],
)
def test_cli_rejects_nonpositive_frame_rate(
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--event-path",
                "events.jsonl",
                "--controlled-player",
                "Noobnog",
                "--frame-rate",
                value,
            ]
        )


def test_cli_suppresses_broken_pipe(
    monkeypatch,
) -> None:
    def raise_broken_pipe(_):
        raise BrokenPipeError

    monkeypatch.setattr(
        cli,
        "main",
        raise_broken_pipe,
    )

    assert cli.run_cli([]) == 0


def test_cli_reports_runtime_error(
    monkeypatch,
    capsys,
) -> None:
    def raise_error(_):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(
        cli,
        "main",
        raise_error,
    )

    assert cli.run_cli([]) == 1

    captured = capsys.readouterr()

    assert "ERROR: capture failed" in (
        captured.err
    )

class FakeDeviceCapture:
    def __init__(
        self,
        snapshot_value: HumanInputSnapshot,
    ) -> None:
        self.snapshot_value = snapshot_value
        self.started = False
        self.closed = False
        self.reset_calls = 0

    def start(self) -> None:
        self.started = True

    def poll(
        self,
        *,
        quiet_period_seconds: float = 0.002,
    ) -> int:
        return self.snapshot_value.raw_event_count

    def snapshot(
        self,
        *,
        duration_ticks: int = 1,
    ) -> HumanInputSnapshot:
        return self.snapshot_value

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True


def test_combined_capture_merges_keyboard_and_mouse(
) -> None:
    keyboard = FakeDeviceCapture(
        HumanInputSnapshot(
            command=CompositeActionCommand(
                forward_axis=1,
                strafe_axis=-1,
                jump=True,
            ),
            pressed_keycodes=(25, 38, 65),
            pressed_buttons=(),
            timestamp_ns=100,
            raw_event_count=3,
        )
    )

    mouse = FakeDeviceCapture(
        HumanInputSnapshot(
            command=CompositeActionCommand(
                turn_delta_x=6.0,
                look_delta_y=-2.0,
                fire=True,
                weapon_delta=1,
            ),
            pressed_keycodes=(),
            pressed_buttons=(1,),
            timestamp_ns=200,
            raw_event_count=4,
        )
    )

    combined = CombinedHumanInputCapture(
        keyboard_capture=keyboard,
        mouse_capture=mouse,
    )

    combined.start()

    assert keyboard.started
    assert mouse.started
    assert combined.poll() == 7

    snapshot = combined.snapshot()

    assert snapshot.command.forward_axis == 1
    assert snapshot.command.strafe_axis == -1
    assert snapshot.command.turn_delta_x == 6.0
    assert snapshot.command.look_delta_y == -2.0
    assert snapshot.command.fire
    assert snapshot.command.jump
    assert snapshot.command.weapon_delta == 1
    assert snapshot.pressed_keycodes == (
        25,
        38,
        65,
    )
    assert snapshot.pressed_buttons == (1,)
    assert snapshot.timestamp_ns == 200
    assert snapshot.raw_event_count == 7

    combined.reset()

    assert keyboard.reset_calls == 1
    assert mouse.reset_calls == 1

    combined.close()

    assert keyboard.closed
    assert mouse.closed


def session_ended_event() -> Event:
    return Event(
        type="session_ended",
        data={
            "reason": "operator_interrupt",
        },
    )


def test_cli_detects_closed_event_session(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"

    event_path.write_text(
        '{"type":"session_started","data":{}}\n'
        '{"type":"player_connected","data":{}}\n'
        '{"type":"session_ended","data":{}}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="event session is already closed",
    ):
        cli._open_active_event_cursor(
            event_path
        )


def test_active_cursor_reads_session_end_appended_after_validation(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"

    event_path.write_text(
        '{"type":"session_started","data":{}}\n',
        encoding="utf-8",
    )

    (
        cursor,
        current_match_data,
    ) = cli._open_active_event_cursor(
        event_path
    )

    assert current_match_data is None

    with event_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            '{"type":"session_ended","data":{}}\n'
        )
        file.flush()

    events = cursor.read_new_events()

    assert [
        event.type
        for event in events
    ] == ["session_ended"]


def test_runtime_stops_when_event_session_closes(
    tmp_path: Path,
) -> None:
    class ClosedCursor:
        def __init__(self) -> None:
            self.calls = 0

        def read_new_events(
            self,
        ) -> tuple[Event, ...]:
            self.calls += 1

            if self.calls == 1:
                return (
                    session_ended_event(),
                )

            raise AssertionError(
                "runtime polled after session_ended"
            )

    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
        )
    )
    capture = FakeCapture()
    human_input = FakeInputCapture()
    clock = FakeClock()
    cursor = ClosedCursor()

    runtime = LivePersonalMatchRecorder(
        recorder=recorder,
        event_cursor=cursor,
        frame_capture=capture,
        window=X11Window(
            window_id=123,
            title="Xonotic",
        ),
        input_capture=human_input,
        frame_rate=5.0,
        max_frames_per_match=10,
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
    )

    assert runtime.run(
        max_completed_matches=1
    ) == ()
    assert runtime.event_session_closed
    assert cursor.calls == 1
    assert not recorder.active
    assert capture.calls == 0


def test_closed_session_preserves_active_partial_match(
    tmp_path: Path,
) -> None:
    class StartThenCloseCursor:
        def __init__(self) -> None:
            self.calls = 0

        def read_new_events(
            self,
        ) -> tuple[Event, ...]:
            self.calls += 1

            if self.calls == 1:
                return (started_event(),)

            if self.calls == 2:
                return (
                    session_ended_event(),
                )

            raise AssertionError(
                "runtime polled after session_ended"
            )

    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
        )
    )
    capture = FakeCapture()
    human_input = FakeInputCapture()
    clock = FakeClock()
    cursor = StartThenCloseCursor()

    runtime = LivePersonalMatchRecorder(
        recorder=recorder,
        event_cursor=cursor,
        frame_capture=capture,
        window=X11Window(
            window_id=123,
            title="Xonotic",
        ),
        input_capture=human_input,
        frame_rate=5.0,
        max_frames_per_match=10,
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
    )

    assert runtime.run(
        max_completed_matches=1
    ) == ()
    assert runtime.event_session_closed
    assert cursor.calls == 2
    assert recorder.active
    assert recorder.step_count == 1

    partial = recorder.close()

    assert partial is not None
    assert partial.status == "interrupted"
    assert partial.outcome == (
        "operator_interrupted"
    )
    assert partial.step_count == 1


def test_cli_accepts_current_match_options() -> None:
    args = cli.parse_args(
        [
            "--event-path",
            "events.jsonl",
            "--controlled-player",
            "Noobnog",
            "--start-current-match",
            "--progress-every-frames",
            "5",
        ]
    )

    assert args.start_current_match
    assert args.progress_every_frames == 5


def test_open_active_cursor_returns_current_match(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"

    event_path.write_text(
        '{"type":"session_started","data":{}}\n'
        '{"type":"match_started","data":'
        '{"match_id":"current-1",'
        '"game_mode":"dm","map_name":"boil"}}\n'
        '{"type":"player_connected","data":{}}\n',
        encoding="utf-8",
    )

    (
        cursor,
        current_match_data,
    ) = cli._open_active_event_cursor(
        event_path
    )

    assert cursor.read_new_events() == ()
    assert current_match_data is not None
    assert current_match_data["match_id"] == (
        "current-1"
    )
    assert current_match_data["game_mode"] == "dm"
    assert current_match_data["map_name"] == "boil"


def test_open_active_cursor_does_not_replay_ended_match(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"

    event_path.write_text(
        '{"type":"session_started","data":{}}\n'
        '{"type":"match_started","data":'
        '{"match_id":"ended-1"}}\n'
        '{"type":"match_ended","data":{}}\n',
        encoding="utf-8",
    )

    (
        _cursor,
        current_match_data,
    ) = cli._open_active_event_cursor(
        event_path
    )

    assert current_match_data is None


def test_runtime_records_prestarted_current_match_and_reports_progress(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
        )
    )

    recorder.start_match(
        started_event().data
    )

    cursor = FakeCursor(
        [
            (),
            (),
            (),
            (),
        ]
    )
    capture = FakeCapture()
    human_input = FakeInputCapture()
    clock = FakeClock()
    progress: list[tuple[int, int]] = []

    runtime = LivePersonalMatchRecorder(
        recorder=recorder,
        event_cursor=cursor,
        frame_capture=capture,
        window=X11Window(
            window_id=123,
            title="Xonotic",
        ),
        input_capture=human_input,
        frame_rate=5.0,
        max_frames_per_match=3,
        progress_every_frames=2,
        progress_callback=(
            lambda current, limit: progress.append(
                (current, limit)
            )
        ),
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
    )

    assert runtime.step() == ()

    clock.value = 1.0
    assert runtime.step() == ()

    clock.value = 2.0
    assert runtime.step() == ()

    assert recorder.step_count == 3
    assert progress == [
        (2, 3),
        (3, 3),
    ]

    clock.value = 3.0
    results = runtime.step()

    assert len(results) == 1
    assert results[0].outcome == (
        "frame_limit_reached"
    )
    assert results[0].step_count == 3
    assert not recorder.active


def test_run_sleeps_to_frame_deadlines(
    tmp_path: Path,
) -> None:
    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
        )
    )

    recorder.start_match(
        started_event().data
    )

    cursor = FakeCursor(
        [
            (),
            (),
            (),
            (),
        ]
    )
    capture = FakeCapture()
    human_input = FakeInputCapture()
    clock = FakeClock()

    runtime = LivePersonalMatchRecorder(
        recorder=recorder,
        event_cursor=cursor,
        frame_capture=capture,
        window=X11Window(
            window_id=123,
            title="Xonotic",
        ),
        input_capture=human_input,
        frame_rate=60.0,
        max_frames_per_match=3,
        poll_interval_seconds=1.0,
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
    )

    results = runtime.run(
        max_completed_matches=1
    )

    assert len(results) == 1
    assert results[0].step_count == 3
    assert results[0].outcome == (
        "frame_limit_reached"
    )
    assert capture.calls == 3

    assert clock.sleep_calls == pytest.approx(
        [
            1.0 / 60.0,
            1.0 / 60.0,
            1.0 / 60.0,
        ]
    )


def test_late_frame_deadline_uses_completion_time(
    tmp_path: Path,
) -> None:
    class SlowCapture(FakeCapture):
        def __init__(
            self,
            clock: FakeClock,
        ) -> None:
            super().__init__()
            self.clock = clock

        def capture_rgb(
            self,
            window: X11Window,
        ) -> np.ndarray:
            self.clock.value += 0.030

            return super().capture_rgb(
                window
            )

    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
        )
    )

    recorder.start_match(
        started_event().data
    )

    clock = FakeClock()
    capture = SlowCapture(clock)

    runtime = LivePersonalMatchRecorder(
        recorder=recorder,
        event_cursor=FakeCursor([()]),
        frame_capture=capture,
        window=X11Window(
            window_id=123,
            title="Xonotic",
        ),
        input_capture=FakeInputCapture(),
        frame_rate=60.0,
        max_frames_per_match=10,
        poll_interval_seconds=0.01,
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
    )

    assert runtime.step() == ()

    assert runtime._next_frame_at == (
        pytest.approx(0.030)
    )
    assert runtime._next_sleep_seconds() == 0.0

    partial = recorder.close()

    assert partial is not None
    assert partial.step_count == 1


def test_combined_capture_sums_wheel_event_counts(
) -> None:
    keyboard = FakeDeviceCapture(
        HumanInputSnapshot(
            command=CompositeActionCommand(
                forward_axis=1,
                weapon_delta=-1,
            ),
            pressed_keycodes=(25,),
            pressed_buttons=(),
            timestamp_ns=100,
            raw_event_count=3,
            weapon_previous_event_count=2,
            weapon_next_event_count=1,
        )
    )

    mouse = FakeDeviceCapture(
        HumanInputSnapshot(
            command=CompositeActionCommand(
                turn_delta_x=2.0,
                weapon_delta=1,
            ),
            pressed_keycodes=(),
            pressed_buttons=(),
            timestamp_ns=200,
            raw_event_count=4,
            weapon_previous_event_count=3,
            weapon_next_event_count=5,
        )
    )

    combined = CombinedHumanInputCapture(
        keyboard_capture=keyboard,
        mouse_capture=mouse,
    )

    snapshot = combined.snapshot()

    assert snapshot.command.weapon_delta == 1
    assert snapshot.raw_event_count == 7
    assert (
        snapshot.weapon_previous_event_count
        == 5
    )
    assert snapshot.weapon_next_event_count == 6


def test_recording_cli_defaults_to_legacy_policy_frame_storage(
) -> None:
    args = cli.parse_args(
        [
            "--event-path",
            "events.jsonl",
            "--controlled-player",
            "Noobnog",
        ]
    )

    assert (
        args.policy_frame_storage_mode
        == "legacy_float32_stack"
    )


def test_recording_cli_accepts_compact_policy_frame_storage(
) -> None:
    args = cli.parse_args(
        [
            "--event-path",
            "events.jsonl",
            "--controlled-player",
            "Noobnog",
            "--policy-frame-storage-mode",
            "compact_uint8_frame",
        ]
    )

    assert (
        args.policy_frame_storage_mode
        == "compact_uint8_frame"
    )



def test_runtime_attaches_new_game_events_to_next_transition(
    tmp_path: Path,
) -> None:
    import json

    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
        )
    )

    recorder.start_match(
        started_event().data
    )

    pickup_event = Event(
        type="item_pickup",
        data={
            "sequence": 42,
            "player_name": "Noobnog",
            "item": "Strength",
        },
    )

    clock = FakeClock()

    runtime = LivePersonalMatchRecorder(
        recorder=recorder,
        event_cursor=FakeCursor(
            [
                (pickup_event,),
            ]
        ),
        frame_capture=FakeCapture(),
        window=X11Window(
            window_id=123,
            title="Xonotic",
        ),
        input_capture=FakeInputCapture(),
        frame_rate=60.0,
        max_frames_per_match=10,
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
    )

    assert runtime.step() == ()
    assert recorder.step_count == 1

    result = recorder.close()

    assert result is not None

    records = [
        json.loads(line)
        for line in result.episode_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    transition = next(
        record["data"]
        for record in records
        if record["type"] == "agent_transition"
    )

    assert (
        transition["info"]["game_event_count"]
        == 1
    )

    assert transition["info"]["game_events"] == [
        {
            "type": "item_pickup",
            "data": {
                "sequence": 42,
                "player_name": "Noobnog",
                "item": "Strength",
            },
        }
    ]



def test_runtime_attaches_latest_telemetry_to_transition(
    tmp_path: Path,
) -> None:
    import json

    from RL.observations.contracts import (
        PlayerTelemetry,
    )
    from RL.recording.xonotic_telemetry_source import (
        XonoticTelemetrySnapshot,
    )

    class FakeTelemetrySource:
        def __init__(self) -> None:
            self.calls = 0

        def read_latest(
            self,
        ) -> XonoticTelemetrySnapshot:
            self.calls += 1

            return XonoticTelemetrySnapshot(
                telemetry=PlayerTelemetry(
                    health=87,
                    armor=25,
                    ammo=14,
                    weapon="shotgun",
                    alive=True,
                    score=3,
                    match_time_seconds=12.5,
                ),
                sample_time=20.25,
                age_seconds=0.015,
                fresh=True,
                weapon_id=2,
                ammo_stat=6,
                spectatee_status=0,
                current_player=0,
                pre_spawn=False,
                source_path=Path(
                    "rl_telemetry.jsonl"
                ),
            )

    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=tmp_path / "matches",
            controlled_player="Noobnog",
        )
    )

    recorder.start_match(
        started_event().data
    )

    telemetry_source = FakeTelemetrySource()
    clock = FakeClock()

    runtime = LivePersonalMatchRecorder(
        recorder=recorder,
        event_cursor=FakeCursor([()]),
        frame_capture=FakeCapture(),
        window=X11Window(
            window_id=123,
            title="Xonotic",
        ),
        input_capture=FakeInputCapture(),
        frame_rate=60.0,
        telemetry_source=telemetry_source,
        max_frames_per_match=10,
        monotonic=clock.monotonic,
        sleeper=clock.sleep,
    )

    assert runtime.step() == ()
    assert telemetry_source.calls == 1

    result = recorder.close()

    assert result is not None

    records = [
        json.loads(line)
        for line in result.episode_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    transition = next(
        record["data"]
        for record in records
        if record["type"] == "agent_transition"
    )

    assert transition[
        "observation"
    ]["telemetry"] == {
        "health": 87,
        "armor": 25,
        "ammo": 14,
        "weapon": "shotgun",
        "alive": True,
        "score": 3,
        "match_time_seconds": 12.5,
    }

    sync = transition[
        "info"
    ]["telemetry_sync"]

    assert sync["fresh"] is True
    assert sync["sample_time"] == 20.25
    assert sync["weapon_id"] == 2
    assert sync["ammo_stat"] == 6
    assert sync["pre_spawn"] is False



def test_recording_cli_accepts_telemetry_options(
) -> None:
    args = cli.parse_args(
        [
            "--event-path",
            "events.jsonl",
            "--controlled-player",
            "Noobnog",
            "--telemetry-path",
            "custom-telemetry.jsonl",
            "--telemetry-max-age-seconds",
            "0.5",
        ]
    )

    assert args.telemetry_path == Path(
        "custom-telemetry.jsonl"
    )
    assert (
        args.telemetry_max_age_seconds
        == 0.5
    )


def test_recording_cli_has_xonotic_telemetry_defaults(
) -> None:
    args = cli.parse_args(
        [
            "--event-path",
            "events.jsonl",
            "--controlled-player",
            "Noobnog",
        ]
    )

    assert args.telemetry_path == (
        Path.home()
        / ".xonotic"
        / "data"
        / "data"
        / "rl_telemetry.jsonl"
    )
    assert (
        args.telemetry_max_age_seconds
        == 0.25
    )
