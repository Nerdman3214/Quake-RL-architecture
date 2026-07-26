#!/usr/bin/env python3
"""Record numbered personal Xonotic matches automatically."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from RL.engine.client.x11_window import (
    X11WindowCapture,
)
from RL.events.jsonl_cursor import (
    JSONLEventCursor,
)
from RL.recording.live_match_recorder import (
    CombinedHumanInputCapture,
    LivePersonalMatchRecorder,
    XInput2ProcessCapture,
)
from RL.recording.match_recorder import (
    AutomaticMatchRecorder,
    COMPACT_POLICY_FRAME_STORAGE,
    LEGACY_POLICY_FRAME_STORAGE,
    MatchRecorderConfig,
)

from RL.recording.xonotic_telemetry_source import (
    XonoticTelemetrySource,
)

def positive_float(value: str) -> float:
    parsed = float(value)

    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "value must be positive"
        )

    return parsed


def nonnegative_integer(value: str) -> int:
    parsed = int(value)

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must not be negative"
        )

    return parsed


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for Xonotic match boundaries and record "
            "synchronized human input and policy frames."
        )
    )

    parser.add_argument(
        "--event-path",
        type=Path,
        required=True,
        help="Active append-only Xonotic event JSONL.",
    )

    parser.add_argument(
        "--telemetry-path",
        type=Path,
        default=(
            Path.home()
            / ".xonotic"
            / "data"
            / "data"
            / "rl_telemetry.jsonl"
        ),
        help=(
            "Append-only Xonotic CSQC telemetry "
            "JSONL. The file may be created after "
            "the recorder starts."
        ),
    )

    parser.add_argument(
        "--telemetry-max-age-seconds",
        type=positive_float,
        default=0.25,
        help=(
            "Maximum age for a telemetry sample to "
            "be marked fresh."
        ),
    )

    parser.add_argument(
        "--controlled-player",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/supervised/raw/"
            "personal_play/matches"
        ),
    )

    parser.add_argument(
        "--window-name-pattern",
        default="Xonotic",
    )

    parser.add_argument(
        "--xi2-device",
        default="2",
        help=(
            "Master pointer device used for mouse "
            "XInput2 events."
        ),
    )

    parser.add_argument(
        "--keyboard-xi2-device",
        default="3",
        help=(
            "Master keyboard device used for keyboard "
            "XInput2 events."
        ),
    )

    parser.add_argument(
        "--keyboard-source-id",
        action="append",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--mouse-source-id",
        action="append",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--frame-rate",
        type=positive_float,
        default=5.0,
    )

    parser.add_argument(
        "--max-matches",
        type=nonnegative_integer,
        default=1,
        help="Zero means no match limit.",
    )

    parser.add_argument(
        "--max-frames-per-match",
        type=nonnegative_integer,
        default=300,
        help="Zero means no frame limit.",
    )

    parser.add_argument(
        "--poll-interval-seconds",
        type=positive_float,
        default=0.01,
    )

    parser.add_argument(
        "--save-raw-frames",
        action="store_true",
    )

    parser.add_argument(
        "--policy-frame-storage-mode",
        choices=(
            LEGACY_POLICY_FRAME_STORAGE,
            COMPACT_POLICY_FRAME_STORAGE,
        ),
        default=LEGACY_POLICY_FRAME_STORAGE,
        help=(
            "Stored policy-frame representation. "
            "Use compact_uint8_frame for long "
            "recurrent demonstrations."
        ),
    )

    parser.add_argument(
        "--start-current-match",
        action="store_true",
        help=(
            "Immediately record the currently active "
            "authoritative match."
        ),
    )

    parser.add_argument(
        "--progress-every-frames",
        type=nonnegative_integer,
        default=10,
        help=(
            "Print capture progress every N frames. "
            "Zero disables progress output."
        ),
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true",
    )

    return parser.parse_args(argv)


def _open_active_event_cursor(
    event_path: Path,
) -> tuple[
    JSONLEventCursor,
    dict[str, object] | None,
]:
    """Validate the session and recover its active match."""

    cursor = JSONLEventCursor(
        event_path,
        start_at_end=False,
    )
    events = cursor.read_new_events()

    session_closed = False
    current_match_data: (
        dict[str, object] | None
    ) = None

    for event in events:
        if event.type == "session_started":
            session_closed = False
            current_match_data = None
            continue

        if event.type == "session_ended":
            session_closed = True
            current_match_data = None
            continue

        if (
            event.type == "match_started"
            and not session_closed
        ):
            current_match_data = dict(event.data)
            continue

        if event.type == "match_ended":
            current_match_data = None

    if session_closed:
        raise RuntimeError(
            "event session is already closed: "
            f"{event_path}"
        )

    # The same cursor remains positioned after all
    # existing complete records, eliminating a gap
    # between validation and live event tailing.
    return cursor, current_match_data


def _print_progress(
    frame_count: int,
    frame_limit: int,
) -> None:
    """Print visible live capture progress."""

    limit = (
        str(frame_limit)
        if frame_limit > 0
        else "unbounded"
    )

    print(
        "recording_progress_frames="
        f"{frame_count}/{limit}",
        flush=True,
    )


def _print_result(result: object) -> None:
    print(
        "recorded_match="
        f"{result.match_index} "
        f"status={result.status} "
        f"steps={result.step_count} "
        f"directory={result.directory}"
    )


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = parse_args(argv)

    event_path = args.event_path.resolve()
    telemetry_path = (
        args.telemetry_path
        .expanduser()
        .resolve()
    )

    if not event_path.is_file():
        raise FileNotFoundError(
            f"event file does not exist: "
            f"{event_path}"
        )

    (
        event_cursor,
        current_match_data,
    ) = _open_active_event_cursor(
        event_path
    )

    frame_capture = X11WindowCapture(
        window_name_pattern=(
            args.window_name_pattern
        )
    )

    window = frame_capture.find_window()

    print(f"event_path={event_path}")
    print(f"telemetry_path={telemetry_path}")
    print(
        "telemetry_path_exists="
        f"{telemetry_path.is_file()}"
    )
    print(
        "telemetry_max_age_seconds="
        f"{args.telemetry_max_age_seconds}"
    )
    print(
        f"window_id={window.window_id} "
        f"window_title={window.title}"
    )
    print(
        f"output_root="
        f"{args.output_root.resolve()}"
    )
    print(f"frame_rate={args.frame_rate}")
    print(
        "policy_frame_storage_mode="
        f"{args.policy_frame_storage_mode}"
    )
    print(
        "max_frames_per_match="
        f"{args.max_frames_per_match}"
    )
    print(f"max_matches={args.max_matches}")
    print(
        "start_current_match="
        f"{args.start_current_match}"
    )
    print(
        "progress_every_frames="
        f"{args.progress_every_frames}"
    )
    print(
        f"pointer_xi2_device="
        f"{args.xi2_device}"
    )
    print(
        f"keyboard_xi2_device="
        f"{args.keyboard_xi2_device}"
    )
    print(
        "keyboard_source_ids="
        f"{args.keyboard_source_id}"
    )
    print(
        "mouse_source_ids="
        f"{args.mouse_source_id}"
    )

    if args.preflight_only:
        print(
            "personal_match_recorder_preflight=passed"
        )
        return 0

    if (
        args.start_current_match
        and current_match_data is None
    ):
        raise RuntimeError(
            "no currently active authoritative match "
            "was found in the event session"
        )

    recorder = AutomaticMatchRecorder(
        MatchRecorderConfig(
            output_root=args.output_root,
            controlled_player=(
                args.controlled_player
            ),
            save_raw_frames=(
                args.save_raw_frames
            ),
            policy_frame_storage_mode=(
                args.policy_frame_storage_mode
            ),
        )
    )

    telemetry_source = XonoticTelemetrySource(
        telemetry_path,
        max_age_seconds=(
            args.telemetry_max_age_seconds
        ),
    )

    keyboard_capture = XInput2ProcessCapture(
        xi2_device=(
            args.keyboard_xi2_device
        ),
        keyboard_source_ids=(
            args.keyboard_source_id
        ),
        mouse_source_ids=(),
    )

    mouse_capture = XInput2ProcessCapture(
        xi2_device=args.xi2_device,
        keyboard_source_ids=(),
        mouse_source_ids=(
            args.mouse_source_id
        ),
    )

    input_capture = CombinedHumanInputCapture(
        keyboard_capture=keyboard_capture,
        mouse_capture=mouse_capture,
    )

    runtime = LivePersonalMatchRecorder(
        recorder=recorder,
        event_cursor=event_cursor,
        frame_capture=frame_capture,
        window=window,
        input_capture=input_capture,
        telemetry_source=telemetry_source,
        frame_rate=args.frame_rate,
        max_frames_per_match=(
            args.max_frames_per_match
        ),
        progress_every_frames=(
            args.progress_every_frames
        ),
        progress_callback=_print_progress,
        poll_interval_seconds=(
            args.poll_interval_seconds
        ),
    )

    completed = ()

    try:
        input_capture.start()

        print(
            "personal_match_recorder=armed"
        )

        if args.start_current_match:
            assert current_match_data is not None

            directory = recorder.start_match(
                current_match_data
            )

            print(
                "recording_current_match=True"
            )
            print(
                "active_recording_directory="
                f"{directory}"
            )
        else:
            print(
                "waiting_for_next_match_started=True"
            )

        completed = runtime.run(
            max_completed_matches=(
                args.max_matches
            )
        )
    finally:
        partial = recorder.close()
        input_capture.close()

        if partial is not None:
            _print_result(partial)

    for result in completed:
        _print_result(result)

    requested_match_count_not_reached = (
        args.max_matches == 0
        or len(completed) < args.max_matches
    )

    if (
        runtime.event_session_closed
        and requested_match_count_not_reached
    ):
        raise RuntimeError(
            "event session ended before the requested "
            "recording completed"
        )

    print(
        f"completed_match_count={len(completed)}"
    )
    print(
        "personal_match_recorder=completed"
    )

    return 0


def run_cli(
    argv: Sequence[str] | None = None,
) -> int:
    try:
        return main(argv)
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        print(
            "Recorder interrupted by operator.",
            file=sys.stderr,
        )
        return 130
    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
