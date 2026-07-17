#!/usr/bin/env python3
"""Launch a real Xonotic server and record normalized events as JSONL."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from RL.engine.server.unified_event_reader import (
    XonoticUnifiedEventReader,
)
from RL.env.multiplayer import normalize_game_mode
from RL.events import Event, JSONLWriter


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
XONOTIC_ROOT = Path("/media/steven/WINPE2/Xonotic")
SERVER_EXECUTABLE = XONOTIC_ROOT / "xonotic-linux64-dedicated"

USER_DATA_DIRECTORY = Path.home() / ".xonotic" / "data"
SERVER_CONFIG_NAME = "rl_event_recorder.cfg"

DEFAULT_PORT = 26000
DEFAULT_MODE = "dm"
DEFAULT_MAP = "boil"
DEFAULT_MAX_PLAYERS = 4
DEFAULT_BOT_COUNT = 3
DEFAULT_BOT_SKILL = 4


def game_mode_argument(value: str) -> str:
    """Normalize a game-mode name to Xonotic's short identifier."""

    try:
        return normalize_game_mode(value).value
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def nonnegative_integer(value: str) -> int:
    """Parse a command-line integer that must not be negative."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, received {value!r}"
        ) from error

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must not be negative"
        )

    return parsed


def positive_integer(value: str) -> int:
    """Parse a command-line integer that must be positive."""

    parsed = nonnegative_integer(value)

    if parsed == 0:
        raise argparse.ArgumentTypeError(
            "value must be greater than zero"
        )

    return parsed


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse live-recorder command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch a local Xonotic dedicated server and record "
            "its normalized gameplay events."
        )
    )

    parser.add_argument(
        "--mode",
        type=game_mode_argument,
        default=DEFAULT_MODE,
        help=(
            "Game mode or alias, such as dm, ctf, dom, kh, "
            "deathmatch, or capture-the-flag."
        ),
    )
    parser.add_argument(
        "--map",
        dest="map_name",
        default=DEFAULT_MAP,
        help="Installed Xonotic map name.",
    )
    parser.add_argument(
        "--port",
        type=positive_integer,
        default=DEFAULT_PORT,
        help=f"UDP server port. Default: {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--max-players",
        type=positive_integer,
        default=DEFAULT_MAX_PLAYERS,
        help=(
            "Maximum players including the controlled client. "
            f"Default: {DEFAULT_MAX_PLAYERS}."
        ),
    )
    parser.add_argument(
        "--bots",
        type=nonnegative_integer,
        default=DEFAULT_BOT_COUNT,
        help=f"Number of bots. Default: {DEFAULT_BOT_COUNT}.",
    )
    parser.add_argument(
        "--skill",
        type=nonnegative_integer,
        default=DEFAULT_BOT_SKILL,
        help=f"Bot skill value. Default: {DEFAULT_BOT_SKILL}.",
    )

    args = parser.parse_args(argv)

    if not args.map_name.strip():
        parser.error("--map must not be blank")

    if args.bots >= args.max_players:
        parser.error(
            "--bots must be less than --max-players so the "
            "controlled client has an available slot"
        )

    return args


def udp_port_available(port: int) -> bool:
    """Return whether a UDP port can currently be bound."""

    test_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    try:
        test_socket.bind(("0.0.0.0", port))
    except OSError:
        return False
    finally:
        test_socket.close()

    return True


def stop_process(
    process: subprocess.Popen[str],
) -> None:
    """Terminate the server without leaving an orphan process."""

    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def write_server_config(
    *,
    mode: str,
    map_name: str,
    directory: Path = USER_DATA_DIRECTORY,
) -> Path:
    """Write the mode/map configuration loaded by Xonotic."""

    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / SERVER_CONFIG_NAME

    config_path.write_text(
        (
            f'gametype "{mode}"\n'
            f'g_maplist "{map_name}"\n'
            'g_maplist_shuffle 0\n'
            'g_maplist_votable 0\n'
            f'map "{map_name}"\n'
        ),
        encoding="utf-8",
    )

    return config_path


def build_command(
    *,
    server_config_name: str,
    port: int,
    max_players: int,
    bot_count: int,
    bot_skill: int,
) -> list[str]:
    """Build the dedicated-server command."""

    return [
        str(SERVER_EXECUTABLE),
        "+set",
        "sv_master",
        "0",
        "+set",
        "port",
        str(port),
        "+set",
        "sv_hostname",
        "RL_Event_Recorder_Test",
        "+maxplayers",
        str(max_players),
        "+set",
        "bot_number",
        str(bot_count),
        "+set",
        "skill",
        str(bot_skill),
        "+set",
        "rcon_password",
        "noobnog_rl_test",
        "+set",
        "sv_eventlog",
        "1",
        "+set",
        "sv_eventlog_console",
        "1",
        "+set",
        "sv_eventlog_files",
        "0",
        "+set",
        "sv_logscores_console",
        "1",
        "+set",
        "sv_logscores_bots",
        "1",
        "+serverconfig",
        server_config_name,
    ]


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Run one live event-recording session."""

    args = parse_args(argv)

    if not SERVER_EXECUTABLE.is_file():
        print(
            f"ERROR: dedicated server not found: "
            f"{SERVER_EXECUTABLE}"
        )
        return 1

    if not udp_port_available(args.port):
        print(
            f"ERROR: UDP port {args.port} is already in use."
        )
        print(
            "Stop the existing Xonotic server before "
            "starting this test."
        )
        return 1

    session_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_directory = (
        REPOSITORY_ROOT / "data" / "xonotic_events"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory / f"session_{session_id}.jsonl"
    )

    server_config_path = write_server_config(
        mode=args.mode,
        map_name=args.map_name,
    )

    command = build_command(
        server_config_name=server_config_path.name,
        port=args.port,
        max_players=args.max_players,
        bot_count=args.bots,
        bot_skill=args.skill,
    )

    print("=" * 72)
    print("XONOTIC LIVE EVENT RECORDER")
    print("=" * 72)
    print(f"Server executable: {SERVER_EXECUTABLE}")
    print(f"Working directory: {XONOTIC_ROOT}")
    print(f"Mode: {args.mode}")
    print(f"Map: {args.map_name}")
    print(f"Server config: {server_config_path}")
    print(f"Port: {args.port}")
    print(f"Maximum players: {args.max_players}")
    print(f"Bots: {args.bots}")
    print(f"Bot skill: {args.skill}")
    print(f"JSONL output: {output_path}")
    print()
    print(
        "Start the graphical client in a second terminal "
        "after the server reports that it is listening."
    )
    print(
        f"Connect the client to 127.0.0.1:{args.port}."
    )
    print(
        "Press Ctrl+C here when the match is finished."
    )
    print("=" * 72)

    writer = JSONLWriter(str(output_path))
    reader = XonoticUnifiedEventReader()

    process: Optional[subprocess.Popen[str]] = None
    record_count = 0
    parsed_event_count = 0

    writer.write_event(
        Event(
            type="session_started",
            data={
                "session_id": session_id,
                "captured_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "source": "live_event_recording_test",
                "server_executable": str(
                    SERVER_EXECUTABLE
                ),
                "working_directory": str(XONOTIC_ROOT),
                "mode": args.mode,
                "map": args.map_name,
                "server_config": str(
                    server_config_path
                ),
                "port": args.port,
                "max_players": args.max_players,
                "bot_count": args.bots,
                "bot_skill": args.skill,
                "command": command,
            },
        )
    )
    record_count += 1

    try:
        process = subprocess.Popen(
            command,
            cwd=str(XONOTIC_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        print(f"Dedicated server PID: {process.pid}")

        if process.stdout is None:
            raise RuntimeError(
                "Dedicated-server stdout pipe "
                "was not created"
            )

        for raw_line in process.stdout:
            clean_display = reader.strip_ansi(
                raw_line
            ).rstrip("\r\n")

            if clean_display:
                print(f"SERVER: {clean_display}")

            event = reader.parse_line(raw_line)

            if event is None:
                continue

            writer.write_event(event)
            record_count += 1

            if event.type != "console_line":
                parsed_event_count += 1
                print(f"  -> EVENT: {event.type}")

        return_code = process.wait()

        if return_code != 0:
            print(
                "WARNING: dedicated server exited "
                f"with code {return_code}"
            )
            return 1

        return 0

    except KeyboardInterrupt:
        print("\nStopping live recording...")
        return 0

    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    finally:
        if process is not None:
            stop_process(process)

        writer.write_event(
            Event(
                type="session_ended",
                data={
                    "session_id": session_id,
                    "captured_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "records_before_session_end": (
                        record_count
                    ),
                    "parsed_gameplay_events": (
                        parsed_event_count
                    ),
                    "server_return_code": (
                        process.returncode
                        if process is not None
                        else None
                    ),
                },
            )
        )
        writer.close()

        print()
        print("=" * 72)
        print("RECORDING SUMMARY")
        print("=" * 72)
        print(f"Output file: {output_path}")
        print(
            "Records before session end: "
            f"{record_count}"
        )
        print(
            "Recognized gameplay events: "
            f"{parsed_event_count}"
        )
        print("Session-ended record written.")


if __name__ == "__main__":
    raise SystemExit(main())
