#!/usr/bin/env python3
"""Launch a real Xonotic dedicated server and record stdout as JSONL."""

from __future__ import annotations

import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from RL.engine.server.unified_event_reader import XonoticUnifiedEventReader
from RL.events import Event, JSONLWriter


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
XONOTIC_ROOT = Path("/media/steven/WINPE2/Xonotic")
SERVER_EXECUTABLE = XONOTIC_ROOT / "xonotic-linux64-dedicated"

PORT = 26000
MAP_NAME = "boil"
MAX_PLAYERS = 4


def udp_port_available(port: int) -> bool:
    """Return whether a UDP port can currently be bound."""

    test_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        test_socket.bind(("0.0.0.0", port))
    except OSError:
        return False
    finally:
        test_socket.close()

    return True


def stop_process(process: subprocess.Popen[str]) -> None:
    """Terminate the dedicated server without leaving an orphan process."""

    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    if not SERVER_EXECUTABLE.is_file():
        print(f"ERROR: dedicated server not found: {SERVER_EXECUTABLE}")
        return 1

    if not udp_port_available(PORT):
        print(f"ERROR: UDP port {PORT} is already in use.")
        print("Stop the existing Xonotic server before starting this test.")
        return 1

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_directory = REPOSITORY_ROOT / "data" / "xonotic_events"
    output_path = output_directory / f"session_{session_id}.jsonl"

    command = [
        str(SERVER_EXECUTABLE),
        "+set",
        "sv_master",
        "0",
        "+set",
        "port",
        str(PORT),
        "+set",
        "sv_hostname",
        "RL_Event_Recorder_Test",
        "+maxplayers",
        str(MAX_PLAYERS),
        "+set",
        "bot_number",
        "3",
        "+set",
        "skill",
        "4",
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
        "+map",
        MAP_NAME,
    ]

    print("=" * 72)
    print("XONOTIC LIVE EVENT RECORDER")
    print("=" * 72)
    print(f"Server executable: {SERVER_EXECUTABLE}")
    print(f"Working directory: {XONOTIC_ROOT}")
    print(f"Port: {PORT}")
    print(f"Map: {MAP_NAME}")
    print(f"JSONL output: {output_path}")
    print()
    print("Start the graphical client in a second terminal after the server")
    print("reports that it is using port 26000.")
    print("Press Ctrl+C here when the match is finished.")
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
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "source": "live_event_recording_test",
                "server_executable": str(SERVER_EXECUTABLE),
                "working_directory": str(XONOTIC_ROOT),
                "port": PORT,
                "map": MAP_NAME,
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
            raise RuntimeError("Dedicated-server stdout pipe was not created")

        for raw_line in process.stdout:
            clean_display = reader.strip_ansi(raw_line)

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
            print(f"WARNING: dedicated server exited with code {return_code}")
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
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "records_before_session_end": record_count,
                    "parsed_gameplay_events": parsed_event_count,
                    "server_return_code": (
                        process.returncode if process is not None else None
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
        print(f"Records before session end: {record_count}")
        print(f"Recognized gameplay events: {parsed_event_count}")
        print("Session-ended record written.")


if __name__ == "__main__":
    raise SystemExit(main())
