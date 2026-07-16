#!/usr/bin/env python3
"""Smoke test the event JSONL writer and reader pipeline."""

import sys
import tempfile
from pathlib import Path

# Allow running directly from the repository root or via PYTHONPATH=src.
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[4]
    src_root = Path(__file__).resolve().parents[3]
    for candidate in (repo_root, src_root):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

try:
    from src.RL.events import Event, JSONLReader, JSONLWriter
except ModuleNotFoundError:
    from RL.events import Event, JSONLReader, JSONLWriter


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="rl-events-", dir=".") as tmpdir:
        path = Path(tmpdir) / "events.jsonl"

        writer = JSONLWriter(str(path))
        writer.write_event(
            Event(type="spawn", data={"player": 1, "health": 100})
        )
        writer.write_event(
            Event(type="move", data={"x": 12, "y": 4})
        )
        writer.close()

        reader = JSONLReader(str(path))
        events = list(reader.read_events())

        expected = [
            ("spawn", {"player": 1, "health": 100}),
            ("move", {"x": 12, "y": 4}),
        ]

        if len(events) != len(expected):
            print("FAIL: unexpected event count")
            print(f"  expected={len(expected)} actual={len(events)}")
            return 1

        for index, (event, (expected_type, expected_data)) in enumerate(
            zip(events, expected), start=1
        ):
            if event.type != expected_type or event.data != expected_data:
                print(f"FAIL: event {index} mismatch")
                print(
                    "  expected_type="
                    f"{expected_type} actual_type={event.type}"
                )
                print(
                    "  expected_data="
                    f"{expected_data} actual_data={event.data}"
                )
                return 1

        print("PASS: event writer/reader pipeline verified")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
