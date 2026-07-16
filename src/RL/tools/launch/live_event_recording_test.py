#!/usr/bin/env python3
"""Launch the event bridge and write a JSONL stream of engine messages."""

import subprocess
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from RL.engine.bridge.event_stream import run_event_stream

def main():
    print("Starting Xonotic and recording events...")
    try:
        # Launch Xonotic in headless mode (adjust path as needed)
        proc = subprocess.Popen(
            ["xonotic", "-dedicated", "-port", "27960"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # Simulate reading events from Xonotic (in practice, this would read real engine output)
        run_event_stream()

        proc.terminate()
        print("Recording finished.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# Make the package importable when run from the repository root.
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[4]
    src_root = Path(__file__).resolve().parents[3]
    for candidate in (repo_root, src_root):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

from RL.engine.bridge.event_stream import EventStream


def main() -> int:
    output_path = Path("events.jsonl")
    stream = EventStream(output_path=str(output_path))

    sample_messages = [
        {"event": "spawn", "tick": 1, "payload": {"player": 1}},
        {"event": "move", "tick": 2, "payload": {"x": 10, "y": 20}},
        {"event": "fire", "tick": 3, "payload": {"weapon": "rocket"}},
    ]

    stream.emit_many(sample_messages)
    stream.close()

    if output_path.exists():
        print(f"Recorded {len(sample_messages)} events to {output_path}")
        return 0

    print("FAIL: event log was not created")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
