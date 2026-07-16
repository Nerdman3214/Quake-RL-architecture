#!/usr/bin/env python3
"""Launch the event bridge and write a JSONL stream of engine messages."""

import subprocess
import os
import sys
from pathlib import Path
import time

# Make the package importable when run from the repository root.
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[4]
    src_root = Path(__file__).resolve().parents[3]
    for candidate in (repo_root, src_root):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

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

        print("Xonotic launched with PID:", proc.pid)

        # Simulate reading events from Xonotic (in practice, this would read real engine output)
        run_event_stream()

        time.sleep(3)  # Give time for events to be recorded

        proc.terminate()
        print("Recording finished.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()