#!/usr/bin/env python3
"""Launch Xonotic and write a JSONL stream of engine messages."""

import subprocess
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[4]
src_root = Path(__file__).resolve().parents[3]

for candidate in (repo_root, src_root):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from RL.engine.bridge.event_stream import run_event_stream


def main():
    print("Starting Xonotic and recording events...")

    XONOTIC_PATH = "/media/steven/WINPE2/Xonotic/xonotic-linux64-glx"

    try:
        if not os.path.exists(XONOTIC_PATH):
            raise FileNotFoundError(
                f"Xonotic executable not found: {XONOTIC_PATH}"
            )

        if not os.access(XONOTIC_PATH, os.X_OK):
            raise PermissionError(
                f"Xonotic executable permission missing: {XONOTIC_PATH}"
            )

        proc = subprocess.Popen(
            [
                XONOTIC_PATH,
                "-dedicated",
                "-port",
                "27960"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        print(f"Xonotic started PID={proc.pid}")

        run_event_stream()

        proc.terminate()

        print("Recording finished.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
