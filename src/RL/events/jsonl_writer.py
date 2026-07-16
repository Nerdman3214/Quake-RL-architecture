"""Write events to JSON Lines files."""

import json
import threading
from pathlib import Path

from .contracts import Event
from .schema import to_payload


class JSONLWriter:
    """Write events to JSON Lines files."""

    def __init__(self, filepath: str):
        self.path = Path(filepath)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write_event(self, event: Event) -> None:
        if not isinstance(event, Event):
            raise TypeError("Events must be Event instances")

        line = json.dumps(
            to_payload(event),
            separators=(",", ":"),
            ensure_ascii=False,
        )

        with self.lock:
            self.file.write(line + "\n")
            self.file.flush()

    def close(self) -> None:
        with self.lock:
            self.file.close()