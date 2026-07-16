"""Read events from JSON Lines files."""

import json
from pathlib import Path
from typing import Iterator

from .contracts import Event


class JSONLReader:
    """Read events from JSON Lines files."""

    def __init__(self, filepath: str):
        self.path = Path(filepath)
        if not self.path.exists():
            raise FileNotFoundError(filepath)

    def read_events(self) -> Iterator[Event]:
        with open(self.path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}"
                    ) from exc

                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Invalid event payload on line {line_number}"
                    )

                event_type = payload.get("type")
                event_data = payload.get("data", {})
                if not isinstance(event_type, str):
                    raise ValueError(
                        f"Invalid event type on line {line_number}"
                    )
                if not isinstance(event_data, dict):
                    raise ValueError(
                        f"Invalid event data on line {line_number}"
                    )

                yield Event(type=event_type, data=event_data)
