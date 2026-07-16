import json
import threading
from pathlib import Path
from typing import Dict, Any


class JSONLWriter:
    """
    Writes structured game events to JSON Lines format.

    Responsibility:
    - Serialize events
    - Persist events

    Does NOT:
    - Create events
    - Add timestamps
    - Compute rewards
    """

    def __init__(self, filepath: str):
        self.path = Path(filepath)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.file = open(
            self.path,
            "a",
            encoding="utf-8"
        )

        self.lock = threading.Lock()


    def write_event(
        self,
        event: Dict[str, Any]
    ) -> None:

        if not isinstance(event, dict):
            raise TypeError(
                "Events must be dictionaries"
            )

        line = json.dumps(
            event,
            separators=(",", ":"),
            ensure_ascii=False
        )

        with self.lock:
            self.file.write(
                line + "\n"
            )
            self.file.flush()


    def close(self):
        with self.lock:
            self.file.close()