import json
from pathlib import Path
from typing import Iterator, Dict, Any


class JSONLReader:
    """
    Reads structured game events.

    Does NOT:
    - modify events
    - create observations
    - calculate rewards
    """


    def __init__(self, filepath: str):

        self.path = Path(filepath)

        if not self.path.exists():
            raise FileNotFoundError(
                filepath
            )


    def read_events(
        self
    ) -> Iterator[Dict[str, Any]]:

        with open(
            self.path,
            "r",
            encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(file, 1):

                line = line.strip()

                if not line:
                    continue

                try:

                    yield json.loads(line)

                except json.JSONDecodeError as e:

                    raise ValueError(
                        f"Invalid JSON on line {line_number}"
                    ) from e