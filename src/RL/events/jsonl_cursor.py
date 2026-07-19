"""Incremental cursor for append-only event JSONL files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from RL.events.contracts import Event


class JSONLEventCursor:
    """Read each complete appended event exactly once.

    The cursor records a byte offset and the original file identity.
    File replacement or truncation raises an error instead of silently
    replaying old events or skipping data.
    """

    def __init__(
        self,
        filepath: str | Path,
        *,
        start_at_end: bool = True,
    ) -> None:
        self.path = Path(filepath)

        if not self.path.exists():
            raise FileNotFoundError(str(self.path))

        if not self.path.is_file():
            raise ValueError(
                "event path must identify a regular file"
            )

        stat = self.path.stat()

        self._identity = (
            stat.st_dev,
            stat.st_ino,
        )

        self._offset = (
            stat.st_size
            if start_at_end
            else 0
        )

    @property
    def offset(self) -> int:
        """Return the next unread byte position."""

        return self._offset

    def _validate_stat(
        self,
        stat: os.stat_result,
    ) -> None:
        identity = (
            stat.st_dev,
            stat.st_ino,
        )

        if identity != self._identity:
            raise RuntimeError(
                "event file was replaced"
            )

        if stat.st_size < self._offset:
            raise RuntimeError(
                "event file was truncated"
            )

    @staticmethod
    def _event_from_line(
        raw_line: bytes,
        *,
        byte_offset: int,
    ) -> Event:
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "Invalid UTF-8 event data at byte "
                f"{byte_offset}"
            ) from error

        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Invalid JSON event at byte "
                f"{byte_offset}"
            ) from error

        if not isinstance(payload, dict):
            raise ValueError(
                "Invalid event payload at byte "
                f"{byte_offset}"
            )

        event_type = payload.get("type")
        event_data = payload.get("data", {})

        if not isinstance(event_type, str):
            raise ValueError(
                "Invalid event type at byte "
                f"{byte_offset}"
            )

        if not isinstance(event_data, dict):
            raise ValueError(
                "Invalid event data at byte "
                f"{byte_offset}"
            )

        return Event(
            type=event_type,
            data=event_data,
        )

    def seek_to_end(self) -> None:
        """Mark all currently written bytes as consumed."""

        with self.path.open("rb") as file:
            stat = os.fstat(file.fileno())
            self._validate_stat(stat)
            self._offset = stat.st_size

    def read_new_events(self) -> tuple[Event, ...]:
        """Return complete events appended after the cursor."""

        events: list[Event] = []

        try:
            file = self.path.open("rb")
        except FileNotFoundError as error:
            raise RuntimeError(
                "event file is no longer available"
            ) from error

        with file:
            stat = os.fstat(file.fileno())
            self._validate_stat(stat)

            file.seek(self._offset)

            while True:
                line_start = file.tell()
                raw_line = file.readline()

                if not raw_line:
                    break

                # The writer may currently be producing this line.
                # Leave it unread until its terminating newline exists.
                if not raw_line.endswith(b"\n"):
                    break

                next_offset = file.tell()

                if not raw_line.strip():
                    self._offset = next_offset
                    continue

                event = self._event_from_line(
                    raw_line,
                    byte_offset=line_start,
                )

                events.append(event)
                self._offset = next_offset

        return tuple(events)
