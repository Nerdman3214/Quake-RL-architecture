"""Tests for incremental append-only JSONL event reading."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from RL.events import JSONLEventCursor


def event_line(
    event_type: str,
    **data: object,
) -> str:
    return json.dumps(
        {
            "type": event_type,
            "data": data,
        },
        separators=(",", ":"),
    )


def append_text(
    path: Path,
    text: str,
) -> None:
    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(text)
        file.flush()


def test_starts_at_end_and_reads_only_appended_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"

    path.write_text(
        event_line("old_event") + "\n",
        encoding="utf-8",
    )

    cursor = JSONLEventCursor(path)

    assert cursor.read_new_events() == ()

    append_text(
        path,
        event_line(
            "player_kill",
            killer="Noobnog",
        )
        + "\n",
    )

    events = cursor.read_new_events()

    assert len(events) == 1
    assert events[0].type == "player_kill"
    assert events[0].data["killer"] == "Noobnog"


def test_start_at_beginning_reads_existing_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"

    path.write_text(
        event_line("match_started") + "\n",
        encoding="utf-8",
    )

    cursor = JSONLEventCursor(
        path,
        start_at_end=False,
    )

    assert [
        event.type
        for event in cursor.read_new_events()
    ] == ["match_started"]

    assert cursor.read_new_events() == ()


def test_preserves_appended_event_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("", encoding="utf-8")

    cursor = JSONLEventCursor(path)

    append_text(
        path,
        event_line("first")
        + "\n"
        + event_line("second")
        + "\n",
    )

    assert [
        event.type
        for event in cursor.read_new_events()
    ] == [
        "first",
        "second",
    ]


def test_partial_last_line_waits_for_newline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"

    partial = event_line(
        "player_kill",
        killer="Noobnog",
    )

    path.write_text(
        partial,
        encoding="utf-8",
    )

    cursor = JSONLEventCursor(
        path,
        start_at_end=False,
    )

    assert cursor.read_new_events() == ()
    assert cursor.offset == 0

    append_text(path, "\n")

    events = cursor.read_new_events()

    assert len(events) == 1
    assert events[0].type == "player_kill"


def test_blank_lines_are_consumed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"

    path.write_text(
        "\n\n",
        encoding="utf-8",
    )

    cursor = JSONLEventCursor(
        path,
        start_at_end=False,
    )

    assert cursor.read_new_events() == ()
    assert cursor.offset == path.stat().st_size


def test_invalid_json_does_not_advance_cursor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"

    path.write_text(
        "{invalid-json}\n",
        encoding="utf-8",
    )

    cursor = JSONLEventCursor(
        path,
        start_at_end=False,
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON event",
    ):
        cursor.read_new_events()

    assert cursor.offset == 0


def test_rejects_invalid_event_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"

    path.write_text(
        json.dumps(
            {
                "type": 123,
                "data": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cursor = JSONLEventCursor(
        path,
        start_at_end=False,
    )

    with pytest.raises(
        ValueError,
        match="Invalid event type",
    ):
        cursor.read_new_events()

    assert cursor.offset == 0


def test_detects_file_truncation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"

    path.write_text(
        event_line("match_started") + "\n",
        encoding="utf-8",
    )

    cursor = JSONLEventCursor(
        path,
        start_at_end=False,
    )

    assert len(cursor.read_new_events()) == 1

    path.write_text("", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="truncated",
    ):
        cursor.read_new_events()


def test_detects_file_replacement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    replacement = tmp_path / "replacement.jsonl"

    path.write_text("", encoding="utf-8")
    cursor = JSONLEventCursor(path)

    replacement.write_text(
        event_line("replacement") + "\n",
        encoding="utf-8",
    )

    os.replace(replacement, path)

    with pytest.raises(
        RuntimeError,
        match="replaced",
    ):
        cursor.read_new_events()


def test_seek_to_end_marks_a_new_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"

    path.write_text(
        event_line("existing") + "\n",
        encoding="utf-8",
    )

    cursor = JSONLEventCursor(
        path,
        start_at_end=False,
    )

    cursor.seek_to_end()

    append_text(
        path,
        event_line("new") + "\n",
    )

    events = cursor.read_new_events()

    assert [
        event.type
        for event in events
    ] == ["new"]


def test_missing_file_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        JSONLEventCursor(
            tmp_path / "missing.jsonl"
        )
