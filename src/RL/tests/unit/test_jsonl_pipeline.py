from pathlib import Path

from src.RL.events import Event, JSONLReader, JSONLWriter


def test_jsonl_writer_reader_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"

    writer = JSONLWriter(str(path))
    writer.write_event(
        Event(
            type="spawn",
            data={"player": 1, "health": 100},
        )
    )
    writer.write_event(
        Event(
            type="move",
            data={"x": 12, "y": 4},
        )
    )
    writer.close()

    reader = JSONLReader(str(path))
    events = list(reader.read_events())

    assert [event.type for event in events] == ["spawn", "move"]
    assert events[0].data["player"] == 1
    assert events[1].data["y"] == 4
