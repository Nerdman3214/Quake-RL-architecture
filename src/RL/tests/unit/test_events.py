from pathlib import Path

from src.RL.events import Event, JSONLReader, JSONLWriter


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"

    writer = JSONLWriter(str(path))
    writer.write_event(
        Event(
            type="player_kill",
            data={"killer": "bot1", "victim": "bot2"},
        )
    )
    writer.close()

    reader = JSONLReader(str(path))
    events = list(reader.read_events())

    assert len(events) == 1
    assert events[0].type == "player_kill"
    assert events[0].data["killer"] == "bot1"
