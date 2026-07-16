"""Bridge that normalizes raw engine messages into event dictionaries."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from RL.events import Event, JSONLWriter

def process_raw_event(raw_event):
    # This is where you'd parse raw engine output
    # For now, simulate an example event:
    if isinstance(raw_event, str):
        if "spawn" in raw_event:
            return {
                "tick": 1520,
                "event": "spawn",
                "data": {"entity": 1}
            }
        elif "weapon_switch" in raw_event:
            parts = raw_event.split()
            weapon = parts[-1]
            return {
                "tick": int(parts[0]),
                "event": "weapon_switch",
                "data": {"weapon": weapon}
            }
    # Add more rules as needed
    return None

def run_event_stream():
    writer = JSONLWriter("events.jsonl")
    try:
        # Simulate reading from Xonotic (replace with real input)
        sample_events = [
            "1520 spawn entity=1",
            "1521 weapon_switch shotgun",
            "1524 move x=14.2 y=38.7 z=0.0"
        ]
        for raw in sample_events:
            event_dict = process_raw_event(raw)
            if event_dict:
                writer.write(event_dict)
    finally:
        writer.close()


class EventStream:
    """Normalize incoming engine messages into structured events."""

    def __init__(
        self,
        writer: Optional[JSONLWriter] = None,
        output_path: Optional[str] = None,
    ):
        if writer is not None:
            self.writer = writer
        else:
            self.writer = JSONLWriter(output_path or "events.jsonl")

    def normalize(self, raw_message: Dict[str, Any]) -> Event:
        """Convert a raw message into a normalized event object."""

        if not isinstance(raw_message, dict):
            raise TypeError("raw_message must be a dictionary")

        normalized = {
            "event": (
                raw_message.get("event")
                or raw_message.get("type")
                or "unknown"
            ),
            "source": raw_message.get("source", "engine"),
            "payload": raw_message.get("payload", {}),
        }

        if "tick" in raw_message:
            normalized["tick"] = raw_message["tick"]

        if "timestamp" in raw_message:
            normalized["timestamp"] = raw_message["timestamp"]

        return Event(type=normalized["event"], data=normalized)

    def emit(self, raw_message: Dict[str, Any]) -> Event:
        """Normalize and persist a single raw message."""

        event = self.normalize(raw_message)
        self.writer.write_event(event)
        return event

    def emit_many(self, messages: Iterable[Dict[str, Any]]) -> List[Event]:
        """Normalize and persist a batch of raw messages."""

        return [self.emit(message) for message in messages]

    def close(self) -> None:
        """Close the underlying writer."""

        self.writer.close()
