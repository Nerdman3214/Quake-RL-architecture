"""Bridge that normalizes raw engine messages into events."""

from typing import Any, Dict, Iterable, List, Optional

from RL.events.contracts import Event
from RL.events.jsonl_writer import JSONLWriter


def process_raw_event(raw_event):

    if isinstance(raw_event, str):

        if "spawn" in raw_event:
            return {
                "event": "spawn",
                "data": {"entity": 1}
            }

        if "weapon_switch" in raw_event:
            parts = raw_event.split()

            return {
                "event": "weapon_switch",
                "data": {"weapon": parts[-1]}
            }

    return None


def run_event_stream():

    writer = JSONLWriter("events.jsonl")

    try:

        sample_events = [
            "1520 spawn entity=1",
            "1521 weapon_switch shotgun",
            "1524 move x=14.2 y=38.7 z=0.0"
        ]

        for raw in sample_events:

            event_dict = process_raw_event(raw)

            if event_dict:

                event = Event(
                    type=event_dict["event"],
                    data=event_dict["data"]
                )

                writer.write_event(event)

                print(f"Recorded: {event}")

    finally:
        writer.close()


class EventStream:

    def __init__(
        self,
        writer: Optional[JSONLWriter] = None,
        output_path: Optional[str] = None
    ):

        self.writer = writer or JSONLWriter(
            output_path or "events.jsonl"
        )


    def normalize(self, raw_message: Dict[str, Any]):

        return Event(
            type=(
                raw_message.get("event")
                or raw_message.get("type")
                or "unknown"
            ),
            data=raw_message
        )


    def emit(self, raw_message):

        event = self.normalize(raw_message)
        self.writer.write_event(event)

        return event


    def emit_many(self, messages: Iterable[Dict[str, Any]]) -> List[Event]:

        return [
            self.emit(message)
            for message in messages
        ]


    def close(self):

        self.writer.close()
