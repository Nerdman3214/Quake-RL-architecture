"""Helpers for serializing event payloads."""

from typing import Any, Dict

from .contracts import Event


EVENT_TYPE_FIELD = "type"
EVENT_DATA_FIELD = "data"


def to_payload(event: Event) -> Dict[str, Any]:
    """Convert an event to the JSON Lines payload shape."""

    return {
        EVENT_TYPE_FIELD: event.type,
        EVENT_DATA_FIELD: event.data,
    }
