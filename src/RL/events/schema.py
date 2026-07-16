"""Event schema definitions."""

from typing import Any, Dict

from .contracts import Event


def to_payload(event: Event) -> Dict[str, Any]:
    """Convert an event into a JSON-serializable payload."""
    return {
        "type": event.type,
        "data": event.data,
    }