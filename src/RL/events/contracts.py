"""Core contracts for structured RL events."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class Event:
    """A serializable telemetry event."""

    type: str
    data: Dict[str, Any] = field(default_factory=dict)
