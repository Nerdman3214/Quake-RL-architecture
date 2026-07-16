"""Event contracts."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Event:
    """An event in the system."""

    type: str
    data: Dict[str, Any]