"""Observation contracts shared by engine bridges and agents."""
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class PlayerTelemetry:
    health: int
    armor: int
    ammo: int
    weapon: str
    alive: bool
    score: int
    match_time_seconds: float

@dataclass(frozen=True)
class Observation:
    frame: Any | None
    telemetry: PlayerTelemetry | None
    tick: int
