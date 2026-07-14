"""Versioned action contracts. No engine-specific key injection lives here."""
from dataclasses import dataclass
from enum import IntEnum

class DiscreteAction(IntEnum):
    NO_OP = 0
    FORWARD = 1
    BACKWARD = 2
    STRAFE_LEFT = 3
    STRAFE_RIGHT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6
    FIRE = 7
    JUMP = 8
    NEXT_WEAPON = 9
    PREVIOUS_WEAPON = 10

@dataclass(frozen=True)
class ActionCommand:
    action: DiscreteAction
    duration_ticks: int = 1
