"""Auditable reward contracts for the controlled RL player."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict


@dataclass(frozen=True)
class RewardWeights:
    """Configurable values used by the event-to-reward mapper."""

    # Combat.
    frag: float = 1.0
    death: float = -1.0
    suicide: float = -1.5
    team_kill: float = -2.0

    # Generic objectives and match results.
    objective: float = 2.0
    win: float = 5.0

    # Capture the Flag.
    ctf_steal: float = 0.25
    ctf_pickup: float = 0.10
    ctf_return: float = 0.50
    ctf_capture: float = 3.0
    ctf_drop: float = -0.25

    # Domination.
    domination_capture: float = 1.0

    # Key Hunt.
    keyhunt_capture: float = 2.0
    keyhunt_carrier_frag: float = 1.0
    keyhunt_collect: float = 0.25
    keyhunt_destroyed: float = 0.50
    keyhunt_destroyed_holding_key: float = 0.0
    keyhunt_drop: float = -0.25
    keyhunt_lose: float = -0.50
    keyhunt_push: float = 0.0
    keyhunt_pushed: float = 0.0

    # Small shaping signals.
    item_pickup: float = 0.05
    first_blood: float = 0.25

    # Frags already receive reward, so streaks default to zero.
    streak: float = 0.0

    disconnect: float = -2.0
    shaping: float = 0.0


@dataclass(frozen=True)
class RewardLedger:
    """Independently inspectable reward components."""

    frag: float = 0.0
    death: float = 0.0
    suicide: float = 0.0
    team_kill: float = 0.0

    objective: float = 0.0
    ctf: float = 0.0
    domination: float = 0.0
    keyhunt: float = 0.0

    win: float = 0.0
    item_pickup: float = 0.0
    first_blood: float = 0.0
    streak: float = 0.0
    disconnect: float = 0.0
    shaping: float = 0.0

    @property
    def components(self) -> Dict[str, float]:
        """Return each reward component separately."""

        return {
            field.name: float(getattr(self, field.name))
            for field in fields(self)
        }

    @property
    def total(self) -> float:
        """Return the sum of every reward component."""

        return sum(self.components.values())

    @property
    def is_zero(self) -> bool:
        """Return whether the ledger contains no reward."""

        return all(
            value == 0.0
            for value in self.components.values()
        )

    def __add__(self, other: object) -> "RewardLedger":
        """Combine ledgers without losing component accounting."""

        if not isinstance(other, RewardLedger):
            return NotImplemented

        return RewardLedger(
            **{
                field.name: (
                    getattr(self, field.name)
                    + getattr(other, field.name)
                )
                for field in fields(self)
            }
        )
