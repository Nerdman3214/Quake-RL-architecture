"""Reward ledger contracts; each component remains independently inspectable."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RewardLedger:
    frag: float = 0.0
    death: float = 0.0
    objective: float = 0.0
    win: float = 0.0
    shaping: float = 0.0

    @property
    def total(self) -> float:
        return self.frag + self.death + self.objective + self.win + self.shaping
