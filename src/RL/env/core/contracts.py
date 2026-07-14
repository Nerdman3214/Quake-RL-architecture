"""Gym-like environment contract without requiring Gymnasium yet."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from RL.actions.contracts import ActionCommand
from RL.observations.contracts import Observation

@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

class Environment(ABC):
    @abstractmethod
    def reset(self, *, seed: int | None = None) -> tuple[Observation, dict[str, Any]]: ...

    @abstractmethod
    def step(self, action: ActionCommand) -> StepResult: ...

    @abstractmethod
    def close(self) -> None: ...
