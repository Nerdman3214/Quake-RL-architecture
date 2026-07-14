"""Transport-neutral boundary between the game processes and Python."""
from __future__ import annotations
from abc import ABC, abstractmethod
from RL.actions.contracts import ActionCommand
from RL.observations.contracts import Observation

class EngineBridge(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def reset_match(self) -> Observation: ...

    @abstractmethod
    def send_action(self, action: ActionCommand) -> None: ...

    @abstractmethod
    def read_observation(self) -> Observation: ...

    @abstractmethod
    def close(self) -> None: ...
