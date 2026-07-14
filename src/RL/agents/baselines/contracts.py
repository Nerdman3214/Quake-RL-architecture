"""Baseline-agent interface used before neural-network training."""
from abc import ABC, abstractmethod
from RL.actions.contracts import ActionCommand
from RL.observations.contracts import Observation

class Agent(ABC):
    @abstractmethod
    def act(self, observation: Observation) -> ActionCommand: ...
