"""Deterministic baseline agent for bounded integration tests."""

from __future__ import annotations

from RL.actions.contracts import ActionCommand
from RL.agents.baselines.contracts import Agent
from RL.observations.contracts import Observation


class FixedActionAgent(Agent):
    """Return one configured action for every valid observation.

    This baseline contains no learning, randomness, or autonomous loop.
    It exists to validate the concrete Agent-to-Environment call path.
    """

    def __init__(
        self,
        command: ActionCommand,
    ) -> None:
        if not isinstance(command, ActionCommand):
            raise TypeError(
                "command must be an ActionCommand"
            )

        self._command = command

    @property
    def command(self) -> ActionCommand:
        """Return the immutable configured action command."""

        return self._command

    def act(
        self,
        observation: Observation,
    ) -> ActionCommand:
        """Validate the observation and return the fixed command."""

        if not isinstance(observation, Observation):
            raise TypeError(
                "observation must be an Observation"
            )

        return self._command
