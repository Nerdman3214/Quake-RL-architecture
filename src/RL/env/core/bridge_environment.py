"""Environment adapter around an engine bridge.

Rewards and episode termination remain disabled until an
authoritative, step-scoped server-event source is connected.
"""

from __future__ import annotations

from typing import Any

from RL.actions.contracts import ActionCommand
from RL.engine.bridge.contracts import EngineBridge
from RL.env.core.contracts import Environment, StepResult
from RL.observations.contracts import Observation


class BridgeEnvironment(Environment):
    """Expose reset/step semantics around one engine bridge."""

    def __init__(
        self,
        bridge: EngineBridge,
    ) -> None:
        if bridge is None:
            raise TypeError(
                "bridge must not be None"
            )

        self._bridge = bridge

        self._connected = False
        self._ready = False
        self._closed = False

        self._episode_step = 0
        self._requested_seed: int | None = None

    @property
    def connected(self) -> bool:
        """Return whether bridge connection was requested."""

        return self._connected

    @property
    def ready(self) -> bool:
        """Return whether reset completed successfully."""

        return self._ready

    @property
    def closed(self) -> bool:
        """Return whether the environment has been closed."""

        return self._closed

    @property
    def episode_step(self) -> int:
        """Return the number of completed environment steps."""

        return self._episode_step

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "environment is closed"
            )

    @staticmethod
    def _validate_observation(
        observation: Observation,
    ) -> Observation:
        if not isinstance(
            observation,
            Observation,
        ):
            raise TypeError(
                "bridge must return an Observation"
            )

        return observation

    def _info(self) -> dict[str, Any]:
        return {
            "reward_mode": "disabled",
            "termination_mode": "disabled",
            "reset_mode": (
                "bridge_local_observation_state_only"
            ),
            "seed_requested": self._requested_seed,
            "seed_applied": False,
            "episode_step": self._episode_step,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset bridge-local state and read an observation.

        This does not launch, restart, or reconfigure the match.
        """

        self._require_open()

        if not self._connected:
            self._bridge.connect()
            self._connected = True

        observation = self._validate_observation(
            self._bridge.reset_match()
        )

        self._requested_seed = seed
        self._episode_step = 0
        self._ready = True

        return observation, self._info()

    def step(
        self,
        action: ActionCommand,
    ) -> StepResult:
        """Execute one action and read its following observation."""

        self._require_open()

        if not self._ready:
            raise RuntimeError(
                "environment must be reset before step"
            )

        if not isinstance(action, ActionCommand):
            raise TypeError(
                "action must be an ActionCommand"
            )

        self._bridge.send_action(action)

        observation = self._validate_observation(
            self._bridge.read_observation()
        )

        self._episode_step += 1

        return StepResult(
            observation=observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=self._info(),
        )

    def close(self) -> None:
        """Close the bridge and permanently close this adapter."""

        if self._closed:
            return

        try:
            self._bridge.close()
        finally:
            self._connected = False
            self._ready = False
            self._closed = True
