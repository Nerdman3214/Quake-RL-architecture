"""Environment adapter around an engine bridge.

Authoritative rewards and lifecycle signals are optional. Constructing
the environment without an event processor preserves the original
reward-disabled behavior.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from RL.actions.contracts import ActionCommand
from RL.engine.bridge.contracts import EngineBridge
from RL.env.core.contracts import Environment, StepResult
from RL.observations.contracts import Observation


class EventStepOutcomeLike(Protocol):
    """Outcome fields consumed by BridgeEnvironment."""

    events: tuple[Any, ...]
    reward: float
    reward_components: dict[str, float]
    terminated: bool
    truncated: bool
    termination_reason: str | None
    match_active: bool


class EventStepProcessor(Protocol):
    """Optional authoritative event processor operations."""

    def reset_episode(self) -> None:
        """Establish a new event boundary and prime state."""

    def process_step(self) -> EventStepOutcomeLike:
        """Return the events and lifecycle result for one step."""


class BridgeEnvironment(Environment):
    """Expose reset/step semantics around one engine bridge."""

    def __init__(
        self,
        bridge: EngineBridge,
        *,
        event_processor: EventStepProcessor | None = None,
    ) -> None:
        if bridge is None:
            raise TypeError(
                "bridge must not be None"
            )

        if event_processor is not None:
            missing = [
                method_name
                for method_name in (
                    "reset_episode",
                    "process_step",
                )
                if not callable(
                    getattr(
                        event_processor,
                        method_name,
                        None,
                    )
                )
            ]

            if missing:
                raise TypeError(
                    "event_processor must provide "
                    "reset_episode and process_step"
                )

        self._bridge = bridge
        self._event_processor = event_processor

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

    @property
    def event_enabled(self) -> bool:
        """Return whether authoritative event processing is enabled."""

        return self._event_processor is not None

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

    @staticmethod
    def _validate_event_outcome(
        outcome: object,
    ) -> EventStepOutcomeLike:
        required_attributes = (
            "events",
            "reward",
            "reward_components",
            "terminated",
            "truncated",
            "termination_reason",
            "match_active",
        )

        if any(
            not hasattr(outcome, attribute)
            for attribute in required_attributes
        ):
            raise TypeError(
                "event processor must return an event "
                "step outcome"
            )

        try:
            tuple(getattr(outcome, "events"))
            float(getattr(outcome, "reward"))
            dict(getattr(outcome, "reward_components"))
        except (TypeError, ValueError) as error:
            raise TypeError(
                "event processor returned an invalid "
                "event step outcome"
            ) from error

        if not isinstance(
            getattr(outcome, "terminated"),
            bool,
        ):
            raise TypeError(
                "event outcome terminated must be bool"
            )

        if not isinstance(
            getattr(outcome, "truncated"),
            bool,
        ):
            raise TypeError(
                "event outcome truncated must be bool"
            )

        return cast(
            EventStepOutcomeLike,
            outcome,
        )

    def _base_info(self) -> dict[str, Any]:
        return {
            "seed_requested": self._requested_seed,
            "seed_applied": False,
            "episode_step": self._episode_step,
        }

    def _info(
        self,
        outcome: EventStepOutcomeLike | None = None,
    ) -> dict[str, Any]:
        info = self._base_info()

        if self._event_processor is None:
            return {
                "reward_mode": "disabled",
                "termination_mode": "disabled",
                "reset_mode": (
                    "bridge_local_observation_state_only"
                ),
                **info,
            }

        processor = self._event_processor

        enabled_info: dict[str, Any] = {
            "reward_mode": (
                "authoritative_server_events"
            ),
            "termination_mode": (
                "authoritative_server_events"
            ),
            "reset_mode": (
                "bridge_local_observation_state_"
                "with_event_boundary"
            ),
            **info,
            "event_processor_initialized": bool(
                getattr(
                    processor,
                    "initialized",
                    True,
                )
            ),
            "event_cursor_offset": getattr(
                processor,
                "cursor_offset",
                None,
            ),
            "event_history_count": getattr(
                processor,
                "history_event_count",
                None,
            ),
            "event_priming_reason": getattr(
                processor,
                "priming_reason",
                None,
            ),
            "event_primed_count": getattr(
                processor,
                "primed_event_count",
                None,
            ),
            "event_match_active": (
                outcome.match_active
                if outcome is not None
                else getattr(
                    processor,
                    "match_active",
                    False,
                )
            ),
            "event_episode_done": getattr(
                processor,
                "episode_done",
                False,
            ),
            "event_game_mode": getattr(
                processor,
                "current_mode",
                None,
            ),
            "event_controlled_team": getattr(
                processor,
                "controlled_team",
                None,
            ),
        }

        if outcome is not None:
            events = tuple(outcome.events)

            enabled_info.update(
                {
                    "event_count": len(events),
                    "event_types": tuple(
                        getattr(
                            event,
                            "type",
                            type(event).__name__,
                        )
                        for event in events
                    ),
                    "reward_components": dict(
                        outcome.reward_components
                    ),
                    "termination_reason": (
                        outcome.termination_reason
                    ),
                }
            )

        return enabled_info

    def reset(
        self,
        *,
        seed: int | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset local bridge and event-processing state.

        This does not launch, restart, or reconfigure the server match.
        """

        self._require_open()
        self._ready = False

        if not self._connected:
            self._bridge.connect()
            self._connected = True

        observation = self._validate_observation(
            self._bridge.reset_match()
        )

        if self._event_processor is not None:
            self._event_processor.reset_episode()

        self._requested_seed = seed
        self._episode_step = 0
        self._ready = True

        return observation, self._info()

    def step(
        self,
        action: ActionCommand,
    ) -> StepResult:
        """Execute one action and process its resulting event batch."""

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

        event_outcome: EventStepOutcomeLike | None = None

        if self._event_processor is not None:
            try:
                event_outcome = (
                    self._validate_event_outcome(
                        self._event_processor.process_step()
                    )
                )
            except Exception:
                # The action and observation have already occurred.
                # Require a reset rather than continuing with an
                # uncertain event boundary.
                self._ready = False
                raise

        self._episode_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        if event_outcome is not None:
            reward = float(event_outcome.reward)
            terminated = event_outcome.terminated
            truncated = event_outcome.truncated

        info = self._info(event_outcome)

        if terminated or truncated:
            self._ready = False

        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
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
