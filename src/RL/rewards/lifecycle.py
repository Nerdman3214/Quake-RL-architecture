"""Ordered reward and episode-lifecycle event processing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from RL.events import Event
from RL.rewards.contracts import RewardLedger
from RL.rewards.event_mapper import RewardMapper


@dataclass(frozen=True)
class EventStepOutcome:
    """Auditable result from one ordered event batch."""

    events: tuple[Event, ...]
    reward_ledger: RewardLedger
    terminated: bool
    truncated: bool
    termination_reason: str | None
    match_active: bool

    @property
    def reward(self) -> float:
        """Return the scalar reward for this event batch."""

        return self.reward_ledger.total

    @property
    def reward_components(
        self,
    ) -> dict[str, float]:
        """Return the independently inspectable components."""

        return self.reward_ledger.components


class EventLifecycleGate:
    """Map ordered events to rewards and one terminal outcome.

    All events in the supplied batch are passed through RewardMapper,
    including events occurring after the first terminal signal in that
    same batch. This preserves Xonotic's observed ordering where
    ``match_ended`` precedes winner announcements.

    Once a batch produces termination or truncation, another batch
    cannot be processed until ``reset_episode`` is called.
    """

    def __init__(
        self,
        reward_mapper: RewardMapper,
        *,
        match_active: bool = False,
    ) -> None:
        if not isinstance(
            reward_mapper,
            RewardMapper,
        ):
            raise TypeError(
                "reward_mapper must be a RewardMapper"
            )

        self._reward_mapper = reward_mapper
        self._match_active = bool(match_active)
        self._episode_done = False

    @property
    def match_active(self) -> bool:
        """Return whether the current match is active."""

        return self._match_active

    @property
    def episode_done(self) -> bool:
        """Return whether a terminal outcome was emitted."""

        return self._episode_done

    def reset_episode(
        self,
        *,
        match_active: bool = False,
    ) -> None:
        """Allow processing for a new environment episode.

        RewardMapper state is intentionally not altered here. The
        caller must prime it or wait for a new ``match_started`` event.
        """

        self._match_active = bool(match_active)
        self._episode_done = False

    @staticmethod
    def _validate_events(
        events: Iterable[Event],
    ) -> tuple[Event, ...]:
        records = tuple(events)

        for event in records:
            if not isinstance(event, Event):
                raise TypeError(
                    "events must contain only Event instances"
                )

        return records

    def process(
        self,
        events: Iterable[Event],
    ) -> EventStepOutcome:
        """Process one ordered batch and return its lifecycle result."""

        records = self._validate_events(events)

        if self._episode_done:
            raise RuntimeError(
                "lifecycle gate must be reset before "
                "processing more events"
            )

        ledger = RewardLedger()

        terminated = False
        truncated = False
        reason: str | None = None

        for event in records:
            # RewardMapper receives every event in order, including
            # post-match winner announcements in this same batch.
            ledger = (
                ledger
                + self._reward_mapper.map_event(event)
            )

            event_type = event.type

            if event_type == "session_started":
                self._match_active = False
                continue

            if event_type in {
                "match_started",
                "match_restarted",
            }:
                self._match_active = True
                continue

            if event_type == "match_ended":
                if reason is None:
                    terminated = True
                    reason = "match_ended"

                self._match_active = False
                continue

            if event_type == "player_disconnected":
                is_controlled_player = (
                    event.data.get("player_name")
                    == self._reward_mapper.controlled_player
                )

                if (
                    reason is None
                    and self._match_active
                    and is_controlled_player
                ):
                    truncated = True
                    reason = (
                        "controlled_player_disconnected"
                    )

                if is_controlled_player:
                    self._match_active = False

                continue

            if event_type == "session_ended":
                if (
                    reason is None
                    and self._match_active
                ):
                    truncated = True
                    reason = "session_ended"

                self._match_active = False

        if reason is not None:
            self._episode_done = True
            self._match_active = False

        return EventStepOutcome(
            events=records,
            reward_ledger=ledger,
            terminated=terminated,
            truncated=truncated,
            termination_reason=reason,
            match_active=self._match_active,
        )
