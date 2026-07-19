"""Step-scoped processing for an authoritative event JSONL file."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from RL.events import (
    Event,
    JSONLEventCursor,
    TerminalEventBatchCollector,
)
from RL.rewards.contracts import RewardLedger
from RL.rewards.event_mapper import RewardMapper
from RL.rewards.lifecycle import (
    EventLifecycleGate,
    EventStepOutcome,
)
from RL.rewards.priming import (
    RewardPrimingPlan,
    build_reward_priming_plan,
)


class JSONLEventStepProcessor:
    """Combine history priming and incremental event processing.

    Reset reads all complete existing records through one cursor,
    initializes RewardMapper state without returning historical
    rewards, and leaves the cursor at the current append boundary.

    Each later call consumes only newly appended events. Winner
    announcements immediately following ``match_ended`` are collected
    through the bounded terminal-tail collector.
    """

    def __init__(
        self,
        event_path: str | Path,
        controlled_player: str,
        *,
        quiet_period_seconds: float = 0.05,
        max_wait_seconds: float = 0.25,
        poll_interval_seconds: float = 0.005,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.event_path = Path(event_path)

        if not controlled_player.strip():
            raise ValueError(
                "controlled_player must not be blank"
            )

        self.controlled_player = controlled_player

        self.quiet_period_seconds = float(
            quiet_period_seconds
        )
        self.max_wait_seconds = float(
            max_wait_seconds
        )
        self.poll_interval_seconds = float(
            poll_interval_seconds
        )

        self._monotonic = monotonic
        self._sleep = sleep

        self._cursor: JSONLEventCursor | None = None

        self._collector: (
            TerminalEventBatchCollector | None
        ) = None

        self._mapper: RewardMapper | None = None
        self._gate: EventLifecycleGate | None = None

        self._priming_plan: RewardPrimingPlan | None = (
            None
        )

        self._history_event_count = 0

    @property
    def initialized(self) -> bool:
        """Return whether an episode boundary was established."""

        return (
            self._cursor is not None
            and self._collector is not None
            and self._mapper is not None
            and self._gate is not None
            and self._priming_plan is not None
        )

    @property
    def cursor_offset(self) -> int | None:
        """Return the next unread event-file byte position."""

        if self._cursor is None:
            return None

        return self._cursor.offset

    @property
    def history_event_count(self) -> int:
        """Return the records observed while establishing reset."""

        return self._history_event_count

    @property
    def priming_reason(self) -> str:
        """Return the most recent historical priming decision."""

        if self._priming_plan is None:
            return "not_initialized"

        return self._priming_plan.reason

    @property
    def primed_event_count(self) -> int:
        """Return the number of events used only for mapper state."""

        if self._priming_plan is None:
            return 0

        return self._priming_plan.selected_count

    @property
    def match_active(self) -> bool:
        """Return the lifecycle gate's active-match state."""

        if self._gate is None:
            return False

        return self._gate.match_active

    @property
    def episode_done(self) -> bool:
        """Return whether a terminal event outcome was emitted."""

        if self._gate is None:
            return False

        return self._gate.episode_done

    @property
    def current_mode(self) -> str | None:
        """Return the reward mapper's normalized game mode."""

        if (
            self._mapper is None
            or self._mapper.current_mode is None
        ):
            return None

        return self._mapper.current_mode.value

    @property
    def controlled_team(self) -> str | None:
        """Return the currently reconstructed controlled team."""

        if self._mapper is None:
            return None

        return self._mapper.controlled_team

    def reset_episode(self) -> None:
        """Prime state and establish a new append boundary."""

        cursor = JSONLEventCursor(
            self.event_path,
            start_at_end=False,
        )

        history = cursor.read_new_events()
        plan = build_reward_priming_plan(history)

        mapper = RewardMapper(
            self.controlled_player
        )

        # Historical ledgers are deliberately discarded. These events
        # exist only to reconstruct game mode, team, active-match state,
        # and win-deduplication state.
        for historical_event in plan.events:
            mapper.map_event(historical_event)

        gate = EventLifecycleGate(
            mapper,
            match_active=plan.match_active,
        )

        collector = TerminalEventBatchCollector(
            cursor,
            quiet_period_seconds=(
                self.quiet_period_seconds
            ),
            max_wait_seconds=self.max_wait_seconds,
            poll_interval_seconds=(
                self.poll_interval_seconds
            ),
            monotonic=self._monotonic,
            sleep=self._sleep,
        )

        self._cursor = cursor
        self._collector = collector
        self._mapper = mapper
        self._gate = gate
        self._priming_plan = plan
        self._history_event_count = len(history)

    @staticmethod
    def _zero_outcome(
        events: tuple[Event, ...],
    ) -> EventStepOutcome:
        """Return an inactive, non-rewarding batch outcome."""

        return EventStepOutcome(
            events=events,
            reward_ledger=RewardLedger(),
            terminated=False,
            truncated=False,
            termination_reason=None,
            match_active=False,
        )

    def process_step(self) -> EventStepOutcome:
        """Process events appended since the previous step boundary."""

        if (
            self._collector is None
            or self._gate is None
        ):
            raise RuntimeError(
                "event processor must be reset before step"
            )

        if self._gate.episode_done:
            raise RuntimeError(
                "event processor must be reset after "
                "a terminal outcome"
            )

        events = self._collector.read_batch()

        if self._gate.match_active:
            return self._gate.process(events)

        # While inactive, ignore orphan winner announcements and other
        # cleanup records. Arm only when an authoritative match_started
        # event appears after the established cursor boundary.
        match_start_index = next(
            (
                index
                for index, event in enumerate(events)
                if event.type == "match_started"
            ),
            None,
        )

        if match_start_index is None:
            return self._zero_outcome(events)

        return self._gate.process(
            events[match_start_index:]
        )
