"""Select historical events needed to prime reward state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from RL.events import Event


@dataclass(frozen=True)
class RewardPrimingPlan:
    """One non-rewarding historical-state priming decision."""

    events: tuple[Event, ...]
    session_open: bool
    match_active: bool
    boundary_index: int | None
    reason: str

    @property
    def selected_count(self) -> int:
        """Return the number of selected historical events."""

        return len(self.events)


def build_reward_priming_plan(
    events: Iterable[Event],
) -> RewardPrimingPlan:
    """Select the current active match segment, if one exists.

    The selected events may be passed through RewardMapper solely to
    initialize state. Any RewardLedger values produced during priming
    must be discarded.

    A match restart remains part of the original match-start segment
    because RewardMapper retains the existing game mode and controlled
    team across ``match_restarted``.

    A closed session or completed match produces an empty plan.
    """

    records = tuple(events)

    for event in records:
        if not isinstance(event, Event):
            raise TypeError(
                "events must contain only Event instances"
            )

    # Some unit or embedded streams do not include explicit session
    # records, so they are treated as open until session_ended appears.
    session_open = True

    match_start_index: int | None = None
    match_active = False

    for index, event in enumerate(records):
        event_type = event.type

        if event_type == "session_started":
            session_open = True
            match_start_index = None
            match_active = False
            continue

        if event_type == "session_ended":
            session_open = False
            match_start_index = None
            match_active = False
            continue

        if event_type == "match_started":
            match_start_index = index
            match_active = True
            continue

        if event_type == "match_restarted":
            # A restart cannot reconstruct mode/team without the
            # preceding match_started segment.
            match_active = match_start_index is not None
            continue

        if event_type == "match_ended":
            match_active = False

    if not session_open:
        return RewardPrimingPlan(
            events=(),
            session_open=False,
            match_active=False,
            boundary_index=None,
            reason="session_closed",
        )

    if match_start_index is None:
        return RewardPrimingPlan(
            events=(),
            session_open=True,
            match_active=False,
            boundary_index=None,
            reason="no_match_start",
        )

    if not match_active:
        return RewardPrimingPlan(
            events=(),
            session_open=True,
            match_active=False,
            boundary_index=None,
            reason="match_inactive",
        )

    return RewardPrimingPlan(
        events=records[match_start_index:],
        session_open=True,
        match_active=True,
        boundary_index=match_start_index,
        reason="active_match",
    )
