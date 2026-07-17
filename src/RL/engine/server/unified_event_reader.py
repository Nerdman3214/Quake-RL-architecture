"""Select authoritative Xonotic events without double-counting."""

from __future__ import annotations

from typing import Optional

from RL.engine.server.event_reader import XonoticEventReader
from RL.engine.server.eventlog_reader import XonoticEventLogReader
from RL.engine.server.scoreboard_tracker import (
    XonoticScoreboardTracker,
)
from RL.events import Event


class XonoticUnifiedEventReader:
    """Combine structured eventlog and human-readable server events.

    Structured eventlog records are the primary source for combat,
    objectives, teams, scores, and match lifecycle.

    Human-readable messages remain useful for supplemental signals such
    as item pickups, first blood, and frag-streak announcements.
    """

    HUMAN_EVENTS_SUPERSEDED_BY_EVENTLOG = frozenset(
        {
            "player_kill",
            "player_suicide",
            "control_point_captured",
        }
    )

    def __init__(
        self,
        duplicate_window_seconds: float = 0.25,
    ) -> None:
        self.human_reader = XonoticEventReader(
            duplicate_window_seconds=duplicate_window_seconds
        )
        self.eventlog_reader = XonoticEventLogReader()
        self.scoreboard_tracker = XonoticScoreboardTracker()

        self.structured_match_active = False
        self.suppressed_human_count = 0

    @staticmethod
    def _with_metadata(
        event: Event,
        *,
        channel: str,
        authority_tier: str,
    ) -> Event:
        data = dict(event.data)
        data["event_channel"] = channel
        data["authority_tier"] = authority_tier

        return Event(type=event.type, data=data)

    def _suppressed_human_event(self, event: Event) -> Event:
        """Preserve an overlapping human line without rewarding it."""

        self.suppressed_human_count += 1

        data = dict(event.data)
        data["suppressed_event_type"] = event.type
        data["suppression_reason"] = (
            "structured_eventlog_is_primary"
        )
        data["event_channel"] = "human_console"
        data["authority_tier"] = "suppressed_duplicate"

        return Event(
            type="suppressed_human_event",
            data=data,
        )

    def strip_ansi(self, raw_line: str) -> str:
        """Remove terminal color codes from one server line."""

        return self.human_reader.strip_ansi(raw_line)

    def parse_line(self, raw_line: str) -> Optional[Event]:
        """Parse one server line through the appropriate source."""

        clean_line = self.strip_ansi(raw_line)
        stripped = clean_line.strip()

        if stripped.startswith(":"):
            event = self.eventlog_reader.parse_line(stripped)

            if event is None:
                return None

            if event.type == "match_started":
                self.structured_match_active = True
                self.scoreboard_tracker = (
                    XonoticScoreboardTracker()
                )

            event = self.scoreboard_tracker.process(event)

            return self._with_metadata(
                event,
                channel="structured_eventlog",
                authority_tier="primary",
            )

        event = self.human_reader.parse_line(clean_line)

        if event is None:
            return None

        if (
            self.structured_match_active
            and event.type
            in self.HUMAN_EVENTS_SUPERSEDED_BY_EVENTLOG
        ):
            return self._suppressed_human_event(event)

        return self._with_metadata(
            event,
            channel="human_console",
            authority_tier="supplemental",
        )
