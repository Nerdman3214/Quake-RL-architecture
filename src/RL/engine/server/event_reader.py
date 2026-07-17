"""Parse live Xonotic dedicated-server output into structured events."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Optional

from RL.events import Event


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class XonoticEventReader:
    """Convert dedicated-server console lines into structured events."""

    def __init__(self, duplicate_window_seconds: float = 0.25) -> None:
        self.duplicate_window_seconds = duplicate_window_seconds
        self.sequence = 0
        self._last_clean_line: Optional[str] = None
        self._last_line_time = 0.0

    @staticmethod
    def strip_ansi(value: str) -> str:
        """Remove Xonotic terminal color/control sequences."""

        return ANSI_ESCAPE_RE.sub("", value).strip()

    def _is_duplicate(self, clean_line: str) -> bool:
        """Suppress immediately repeated ANSI/plain copies of one line."""

        now = time.monotonic()

        duplicate = (
            clean_line == self._last_clean_line
            and now - self._last_line_time <= self.duplicate_window_seconds
        )

        self._last_clean_line = clean_line
        self._last_line_time = now
        return duplicate

    def _event(self, event_type: str, raw_line: str, **data: object) -> Event:
        self.sequence += 1

        return Event(
            type=event_type,
            data={
                "sequence": self.sequence,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "source": "xonotic_dedicated_stdout",
                "raw_line": raw_line,
                **data,
            },
        )

    def parse_line(self, raw_line: str) -> Optional[Event]:
        """Parse one console line.

        Every nonblank, nonduplicate line produces an Event. Unrecognized
        output is retained as ``console_line`` so the raw session is complete.
        """

        clean_line = self.strip_ansi(raw_line)

        if not clean_line:
            return None

        if self._is_duplicate(clean_line):
            return None

        match = re.fullmatch(r"(.+?) is connecting\.\.\.", clean_line)
        if match:
            return self._event(
                "player_connecting",
                clean_line,
                player_name=match.group(1),
            )

        match = re.fullmatch(r"(.+?) connected", clean_line)
        if match:
            return self._event(
                "player_connected",
                clean_line,
                player_name=match.group(1),
            )

        match = re.fullmatch(
            r"(.+?) is now playing(?: on the ([A-Z]+) team)?",
            clean_line,
        )
        if match:
            data = {"player_name": match.group(1)}
            if match.group(2):
                data["team"] = match.group(2)

            return self._event("player_now_playing", clean_line, **data)

        match = re.fullmatch(r'Client "(.+?)" dropped', clean_line)
        if match:
            return self._event(
                "player_disconnected",
                clean_line,
                player_name=match.group(1),
            )

        match = re.fullmatch(r"(.+?) changed name to (.+)", clean_line)
        if match:
            return self._event(
                "player_name_changed",
                clean_line,
                old_name=match.group(1),
                new_name=match.group(2),
            )

        match = re.fullmatch(r"(.+?) drew first blood!\s*", clean_line)
        if match:
            return self._event(
                "first_blood",
                clean_line,
                player_name=match.group(1),
            )

        match = re.fullmatch(r"(.+?) picked up (.+)", clean_line)
        if match:
            return self._event(
                "item_pickup",
                clean_line,
                player_name=match.group(1),
                item=match.group(2),
            )

        match = re.fullmatch(
            r"([A-Za-z]+) Team has captured the (.+?) control point"
            r"(?: \(.+\))?",
            clean_line,
        )
        if match:
            return self._event(
                "control_point_captured",
                clean_line,
                team=match.group(1).upper(),
                objective=match.group(2),
            )

        match = re.fullmatch(r"([A-Z]+) team wins the match", clean_line)
        if match:
            return self._event(
                "team_match_win",
                clean_line,
                team=match.group(1),
            )

        match = re.fullmatch(r"(.+?) wins", clean_line)
        if match:
            return self._event(
                "player_match_win",
                clean_line,
                player_name=match.group(1),
            )

        kill_patterns = (
            (
                r"(.+?) was gunned down by (.+?)'s (.+?)(?:, ending .+)?",
                None,
            ),
            (
                r"(.+?) was riddled full of holes by (.+?)'s "
                r"Machine ?Gun(?:, ending .+)?",
                "Machine Gun",
            ),
            (
                r"(.+?) has been vaporized by (.+?)'s "
                r"Vortex(?:, ending .+)?",
                "Vortex",
            ),
            (
                r"(.+?) was pummeled by (?:a burst of )?(.+?)'s "
                r"Hagar rockets(?:, ending .+)?",
                "Hagar",
            ),
            (
                r"(.+?) slapped (.+?) around a bit with a large "
                r"Shotgun(?:, ending .+)?",
                "Shotgun",
            ),
            (
                r"(.+?) ate (.+?)'s (.+?)(?:, ending .+)?",
                None,
            ),
            (
                r"(.+?) got too close to (.+?)'s (.+?)(?:, ending .+)?",
                None,
            ),
            (
                r"(.+?) felt the strong pull of (.+?)'s "
                r"Crylink(?:, ending .+)?",
                "Crylink",
            ),
            (
                r"(.+?) was cooked by (.+?)(?:, ending .+)?",
                "Cooked",
            ),
            (
                r"(.+?) was grounded by (.+?)(?:, ending .+)?",
                "Grounded",
            ),
            (
                r"(.+?) was telefragged by (.+?)(?:, ending .+)?",
                "Telefrag",
            ),
        )

        for pattern, fixed_weapon in kill_patterns:
            match = re.fullmatch(pattern, clean_line)
            if not match:
                continue

            groups = match.groups()

            if pattern.startswith(r"(.+?) slapped"):
                killer = groups[0]
                victim = groups[1]
                weapon = fixed_weapon
            else:
                victim = groups[0]
                killer = groups[1]
                weapon = fixed_weapon or groups[2]

            return self._event(
                "player_kill",
                clean_line,
                killer=killer,
                victim=victim,
                weapon=weapon,
            )

        suicide_patterns = (
            (
                r"(.+?) blew themself up with their own "
                r"(.+?)(?:, losing .+)?",
                None,
            ),
            (
                r"(.+?) blew themself up with their "
                r"(.+?)(?:, losing .+)?",
                None,
            ),
            (
                r"(.+?) didn't see their own "
                r"(.+?)(?:, losing .+)?",
                None,
            ),
            (
                r"(.+?) felt the strong pull of their Crylink",
                "Crylink",
            ),
            (
                r"(.+?) played with tiny Hagar rockets"
                r"(?:, losing .+)?",
                "Hagar",
            ),
            (
                r"(.+?) was in the wrong place"
                r"(?:, losing .+)?",
                "environment",
            ),
        )

        for pattern, fixed_weapon in suicide_patterns:
            match = re.fullmatch(pattern, clean_line)
            if not match:
                continue

            weapon = fixed_weapon
            if weapon is None and len(match.groups()) >= 2:
                weapon = match.group(2)

            return self._event(
                "player_suicide",
                clean_line,
                player_name=match.group(1),
                weapon=weapon,
            )

        match = re.fullmatch(r"(.+?) made a TRIPLE FRAG!\s*", clean_line)
        if match:
            return self._event(
                "frag_streak",
                clean_line,
                player_name=match.group(1),
                streak="triple_frag",
            )

        match = re.fullmatch(r"(.+?) unlocked RAGE!\s*", clean_line)
        if match:
            return self._event(
                "frag_streak",
                clean_line,
                player_name=match.group(1),
                streak="rage",
            )

        match = re.fullmatch(
            r"(.+?) started a MASSACRE!\s*",
            clean_line,
        )
        if match:
            return self._event(
                "frag_streak",
                clean_line,
                player_name=match.group(1),
                streak="massacre",
            )

        match = re.fullmatch(
            r"(.+?) executed MAYHEM!\s*",
            clean_line,
        )
        if match:
            return self._event(
                "frag_streak",
                clean_line,
                player_name=match.group(1),
                streak="mayhem",
            )

        if clean_line == "The current level has been LOST.":
            return self._event("level_lost", clean_line)

        match = re.fullmatch(r"Switching to map (.+)", clean_line)
        if match:
            return self._event(
                "map_changed",
                clean_line,
                map_name=match.group(1),
            )

        return self._event("console_line", clean_line)
