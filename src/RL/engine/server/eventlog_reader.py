"""Parse Xonotic's authoritative colon-delimited eventlog format."""

from __future__ import annotations

from typing import Optional

from RL.env.multiplayer import split_mode_and_map
from RL.events import Event


TEAM_NAMES = {
    "-1": "SPECTATOR",
    "1": "NONE",
    "5": "RED",
    "14": "BLUE",
    "13": "YELLOW",
    "10": "PINK",
}


class XonoticEventLogReader:
    """Convert structured Xonotic eventlog lines into Event objects."""

    def __init__(self) -> None:
        self.player_names: dict[str, str] = {}
        self.current_raw_game_type: Optional[str] = None
        self.current_map: Optional[str] = None
        self.current_match_id: Optional[str] = None

    def _player_name(self, player_id: str) -> Optional[str]:
        return self.player_names.get(player_id)

    @staticmethod
    def _attributes(values: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}

        for value in values:
            if "=" not in value:
                continue

            key, content = value.split("=", 1)
            result[key] = content

        return result

    @staticmethod
    def _mode_data(value: str) -> dict[str, object]:
        mode, map_name = split_mode_and_map(value)

        return {
            "raw_game_type_map": value,
            "game_mode": mode.value if mode is not None else None,
            "map_name": map_name,
        }

    def parse_line(self, raw_line: str) -> Optional[Event]:
        """Parse one structured eventlog line."""

        line = raw_line.strip()

        if not line.startswith(":"):
            return None

        if line.startswith(":logversion:"):
            parts = line.split(":", 2)

            return Event(
                type="eventlog_version",
                data={
                    "version": parts[2],
                    "raw_line": line,
                },
            )

        if line.startswith(":gamestart:"):
            parts = line.split(":", 3)

            if len(parts) != 4:
                return Event(
                    type="eventlog_line",
                    data={"raw_line": line},
                )

            game_type_map = parts[2]
            match_id = parts[3]
            mode_data = self._mode_data(game_type_map)

            self.current_raw_game_type = game_type_map
            self.current_game_mode = mode_data["game_mode"]
            self.current_map = mode_data["map_name"]
            self.current_match_id = match_id

            return Event(
                type="match_started",
                data={
                    **mode_data,
                    "match_id": match_id,
                    "raw_line": line,
                },
            )

        if line.startswith(":gameinfo:mutators:LIST"):
            parts = line.split(":")

            return Event(
                type="match_mutators",
                data={
                    "mutators": parts[4:],
                    "raw_line": line,
                },
            )

        if line == ":gameinfo:end":
            return Event(
                type="match_info_complete",
                data={"raw_line": line},
            )

        if line == ":startdelay_ended":
            return Event(
                type="match_start_delay_ended",
                data={"raw_line": line},
            )

        if line.startswith(":join:"):
            parts = line.split(":", 5)

            if len(parts) != 6:
                return Event(
                    type="eventlog_line",
                    data={"raw_line": line},
                )

            player_id = parts[2]
            nickname = parts[5]
            self.player_names[player_id] = nickname

            return Event(
                type="player_joined",
                data={
                    "player_id": player_id,
                    "slot": parts[3],
                    "connection": parts[4],
                    "is_bot": parts[4] == "bot",
                    "player_name": nickname,
                    "raw_line": line,
                },
            )

        if line.startswith(":name:"):
            parts = line.split(":", 3)

            if len(parts) != 4:
                return Event(
                    type="eventlog_line",
                    data={"raw_line": line},
                )

            player_id = parts[2]
            old_name = self._player_name(player_id)
            new_name = parts[3]
            self.player_names[player_id] = new_name

            return Event(
                type="player_name_changed",
                data={
                    "player_id": player_id,
                    "old_name": old_name,
                    "new_name": new_name,
                    "raw_line": line,
                },
            )

        if line.startswith(":part:"):
            parts = line.split(":", 2)
            player_id = parts[2]
            player_name = self._player_name(player_id)

            return Event(
                type="player_left",
                data={
                    "player_id": player_id,
                    "player_name": player_name,
                    "raw_line": line,
                },
            )

        if line.startswith(":team:"):
            parts = line.split(":", 4)

            if len(parts) != 5:
                return Event(
                    type="eventlog_line",
                    data={"raw_line": line},
                )

            player_id = parts[2]
            team_id = parts[3]

            return Event(
                type="player_team_changed",
                data={
                    "player_id": player_id,
                    "player_name": self._player_name(player_id),
                    "team_id": team_id,
                    "team": TEAM_NAMES.get(team_id, team_id),
                    "join_type": parts[4],
                    "raw_line": line,
                },
            )

        if line.startswith(":kill:"):
            parts = line.split(":")

            if len(parts) < 6:
                return Event(
                    type="eventlog_line",
                    data={"raw_line": line},
                )

            kill_kind = parts[2]
            killer_id = parts[3]
            victim_id = parts[4]
            attributes = self._attributes(parts[5:])

            common = {
                "kill_kind": kill_kind,
                "killer_id": killer_id,
                "victim_id": victim_id,
                "killer": self._player_name(killer_id),
                "victim": self._player_name(victim_id),
                "death_type": attributes.get("type"),
                "killer_items": attributes.get("items"),
                "victim_items": attributes.get("victimitems"),
                "raw_line": line,
            }

            if kill_kind in {"suicide", "accident"}:
                return Event(
                    type="player_suicide",
                    data={
                        **common,
                        "player_id": victim_id,
                        "player_name": self._player_name(victim_id),
                    },
                )

            return Event(
                type="player_kill",
                data=common,
            )

        if line.startswith(":ctf:"):
            parts = line.split(":")
            action = parts[2]
            flag_color = (
                parts[3]
                if len(parts) > 3
                else None
            )

            # Real Xonotic CTF lines contain both a team ID
            # and a final player ID:
            #
            # :ctf:steal:5:14:1
            #
            # Older synthetic tests used the shorter form:
            #
            # :ctf:steal:5:1
            if len(parts) > 5:
                player_id = parts[5] or None
            elif len(parts) > 4:
                player_id = parts[4] or None
            else:
                player_id = None

            return Event(
                type="ctf_flag_event",
                data={
                    "action": action,
                    "flag_color": flag_color,
                    "player_id": player_id,
                    "player_name": (
                        self._player_name(player_id)
                        if player_id is not None
                        else None
                    ),
                    "raw_line": line,
                },
            )

        if line.startswith(":dom:taken:"):
            parts = line.split(":", 4)

            if len(parts) != 5:
                return Event(
                    type="eventlog_line",
                    data={"raw_line": line},
                )

            player_id = parts[4]

            return Event(
                type="domination_point_taken",
                data={
                    "previous_team_id": parts[3],
                    "previous_team": TEAM_NAMES.get(
                        parts[3],
                        parts[3],
                    ),
                    "player_id": player_id,
                    "player_name": self._player_name(player_id),
                    "raw_line": line,
                },
            )

        if line.startswith(":keyhunt:"):
            parts = line.split(":", 7)

            if len(parts) != 8:
                return Event(
                    type="eventlog_line",
                    data={"raw_line": line},
                )

            player_id = parts[3]
            owner_id = parts[5]

            return Event(
                type="keyhunt_event",
                data={
                    "action": parts[2],
                    "player_id": player_id,
                    "player_name": self._player_name(player_id),
                    "player_points": parts[4],
                    "key_owner_id": owner_id,
                    "key_owner_name": self._player_name(owner_id),
                    "key_owner_points": parts[6],
                    "key_name": parts[7],
                    "raw_line": line,
                },
            )

        if line.startswith(":scores:"):
            parts = line.split(":", 3)
            mode_data = self._mode_data(parts[2])
            self.current_game_mode = mode_data.get("game_mode")

            return Event(
                type="score_section_started",
                data={
                    **mode_data,
                    "runtime_seconds": parts[3],
                    "raw_line": line,
                },
            )

        if line.startswith(":labels:player:"):
            parts = line.split(":", 3)

            return Event(
                type="player_score_labels",
                data={
                    "labels": parts[3].split(","),
                    "raw_line": line,
                },
            )

        if line.startswith(":player:see-labels:"):
            parts = line.split(":", 7)

            if len(parts) != 8:
                return Event(
                    type="eventlog_line",
                    data={"raw_line": line},
                )

            player_id = parts[6]
            nickname = parts[7]
            self.player_names[player_id] = nickname

            return Event(
                type="player_score_snapshot",
                data={
                    "scores": parts[3].split(","),
                    "playtime_seconds": parts[4],
                    "team": parts[5],
                    "player_id": player_id,
                    "player_name": nickname,
                    "raw_line": line,
                },
            )

        if line.startswith(":labels:teamscores:"):
            parts = line.split(":", 3)

            return Event(
                type="team_score_labels",
                data={
                    "labels": parts[3].split(","),
                    "raw_line": line,
                },
            )

        if line.startswith(":teamscores:see-labels:"):
            parts = line.split(":", 4)
            team_id = parts[4]

            return Event(
                type="team_score_snapshot",
                data={
                    "scores": parts[3].split(","),
                    "team_id": team_id,
                    "team": TEAM_NAMES.get(team_id, team_id),
                    "raw_line": line,
                },
            )

        if line == ":end":
            return Event(
                type="score_section_ended",
                data={"raw_line": line},
            )

        if line == ":restart":
            return Event(
                type="match_restarted",
                data={"raw_line": line},
            )

        if line == ":gameover":
            return Event(
                type="match_ended",
                data={"raw_line": line},
            )

        if line.startswith(":recordset:"):
            parts = line.split(":", 3)
            player_id = parts[2]

            current_mode = getattr(
                self,
                "current_game_mode",
                None,
            )
            mode_value = getattr(
                current_mode,
                "value",
                current_mode,
            )

            event_type = (
                "ctf_capture_record_set"
                if str(mode_value).casefold() == "ctf"
                else "race_record_set"
            )

            return Event(
                type=event_type,
                data={
                    "player_id": player_id,
                    "player_name": self._player_name(
                        player_id
                    ),
                    "time_seconds": parts[3],
                    "raw_line": line,
                },
            )

        if line.startswith(":time:"):
            parts = line.split(":", 2)

            return Event(
                type="eventlog_time",
                data={
                    "value": parts[2],
                    "raw_line": line,
                },
            )

        return Event(
            type="eventlog_line",
            data={"raw_line": line},
        )
