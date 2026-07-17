"""Translate authoritative match events into player rewards."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from RL.env.multiplayer import GameMode, normalize_game_mode
from RL.events import Event
from RL.rewards.contracts import RewardLedger, RewardWeights
from RL.rewards.mode_profiles import (
    ModeRewardProfile,
    get_mode_reward_profile,
)


class RewardMapper:
    """Map normalized events to rewards for one controlled player."""

    _NON_PLAYING_TEAMS = {
        "",
        "-1",
        "none",
        "spectator",
    }

    def __init__(
        self,
        controlled_player: str,
        controlled_team: Optional[str] = None,
        weights: Optional[RewardWeights] = None,
    ) -> None:
        if not controlled_player.strip():
            raise ValueError(
                "controlled_player must not be blank"
            )

        self.controlled_player = controlled_player
        self.controlled_team = controlled_team
        self.weights = weights or RewardWeights()
        self.current_mode: Optional[GameMode] = None
        self._win_reward_awarded = False

    @property
    def current_profile(
        self,
    ) -> Optional[ModeRewardProfile]:
        if self.current_mode is None:
            return None

        return get_mode_reward_profile(self.current_mode)

    def _is_controlled_player(self, value: object) -> bool:
        return value == self.controlled_player

    def _is_controlled_team(self, value: object) -> bool:
        if (
            self.controlled_team is None
            or not isinstance(value, str)
        ):
            return False

        return (
            value.casefold()
            == self.controlled_team.casefold()
        )

    def _update_controlled_team(
        self,
        player_name: object,
        team: object,
    ) -> None:
        if not self._is_controlled_player(player_name):
            return

        if not isinstance(team, str):
            return

        normalized = team.strip().casefold()

        if normalized in self._NON_PLAYING_TEAMS:
            self.controlled_team = None
        else:
            self.controlled_team = team

    def _combat_rewards_enabled(self) -> bool:
        profile = self.current_profile

        return (
            profile is None
            or profile.reward_frags
        )

    def _death_penalties_enabled(self) -> bool:
        profile = self.current_profile

        return (
            profile is None
            or profile.penalize_deaths
        )

    def _team_kill_penalty_enabled(self) -> bool:
        profile = self.current_profile

        return (
            profile is None
            or profile.penalize_team_kills
        )

    def _objective_family_enabled(
        self,
        family: str,
    ) -> bool:
        profile = self.current_profile

        return (
            profile is None
            or profile.objective_family == family
        )

    def _set_match_mode(self, value: object) -> None:
        self.controlled_team = None
        self._win_reward_awarded = False

        if not isinstance(value, str):
            self.current_mode = None
            return

        try:
            self.current_mode = normalize_game_mode(value)
        except ValueError:
            self.current_mode = None

    def _reward_win_once(self) -> RewardLedger:
        if self._win_reward_awarded:
            return RewardLedger()

        self._win_reward_awarded = True

        return RewardLedger(win=self.weights.win)

    def _map_player_kill(
        self,
        data: dict,
    ) -> RewardLedger:
        killer = data.get("killer")
        victim = data.get("victim")
        kill_kind = str(
            data.get("kill_kind", "frag")
        ).casefold()

        if (
            self._is_controlled_player(killer)
            and self._is_controlled_player(victim)
        ):
            if self._death_penalties_enabled():
                return RewardLedger(
                    suicide=self.weights.suicide
                )

            return RewardLedger()

        if kill_kind == "tk":
            if (
                self._is_controlled_player(killer)
                and self._team_kill_penalty_enabled()
            ):
                return RewardLedger(
                    team_kill=self.weights.team_kill
                )

            if (
                self._is_controlled_player(victim)
                and self._death_penalties_enabled()
            ):
                return RewardLedger(
                    death=self.weights.death
                )

            return RewardLedger()

        if (
            self._is_controlled_player(killer)
            and self._combat_rewards_enabled()
        ):
            return RewardLedger(
                frag=self.weights.frag
            )

        if (
            self._is_controlled_player(victim)
            and self._death_penalties_enabled()
        ):
            return RewardLedger(
                death=self.weights.death
            )

        return RewardLedger()

    def _map_ctf_event(
        self,
        data: dict,
    ) -> RewardLedger:
        if not self._objective_family_enabled("ctf"):
            return RewardLedger()

        if not self._is_controlled_player(
            data.get("player_name")
        ):
            return RewardLedger()

        rewards = {
            "steal": self.weights.ctf_steal,
            "pickup": self.weights.ctf_pickup,
            "return": self.weights.ctf_return,
            "capture": self.weights.ctf_capture,
            "dropped": self.weights.ctf_drop,
        }

        value = rewards.get(
            str(data.get("action", "")).casefold(),
            0.0,
        )

        return RewardLedger(ctf=value)

    def _map_keyhunt_event(
        self,
        data: dict,
    ) -> RewardLedger:
        if not self._objective_family_enabled("keyhunt"):
            return RewardLedger()

        if not self._is_controlled_player(
            data.get("player_name")
        ):
            return RewardLedger()

        rewards = {
            "capture": self.weights.keyhunt_capture,
            "carrierfrag": (
                self.weights.keyhunt_carrier_frag
            ),
            "collect": self.weights.keyhunt_collect,
            "destroyed": self.weights.keyhunt_destroyed,
            "destroyed_holdingkey": (
                self.weights.keyhunt_destroyed_holding_key
            ),
            "dropkey": self.weights.keyhunt_drop,
            "losekey": self.weights.keyhunt_lose,
            "push": self.weights.keyhunt_push,
            "pushed": self.weights.keyhunt_pushed,
        }

        value = rewards.get(
            str(data.get("action", "")).casefold(),
            0.0,
        )

        return RewardLedger(keyhunt=value)

    def map_event(self, event: Event) -> RewardLedger:
        """Return the reward generated by one event."""

        if not isinstance(event, Event):
            raise TypeError(
                "event must be an Event instance"
            )

        event_type = event.type
        data = event.data

        if event_type == "match_started":
            self._set_match_mode(data.get("game_mode"))
            return RewardLedger()

        if event_type == "match_restarted":
            self._win_reward_awarded = False
            return RewardLedger()

        if event_type in {
            "player_now_playing",
            "player_team_changed",
        }:
            self._update_controlled_team(
                data.get("player_name"),
                data.get("team"),
            )
            return RewardLedger()

        if event_type == "player_kill":
            return self._map_player_kill(data)

        if event_type == "player_suicide":
            if (
                self._is_controlled_player(
                    data.get("player_name")
                )
                and self._death_penalties_enabled()
            ):
                return RewardLedger(
                    suicide=self.weights.suicide
                )

            return RewardLedger()

        if event_type == "ctf_flag_event":
            return self._map_ctf_event(data)

        if event_type == "domination_point_taken":
            if (
                self._objective_family_enabled(
                    "domination"
                )
                and self._is_controlled_player(
                    data.get("player_name")
                )
            ):
                return RewardLedger(
                    domination=(
                        self.weights.domination_capture
                    )
                )

            return RewardLedger()

        if event_type == "keyhunt_event":
            return self._map_keyhunt_event(data)

        if event_type == "control_point_captured":
            if self._is_controlled_team(data.get("team")):
                return RewardLedger(
                    objective=self.weights.objective
                )

            return RewardLedger()

        if event_type == "player_match_win":
            if self._is_controlled_player(
                data.get("player_name")
            ):
                return self._reward_win_once()

            return RewardLedger()

        if event_type == "team_match_win":
            if self._is_controlled_team(data.get("team")):
                return self._reward_win_once()

            return RewardLedger()

        if event_type == "item_pickup":
            if self._is_controlled_player(
                data.get("player_name")
            ):
                return RewardLedger(
                    item_pickup=self.weights.item_pickup
                )

            return RewardLedger()

        if event_type == "first_blood":
            if self._is_controlled_player(
                data.get("player_name")
            ):
                return RewardLedger(
                    first_blood=self.weights.first_blood
                )

            return RewardLedger()

        if event_type == "frag_streak":
            if self._is_controlled_player(
                data.get("player_name")
            ):
                return RewardLedger(
                    streak=self.weights.streak
                )

            return RewardLedger()

        if event_type == "player_disconnected":
            if self._is_controlled_player(
                data.get("player_name")
            ):
                return RewardLedger(
                    disconnect=self.weights.disconnect
                )

            return RewardLedger()

        return RewardLedger()

    def accumulate(
        self,
        events: Iterable[Event],
    ) -> RewardLedger:
        """Combine rewards from an ordered event sequence."""

        ledger = RewardLedger()

        for event in events:
            ledger = ledger + self.map_event(event)

        return ledger
