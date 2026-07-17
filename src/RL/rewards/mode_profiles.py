"""Reward-policy metadata for every official Xonotic game mode."""

from __future__ import annotations

from dataclasses import dataclass

from RL.env.multiplayer import GameMode, normalize_game_mode


@dataclass(frozen=True)
class ModeRewardProfile:
    """High-level reward behavior for one game mode."""

    mode: GameMode
    objective_family: str

    reward_frags: bool = True
    penalize_deaths: bool = True
    penalize_team_kills: bool = False

    round_based: bool = False
    lower_primary_score_is_better: bool = False


MODE_REWARD_PROFILES = {
    GameMode.DEATHMATCH: ModeRewardProfile(
        mode=GameMode.DEATHMATCH,
        objective_family="combat",
    ),
    GameMode.TEAM_DEATHMATCH: ModeRewardProfile(
        mode=GameMode.TEAM_DEATHMATCH,
        objective_family="combat",
        penalize_team_kills=True,
    ),
    GameMode.CAPTURE_THE_FLAG: ModeRewardProfile(
        mode=GameMode.CAPTURE_THE_FLAG,
        objective_family="ctf",
        penalize_team_kills=True,
    ),
    GameMode.CLAN_ARENA: ModeRewardProfile(
        mode=GameMode.CLAN_ARENA,
        objective_family="round",
        penalize_team_kills=True,
        round_based=True,
    ),
    GameMode.FREEZE_TAG: ModeRewardProfile(
        mode=GameMode.FREEZE_TAG,
        objective_family="freeze_tag",
        penalize_team_kills=True,
        round_based=True,
    ),
    GameMode.KEY_HUNT: ModeRewardProfile(
        mode=GameMode.KEY_HUNT,
        objective_family="keyhunt",
        penalize_team_kills=True,
    ),
    GameMode.ASSAULT: ModeRewardProfile(
        mode=GameMode.ASSAULT,
        objective_family="assault",
        penalize_team_kills=True,
    ),
    GameMode.DOMINATION: ModeRewardProfile(
        mode=GameMode.DOMINATION,
        objective_family="domination",
        penalize_team_kills=True,
    ),
    GameMode.LAST_MAN_STANDING: ModeRewardProfile(
        mode=GameMode.LAST_MAN_STANDING,
        objective_family="survival",
        round_based=True,
    ),
    GameMode.KEEPAWAY: ModeRewardProfile(
        mode=GameMode.KEEPAWAY,
        objective_family="keepaway",
    ),
    GameMode.INVASION: ModeRewardProfile(
        mode=GameMode.INVASION,
        objective_family="invasion",
        penalize_team_kills=True,
    ),
    GameMode.ONSLAUGHT: ModeRewardProfile(
        mode=GameMode.ONSLAUGHT,
        objective_family="onslaught",
        penalize_team_kills=True,
    ),
    GameMode.RACE: ModeRewardProfile(
        mode=GameMode.RACE,
        objective_family="race",
        reward_frags=False,
        penalize_deaths=False,
        lower_primary_score_is_better=True,
    ),
    GameMode.COMPLETE_THE_STAGE: ModeRewardProfile(
        mode=GameMode.COMPLETE_THE_STAGE,
        objective_family="race",
        reward_frags=False,
        penalize_deaths=False,
        lower_primary_score_is_better=True,
    ),
    GameMode.NEXBALL: ModeRewardProfile(
        mode=GameMode.NEXBALL,
        objective_family="nexball",
        reward_frags=False,
        penalize_deaths=False,
        penalize_team_kills=True,
    ),
}


def get_mode_reward_profile(
    mode: GameMode | str,
) -> ModeRewardProfile:
    """Return the profile for a mode identifier or alias."""

    resolved = (
        mode
        if isinstance(mode, GameMode)
        else normalize_game_mode(mode)
    )

    return MODE_REWARD_PROFILES[resolved]
