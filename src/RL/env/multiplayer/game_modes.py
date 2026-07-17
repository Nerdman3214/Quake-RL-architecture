"""Canonical Xonotic game-mode definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GameMode(str, Enum):
    """Official Xonotic base game-type identifiers."""

    DEATHMATCH = "dm"
    TEAM_DEATHMATCH = "tdm"
    CAPTURE_THE_FLAG = "ctf"
    CLAN_ARENA = "ca"
    FREEZE_TAG = "ft"
    KEY_HUNT = "kh"
    ASSAULT = "as"
    DOMINATION = "dom"
    LAST_MAN_STANDING = "lms"
    KEEPAWAY = "ka"
    INVASION = "inv"
    ONSLAUGHT = "ons"
    RACE = "rc"
    COMPLETE_THE_STAGE = "cts"
    NEXBALL = "nb"


@dataclass(frozen=True)
class GameModeSpec:
    """Stable metadata describing one base game mode."""

    mode: GameMode
    display_name: str
    family: str
    team_based: bool
    objective_based: bool
    time_or_progress_based: bool = False


GAME_MODE_SPECS = {
    GameMode.DEATHMATCH: GameModeSpec(
        GameMode.DEATHMATCH,
        "Deathmatch",
        "combat",
        team_based=False,
        objective_based=False,
    ),
    GameMode.TEAM_DEATHMATCH: GameModeSpec(
        GameMode.TEAM_DEATHMATCH,
        "Team Deathmatch",
        "combat",
        team_based=True,
        objective_based=False,
    ),
    GameMode.CAPTURE_THE_FLAG: GameModeSpec(
        GameMode.CAPTURE_THE_FLAG,
        "Capture the Flag",
        "possession",
        team_based=True,
        objective_based=True,
    ),
    GameMode.CLAN_ARENA: GameModeSpec(
        GameMode.CLAN_ARENA,
        "Clan Arena",
        "round_elimination",
        team_based=True,
        objective_based=False,
    ),
    GameMode.FREEZE_TAG: GameModeSpec(
        GameMode.FREEZE_TAG,
        "Freeze Tag",
        "round_elimination",
        team_based=True,
        objective_based=True,
    ),
    GameMode.KEY_HUNT: GameModeSpec(
        GameMode.KEY_HUNT,
        "Key Hunt",
        "possession",
        team_based=True,
        objective_based=True,
    ),
    GameMode.ASSAULT: GameModeSpec(
        GameMode.ASSAULT,
        "Assault",
        "territory",
        team_based=True,
        objective_based=True,
    ),
    GameMode.DOMINATION: GameModeSpec(
        GameMode.DOMINATION,
        "Domination",
        "territory",
        team_based=True,
        objective_based=True,
    ),
    GameMode.LAST_MAN_STANDING: GameModeSpec(
        GameMode.LAST_MAN_STANDING,
        "Last Man Standing",
        "round_elimination",
        team_based=False,
        objective_based=False,
    ),
    GameMode.KEEPAWAY: GameModeSpec(
        GameMode.KEEPAWAY,
        "Keepaway",
        "possession",
        team_based=False,
        objective_based=True,
    ),
    GameMode.INVASION: GameModeSpec(
        GameMode.INVASION,
        "Invasion",
        "cooperative",
        team_based=True,
        objective_based=True,
    ),
    GameMode.ONSLAUGHT: GameModeSpec(
        GameMode.ONSLAUGHT,
        "Onslaught",
        "territory",
        team_based=True,
        objective_based=True,
    ),
    GameMode.RACE: GameModeSpec(
        GameMode.RACE,
        "Race",
        "race",
        team_based=False,
        objective_based=True,
        time_or_progress_based=True,
    ),
    GameMode.COMPLETE_THE_STAGE: GameModeSpec(
        GameMode.COMPLETE_THE_STAGE,
        "Complete the Stage",
        "race",
        team_based=False,
        objective_based=True,
        time_or_progress_based=True,
    ),
    GameMode.NEXBALL: GameModeSpec(
        GameMode.NEXBALL,
        "Nexball",
        "ball",
        team_based=True,
        objective_based=True,
    ),
}


_ALIASES = {
    "dm": GameMode.DEATHMATCH,
    "deathmatch": GameMode.DEATHMATCH,
    "tdm": GameMode.TEAM_DEATHMATCH,
    "team deathmatch": GameMode.TEAM_DEATHMATCH,
    "ctf": GameMode.CAPTURE_THE_FLAG,
    "capture the flag": GameMode.CAPTURE_THE_FLAG,
    "ca": GameMode.CLAN_ARENA,
    "clan arena": GameMode.CLAN_ARENA,
    "ft": GameMode.FREEZE_TAG,
    "freeze tag": GameMode.FREEZE_TAG,
    "kh": GameMode.KEY_HUNT,
    "key hunt": GameMode.KEY_HUNT,
    "as": GameMode.ASSAULT,
    "assault": GameMode.ASSAULT,
    "dom": GameMode.DOMINATION,
    "domination": GameMode.DOMINATION,
    "lms": GameMode.LAST_MAN_STANDING,
    "last man standing": GameMode.LAST_MAN_STANDING,
    "ka": GameMode.KEEPAWAY,
    "keepaway": GameMode.KEEPAWAY,
    "inv": GameMode.INVASION,
    "invasion": GameMode.INVASION,
    "ons": GameMode.ONSLAUGHT,
    "onslaught": GameMode.ONSLAUGHT,
    "rc": GameMode.RACE,
    "race": GameMode.RACE,
    "cts": GameMode.COMPLETE_THE_STAGE,
    "complete the stage": GameMode.COMPLETE_THE_STAGE,
    "nb": GameMode.NEXBALL,
    "nexball": GameMode.NEXBALL,
}


def normalize_game_mode(value: str) -> GameMode:
    """Convert a short identifier or readable name to a GameMode."""

    normalized = " ".join(
        value.strip().casefold().replace("_", " ").replace("-", " ").split()
    )

    try:
        return _ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"Unknown Xonotic game mode: {value!r}") from error


def split_mode_and_map(value: str) -> tuple[GameMode | None, str | None]:
    """Split an eventlog token such as ``ctf_afterslime``."""

    normalized = value.strip().casefold()

    for mode in GameMode:
        prefix = f"{mode.value}_"

        if normalized.startswith(prefix):
            return mode, value[len(prefix):]

    return None, None


def get_game_mode_spec(mode: GameMode | str) -> GameModeSpec:
    """Return stable metadata for a mode or alias."""

    resolved = (
        mode
        if isinstance(mode, GameMode)
        else normalize_game_mode(mode)
    )

    return GAME_MODE_SPECS[resolved]
