"""Tests for all official Xonotic game modes."""

import pytest

from RL.env.multiplayer import (
    GAME_MODE_SPECS,
    GameMode,
    get_game_mode_spec,
    normalize_game_mode,
    split_mode_and_map,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("deathmatch", GameMode.DEATHMATCH),
        ("tdm", GameMode.TEAM_DEATHMATCH),
        ("capture the flag", GameMode.CAPTURE_THE_FLAG),
        ("clan arena", GameMode.CLAN_ARENA),
        ("freeze tag", GameMode.FREEZE_TAG),
        ("key hunt", GameMode.KEY_HUNT),
        ("assault", GameMode.ASSAULT),
        ("domination", GameMode.DOMINATION),
        ("last man standing", GameMode.LAST_MAN_STANDING),
        ("keepaway", GameMode.KEEPAWAY),
        ("invasion", GameMode.INVASION),
        ("onslaught", GameMode.ONSLAUGHT),
        ("race", GameMode.RACE),
        ("complete the stage", GameMode.COMPLETE_THE_STAGE),
        ("nexball", GameMode.NEXBALL),
    ],
)
def test_all_official_modes(
    value: str,
    expected: GameMode,
) -> None:
    assert normalize_game_mode(value) is expected


def test_registry_contains_every_mode() -> None:
    assert set(GAME_MODE_SPECS) == set(GameMode)


def test_split_mode_and_map() -> None:
    mode, map_name = split_mode_and_map("ctf_afterslime")

    assert mode is GameMode.CAPTURE_THE_FLAG
    assert map_name == "afterslime"


def test_mode_metadata() -> None:
    spec = get_game_mode_spec("race")

    assert spec.family == "race"
    assert spec.time_or_progress_based
    assert not spec.team_based
