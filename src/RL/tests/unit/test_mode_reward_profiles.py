"""Tests for the complete game-mode reward registry."""

from RL.env.multiplayer import GameMode
from RL.rewards import (
    MODE_REWARD_PROFILES,
    get_mode_reward_profile,
)


def test_every_game_mode_has_a_reward_profile() -> None:
    assert set(MODE_REWARD_PROFILES) == set(GameMode)


def test_specialized_profile_metadata() -> None:
    ctf = get_mode_reward_profile("ctf")
    race = get_mode_reward_profile("race")
    nexball = get_mode_reward_profile("nexball")

    assert ctf.objective_family == "ctf"
    assert ctf.penalize_team_kills

    assert race.objective_family == "race"
    assert not race.reward_frags
    assert not race.penalize_deaths
    assert race.lower_primary_score_is_better

    assert nexball.objective_family == "nexball"
    assert not nexball.reward_frags
