"""Tests for mode-aware controlled-player rewards."""

from RL.env.multiplayer import GameMode
from RL.events import Event
from RL.rewards import RewardMapper


def make_event(event_type: str, **data: object) -> Event:
    return Event(type=event_type, data=dict(data))


def start_mode(
    mapper: RewardMapper,
    mode: str,
) -> None:
    result = mapper.map_event(
        make_event(
            "match_started",
            game_mode=mode,
        )
    )

    assert result.is_zero


def test_race_suppresses_combat_rewards() -> None:
    mapper = RewardMapper("Noobnog")
    start_mode(mapper, "rc")

    frag = mapper.map_event(
        make_event(
            "player_kill",
            kill_kind="frag",
            killer="Noobnog",
            victim="[BOT]Dominator",
        )
    )

    death = mapper.map_event(
        make_event(
            "player_kill",
            kill_kind="frag",
            killer="[BOT]Dominator",
            victim="Noobnog",
        )
    )

    assert mapper.current_mode is GameMode.RACE
    assert frag.is_zero
    assert death.is_zero


def test_structured_team_change_tracks_player_team() -> None:
    mapper = RewardMapper("Noobnog")

    mapper.map_event(
        make_event(
            "player_team_changed",
            player_name="Noobnog",
            team="RED",
        )
    )

    assert mapper.controlled_team == "RED"

    mapper.map_event(
        make_event(
            "player_team_changed",
            player_name="Noobnog",
            team="SPECTATOR",
        )
    )

    assert mapper.controlled_team is None


def test_team_kill_and_team_kill_death() -> None:
    mapper = RewardMapper("Noobnog")
    start_mode(mapper, "tdm")

    controlled_team_kill = mapper.map_event(
        make_event(
            "player_kill",
            kill_kind="tk",
            killer="Noobnog",
            victim="[BOT]Partner",
        )
    )

    controlled_victim = mapper.map_event(
        make_event(
            "player_kill",
            kill_kind="tk",
            killer="[BOT]Partner",
            victim="Noobnog",
        )
    )

    assert controlled_team_kill.team_kill == -2.0
    assert controlled_team_kill.frag == 0.0
    assert controlled_victim.death == -1.0


def test_ctf_controlled_player_actions() -> None:
    mapper = RewardMapper("Noobnog")
    start_mode(mapper, "ctf")

    expected = {
        "steal": 0.25,
        "pickup": 0.10,
        "return": 0.50,
        "capture": 3.0,
        "dropped": -0.25,
    }

    for action, value in expected.items():
        result = mapper.map_event(
            make_event(
                "ctf_flag_event",
                action=action,
                player_name="Noobnog",
                flag_color="14",
            )
        )

        assert result.ctf == value


def test_ctf_bot_and_wrong_mode_are_zero() -> None:
    mapper = RewardMapper("Noobnog")
    start_mode(mapper, "ctf")

    bot_capture = mapper.map_event(
        make_event(
            "ctf_flag_event",
            action="capture",
            player_name="[BOT]Dominator",
        )
    )

    start_mode(mapper, "dm")

    wrong_mode = mapper.map_event(
        make_event(
            "ctf_flag_event",
            action="capture",
            player_name="Noobnog",
        )
    )

    assert bot_capture.is_zero
    assert wrong_mode.is_zero


def test_domination_capture() -> None:
    mapper = RewardMapper("Noobnog")
    start_mode(mapper, "dom")

    controlled = mapper.map_event(
        make_event(
            "domination_point_taken",
            player_name="Noobnog",
            previous_team="BLUE",
        )
    )

    bot = mapper.map_event(
        make_event(
            "domination_point_taken",
            player_name="[BOT]Dominator",
            previous_team="RED",
        )
    )

    assert controlled.domination == 1.0
    assert bot.is_zero


def test_keyhunt_confirmed_actions() -> None:
    mapper = RewardMapper("Noobnog")
    start_mode(mapper, "kh")

    expected = {
        "capture": 2.0,
        "carrierfrag": 1.0,
        "collect": 0.25,
        "destroyed": 0.50,
        "dropkey": -0.25,
        "losekey": -0.50,
    }

    for action, value in expected.items():
        result = mapper.map_event(
            make_event(
                "keyhunt_event",
                action=action,
                player_name="Noobnog",
            )
        )

        assert result.keyhunt == value


def test_unverified_keyhunt_actions_default_to_zero() -> None:
    mapper = RewardMapper("Noobnog")
    start_mode(mapper, "kh")

    for action in (
        "destroyed_holdingkey",
        "push",
        "pushed",
    ):
        result = mapper.map_event(
            make_event(
                "keyhunt_event",
                action=action,
                player_name="Noobnog",
            )
        )

        assert result.keyhunt == 0.0
        assert result.is_zero
