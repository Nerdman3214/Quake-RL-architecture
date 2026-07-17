"""Tests for controlled-player reward accounting."""

from RL.events import Event
from RL.rewards import RewardLedger, RewardMapper, RewardWeights


def make_event(event_type: str, **data: object) -> Event:
    return Event(type=event_type, data=dict(data))


def test_reward_ledger_total_components_and_addition() -> None:
    first = RewardLedger(frag=1.0, death=-1.0)
    second = RewardLedger(win=5.0, item_pickup=0.05)

    combined = first + second

    assert combined.frag == 1.0
    assert combined.death == -1.0
    assert combined.win == 5.0
    assert combined.item_pickup == 0.05
    assert combined.total == 5.05
    assert combined.components["win"] == 5.0
    assert not combined.is_zero


def test_frag_death_and_bot_only_events() -> None:
    mapper = RewardMapper("Noobnog")

    frag = mapper.map_event(
        make_event(
            "player_kill",
            killer="Noobnog",
            victim="[BOT]Dominator",
            weapon="Vortex",
        )
    )

    death = mapper.map_event(
        make_event(
            "player_kill",
            killer="[BOT]Dominator",
            victim="Noobnog",
            weapon="Vortex",
        )
    )

    bot_only = mapper.map_event(
        make_event(
            "player_kill",
            killer="[BOT]Dominator",
            victim="[BOT]Pegasus",
            weapon="Shotgun",
        )
    )

    assert frag.frag == 1.0
    assert frag.total == 1.0

    assert death.death == -1.0
    assert death.total == -1.0

    assert bot_only.is_zero


def test_controlled_player_suicide() -> None:
    mapper = RewardMapper("Noobnog")

    result = mapper.map_event(
        make_event(
            "player_suicide",
            player_name="Noobnog",
            weapon="Mortar",
        )
    )

    assert result.suicide == -1.5
    assert result.death == 0.0
    assert result.total == -1.5


def test_small_bonus_events() -> None:
    mapper = RewardMapper(
        "Noobnog",
        weights=RewardWeights(
            item_pickup=0.1,
            first_blood=0.5,
            streak=0.25,
        ),
    )

    pickup = mapper.map_event(
        make_event(
            "item_pickup",
            player_name="Noobnog",
            item="Strength",
        )
    )

    first_blood = mapper.map_event(
        make_event(
            "first_blood",
            player_name="Noobnog",
        )
    )

    streak = mapper.map_event(
        make_event(
            "frag_streak",
            player_name="Noobnog",
            streak="massacre",
        )
    )

    assert pickup.item_pickup == 0.1
    assert first_blood.first_blood == 0.5
    assert streak.streak == 0.25


def test_player_match_win() -> None:
    mapper = RewardMapper("Noobnog")

    controlled_win = mapper.map_event(
        make_event(
            "player_match_win",
            player_name="Noobnog",
        )
    )

    bot_win = mapper.map_event(
        make_event(
            "player_match_win",
            player_name="[BOT]Dominator",
        )
    )

    assert controlled_win.win == 5.0
    assert bot_win.is_zero


def test_team_rewards_after_team_is_observed() -> None:
    mapper = RewardMapper("Noobnog")

    join_event = make_event(
        "player_now_playing",
        player_name="Noobnog",
        team="YELLOW",
    )

    assert mapper.map_event(join_event).is_zero
    assert mapper.controlled_team == "YELLOW"

    objective = mapper.map_event(
        make_event(
            "control_point_captured",
            team="yellow",
            objective="Alpha",
        )
    )

    win = mapper.map_event(
        make_event(
            "team_match_win",
            team="YELLOW",
        )
    )

    enemy_objective = mapper.map_event(
        make_event(
            "control_point_captured",
            team="PINK",
            objective="Beta",
        )
    )

    assert objective.objective == 2.0
    assert win.win == 5.0
    assert enemy_objective.is_zero


def test_controlled_disconnect() -> None:
    mapper = RewardMapper("Noobnog")

    mapper.map_event(
        make_event(
            "match_started",
            game_mode="dm",
        )
    )

    controlled = mapper.map_event(
        make_event(
            "player_disconnected",
            player_name="Noobnog",
        )
    )

    bot = mapper.map_event(
        make_event(
            "player_disconnected",
            player_name="[BOT]Eureka",
        )
    )

    assert controlled.disconnect == -2.0
    assert bot.is_zero


def test_accumulate_sequence() -> None:
    mapper = RewardMapper("Noobnog")

    events = [
        make_event(
            "player_kill",
            killer="Noobnog",
            victim="[BOT]Eureka",
        ),
        make_event(
            "player_kill",
            killer="Noobnog",
            victim="[BOT]Dominator",
        ),
        make_event(
            "player_kill",
            killer="[BOT]Pegasus",
            victim="Noobnog",
        ),
        make_event(
            "player_match_win",
            player_name="Noobnog",
        ),
    ]

    result = mapper.accumulate(events)

    assert result.frag == 2.0
    assert result.death == -1.0
    assert result.win == 5.0
    assert result.total == 6.0


def test_default_first_blood_reward() -> None:
    mapper = RewardMapper("Noobnog")

    result = mapper.map_event(
        make_event(
            "first_blood",
            player_name="Noobnog",
        )
    )

    assert result.first_blood == 0.25
    assert result.total == 0.25



def test_disconnect_penalty_only_during_active_match() -> None:
    mapper = RewardMapper("Noobnog")

    mapper.map_event(
        Event(
            type="match_started",
            data={"game_mode": "dm"},
        )
    )

    active_disconnect = mapper.map_event(
        Event(
            type="player_disconnected",
            data={"player_name": "Noobnog"},
        )
    )

    assert active_disconnect.disconnect == -2.0

    mapper.map_event(
        Event(
            type="match_started",
            data={"game_mode": "dm"},
        )
    )
    mapper.map_event(
        Event(
            type="match_ended",
            data={},
        )
    )

    completed_disconnect = mapper.map_event(
        Event(
            type="player_disconnected",
            data={"player_name": "Noobnog"},
        )
    )

    assert completed_disconnect.disconnect == 0.0
    assert completed_disconnect.is_zero
