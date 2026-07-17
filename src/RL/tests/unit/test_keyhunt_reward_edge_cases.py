from RL.events import Event
from RL.rewards import RewardMapper


def start_keyhunt(mapper: RewardMapper) -> None:
    mapper.map_event(
        Event(
            type="match_started",
            data={"game_mode": "kh"},
        )
    )


def test_negative_carrierfrag_points_are_not_rewarded() -> None:
    mapper = RewardMapper("Noobnog")
    start_keyhunt(mapper)

    teamkill_carrierfrag = mapper.map_event(
        Event(
            type="keyhunt_event",
            data={
                "action": "carrierfrag",
                "player_name": "Noobnog",
                "player_points": "-3",
                "key_owner_name": "[BOT]Resurrection",
            },
        )
    )

    valid_carrierfrag = mapper.map_event(
        Event(
            type="keyhunt_event",
            data={
                "action": "carrierfrag",
                "player_name": "Noobnog",
                "player_points": "1",
                "key_owner_name": "[BOT]Eureka",
            },
        )
    )

    assert teamkill_carrierfrag.is_zero
    assert teamkill_carrierfrag.keyhunt == 0.0
    assert valid_carrierfrag.keyhunt == 1.0


def test_keyhunt_keys_use_structured_collect_reward_only() -> None:
    mapper = RewardMapper("Noobnog")
    start_keyhunt(mapper)

    supplemental_pickup = mapper.map_event(
        Event(
            type="item_pickup",
            data={
                "player_name": "Noobnog",
                "item": "the BLUE Key",
                "raw_line": "Noobnog picked up the BLUE Key",
            },
        )
    )

    structured_collect = mapper.map_event(
        Event(
            type="keyhunt_event",
            data={
                "action": "collect",
                "player_name": "Noobnog",
                "player_points": "3",
                "key_name": "blue key",
            },
        )
    )

    assert supplemental_pickup.is_zero
    assert supplemental_pickup.item_pickup == 0.0
    assert structured_collect.keyhunt == 0.25
