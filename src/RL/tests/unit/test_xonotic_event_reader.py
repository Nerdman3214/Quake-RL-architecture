"""Regression tests for real Xonotic server messages."""

import pytest

from RL.engine.server.event_reader import XonoticEventReader


@pytest.mark.parametrize(
    ("line", "event_type", "expected"),
    [
        (
            "[BOT]Eureka felt the strong pull of Noobnog's Crylink",
            "player_kill",
            {
                "killer": "Noobnog",
                "victim": "[BOT]Eureka",
                "weapon": "Crylink",
            },
        ),
        (
            "[BOT]Eureka was pummeled by Noobnog's Hagar rockets",
            "player_kill",
            {
                "killer": "Noobnog",
                "victim": "[BOT]Eureka",
                "weapon": "Hagar",
            },
        ),
        (
            "Noobnog was cooked by Resurrection",
            "player_kill",
            {
                "killer": "Resurrection",
                "victim": "Noobnog",
                "weapon": "Cooked",
            },
        ),
        (
            "Noobnog played with tiny Hagar rockets, "
            "losing their 6 frag spree",
            "player_suicide",
            {
                "player_name": "Noobnog",
                "weapon": "Hagar",
            },
        ),
        (
            "Noobnog blew themself up with their own Mortar, "
            "losing their 2 frag spree",
            "player_suicide",
            {
                "player_name": "Noobnog",
                "weapon": "Mortar",
            },
        ),
        (
            "Noobnog was in the wrong place, "
            "losing their 3 frag spree",
            "player_suicide",
            {
                "player_name": "Noobnog",
                "weapon": "environment",
            },
        ),
        (
            "Noobnog started a MASSACRE!",
            "frag_streak",
            {
                "player_name": "Noobnog",
                "streak": "massacre",
            },
        ),
    ],
)
def test_real_xonotic_messages(
    line: str,
    event_type: str,
    expected: dict[str, str],
) -> None:
    reader = XonoticEventReader()
    event = reader.parse_line(line)

    assert event is not None
    assert event.type == event_type

    for key, value in expected.items():
        assert event.data[key] == value


def test_unknown_message_is_preserved() -> None:
    reader = XonoticEventReader()
    event = reader.parse_line(
        "You don't have enough ammo to reload the MachineGun"
    )

    assert event is not None
    assert event.type == "console_line"
    assert event.data["raw_line"] == (
        "You don't have enough ammo to reload the MachineGun"
    )


def test_grounded_kill() -> None:
    reader = XonoticEventReader()
    event = reader.parse_line(
        "[BOT]Pegasus was grounded by Noobnog"
    )

    assert event is not None
    assert event.type == "player_kill"
    assert event.data["killer"] == "Noobnog"
    assert event.data["victim"] == "[BOT]Pegasus"
    assert event.data["weapon"] == "Grounded"


def test_mayhem_streak() -> None:
    reader = XonoticEventReader()
    event = reader.parse_line("Noobnog executed MAYHEM!")

    assert event is not None
    assert event.type == "frag_streak"
    assert event.data["player_name"] == "Noobnog"
    assert event.data["streak"] == "mayhem"


def test_machinegun_without_space() -> None:
    reader = XonoticEventReader()
    event = reader.parse_line(
        "[BOT]Pegasus was riddled full of holes "
        "by Noobnog's MachineGun"
    )

    assert event is not None
    assert event.type == "player_kill"
    assert event.data["killer"] == "Noobnog"
    assert event.data["victim"] == "[BOT]Pegasus"
    assert event.data["weapon"] == "Machine Gun"
