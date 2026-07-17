"""Tests for the configurable Xonotic live recorder."""

from pathlib import Path

import pytest

from RL.tools.launch.live_event_recording_test import (
    build_command,
    parse_args,
    write_server_config,
)


def test_default_arguments() -> None:
    args = parse_args([])

    assert args.mode == "dm"
    assert args.map_name == "boil"
    assert args.port == 26000
    assert args.max_players == 4
    assert args.bots == 3
    assert args.skill == 4


def test_mode_alias_is_normalized() -> None:
    args = parse_args(
        [
            "--mode",
            "capture the flag",
            "--map",
            "runningmanctf",
        ]
    )

    assert args.mode == "ctf"
    assert args.map_name == "runningmanctf"


def test_server_config_contains_mode_and_map(
    tmp_path: Path,
) -> None:
    path = write_server_config(
        mode="ctf",
        map_name="runningmanctf",
        directory=tmp_path,
    )

    content = path.read_text(encoding="utf-8")

    assert 'gametype "ctf"' in content
    assert 'g_maplist "runningmanctf"' in content
    assert 'map "runningmanctf"' in content


def test_command_uses_serverconfig() -> None:
    command = build_command(
        server_config_name="rl_event_recorder.cfg",
        port=26000,
        max_players=8,
        bot_count=7,
        bot_skill=4,
    )

    index = command.index("+serverconfig")

    assert command[index + 1] == (
        "rl_event_recorder.cfg"
    )
    assert "+gametype" not in command
    assert "+map" not in command


def test_bots_must_leave_one_client_slot() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--max-players",
                "8",
                "--bots",
                "8",
            ]
        )
