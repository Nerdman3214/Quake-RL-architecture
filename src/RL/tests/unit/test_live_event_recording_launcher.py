"""Tests for the configurable Xonotic live recorder."""

from pathlib import Path

import pytest

from RL.matchmaking import AdaptiveState, save_state
from RL.tools.launch.live_event_recording_test import (
    build_command,
    parse_args,
    resolve_matchmaking,
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
    assert args.matchmaking == "fixed"
    assert tuple(args.bot_count_range) == (7, 15)
    assert tuple(args.skill_range) == (2, 7)
    assert args.seed is None
    assert args.adaptive_state.name == "agent_rating.json"


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


def test_fixed_matchmaking_preserves_values() -> None:
    args = parse_args(
        [
            "--max-players",
            "8",
            "--bots",
            "7",
            "--skill",
            "4",
        ]
    )

    selection = resolve_matchmaking(args)

    assert selection.policy == "fixed"
    assert selection.bot_count == 7
    assert selection.bot_skill == 4
    assert selection.seed is None


def test_randomized_matchmaking_is_reproducible() -> None:
    argv = [
        "--matchmaking",
        "randomized",
        "--max-players",
        "16",
        "--bots",
        "auto",
        "--bot-count-range",
        "7",
        "15",
        "--skill",
        "random",
        "--skill-range",
        "2",
        "7",
        "--seed",
        "12345",
    ]

    first = resolve_matchmaking(parse_args(argv))
    second = resolve_matchmaking(parse_args(argv))

    assert first == second
    assert first.policy == "randomized"
    assert first.bot_count == 13
    assert first.bot_skill == 7
    assert first.seed == 12345


def test_fixed_matchmaking_rejects_automatic_values() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--bots",
                "auto",
            ]
        )

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--skill",
                "random",
            ]
        )


def test_automatic_bot_range_leaves_client_slot() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--matchmaking",
                "randomized",
                "--max-players",
                "8",
                "--bots",
                "auto",
                "--bot-count-range",
                "7",
                "8",
            ]
        )


def test_adaptive_missing_state_uses_defaults_read_only(
    tmp_path: Path,
) -> None:
    state_path = (
        tmp_path
        / "missing"
        / "agent_rating.json"
    )

    args = parse_args(
        [
            "--matchmaking",
            "adaptive",
            "--max-players",
            "8",
            "--bots",
            "7",
            "--adaptive-state",
            str(state_path),
        ]
    )

    selection = resolve_matchmaking(args)

    assert selection.policy == "adaptive"
    assert selection.bot_count == 7
    assert selection.bot_skill == 4
    assert selection.adaptive_rating == 4.0
    assert selection.adaptive_matches == 0
    assert selection.adaptive_state_path == str(
        state_path.resolve()
    )
    assert not state_path.exists()


def test_adaptive_matchmaking_loads_persisted_skill(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "agent_rating.json"

    save_state(
        AdaptiveState(
            rating=5.75,
            current_skill=6,
            matches=12,
            last_adjustment_match=9,
        ),
        state_path,
    )

    before = state_path.read_text(
        encoding="utf-8"
    )

    args = parse_args(
        [
            "--matchmaking",
            "adaptive",
            "--max-players",
            "8",
            "--bots",
            "7",
            "--adaptive-state",
            str(state_path),
        ]
    )

    selection = resolve_matchmaking(args)

    assert selection.bot_count == 7
    assert selection.bot_skill == 6
    assert selection.adaptive_rating == 5.75
    assert selection.adaptive_matches == 12
    assert state_path.read_text(
        encoding="utf-8"
    ) == before


def test_adaptive_rejects_randomized_markers() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--matchmaking",
                "adaptive",
                "--bots",
                "auto",
            ]
        )

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--matchmaking",
                "adaptive",
                "--skill",
                "random",
            ]
        )


def test_adaptive_rejects_out_of_range_skill(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "agent_rating.json"

    save_state(
        AdaptiveState(
            rating=20.0,
            current_skill=20,
            matches=1,
        ),
        state_path,
    )

    args = parse_args(
        [
            "--matchmaking",
            "adaptive",
            "--max-players",
            "8",
            "--bots",
            "7",
            "--adaptive-state",
            str(state_path),
        ]
    )

    with pytest.raises(
        ValueError,
        match="outside the supported range",
    ):
        resolve_matchmaking(args)
