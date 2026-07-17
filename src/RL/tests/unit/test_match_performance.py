"""Tests for scoreboard-grounded performance extraction."""

import json
from pathlib import Path

import pytest

from RL.matchmaking import (
    extract_match_performance,
    extract_match_performance_from_path,
)


def event(
    kind: str,
    data: dict,
) -> dict:
    return {
        "type": kind,
        "data": data,
    }


def match(
    mode: str,
    controlled: dict,
    teammate: dict,
    ours: int,
    theirs: int,
    winner: str,
) -> list[dict]:
    return [
        event(
            "session_started",
            {
                "session_id": "test-session",
                "bot_skill": 4,
                "bot_count": 7,
            },
        ),
        event(
            "match_started",
            {
                "game_mode": mode,
            },
        ),
        event(
            "player_score_snapshot",
            {
                "player_id": "1",
                "player_name": "Noobnog",
                "team": "14",
                "score_fields": controlled,
            },
        ),
        event(
            "player_score_snapshot",
            {
                "player_id": "2",
                "player_name": "Teammate",
                "team": "14",
                "score_fields": teammate,
            },
        ),
        event(
            "team_score_snapshot",
            {
                "team_id": "1",
                "team": "NONE",
                "primary_score": "",
            },
        ),
        event(
            "team_score_snapshot",
            {
                "team_id": "5",
                "team": "RED",
                "primary_score": theirs,
                "primary_lower_is_better": False,
            },
        ),
        event(
            "team_score_snapshot",
            {
                "team_id": "14",
                "team": "BLUE",
                "primary_score": ours,
                "primary_lower_is_better": False,
            },
        ),
        event(
            "match_ended",
            {},
        ),
        event(
            "team_match_win",
            {
                "team": winner,
            },
        ),
    ]


@pytest.mark.parametrize(
    (
        "records",
        "won",
        "margin",
        "combat",
        "discipline",
    ),
    [
        (
            match(
                "ctf",
                {
                    "caps": 3,
                    "fckills": 2,
                    "drops": 1,
                    "pickups": 4,
                    "kills": 10,
                    "deaths": 5,
                },
                {
                    "caps": 1,
                    "kills": 5,
                    "deaths": 5,
                },
                7,
                2,
                "BLUE",
            ),
            True,
            5 / 7,
            5 / 15,
            0.0,
        ),
        (
            match(
                "dom",
                {
                    "takes": 5,
                    "ticks": 4,
                    "kills": 4,
                    "deaths": 8,
                    "suicides": 1,
                },
                {
                    "takes": 15,
                    "ticks": 4,
                    "kills": 8,
                    "deaths": 4,
                },
                100,
                120,
                "RED",
            ),
            False,
            -1 / 6,
            -4 / 12,
            1 / 13,
        ),
        (
            match(
                "kh",
                {
                    "caps": 2,
                    "kckills": 1,
                    "losses": 3,
                    "destroyed": 1,
                    "pickups": 4,
                    "kills": 5,
                    "deaths": 7,
                    "teamkills": 1,
                },
                {
                    "caps": 4,
                    "kckills": 2,
                    "losses": 1,
                    "pickups": 4,
                    "kills": 8,
                    "deaths": 4,
                },
                1000,
                564,
                "BLUE",
            ),
            True,
            436 / 1000,
            -2 / 12,
            2 / 13,
        ),
    ],
)
def test_supported_mode_performance(
    records: list[dict],
    won: bool,
    margin: float,
    combat: float,
    discipline: float,
) -> None:
    result = extract_match_performance(
        records
    )

    assert result.won is won
    assert result.score_margin == pytest.approx(
        margin
    )
    assert result.combat_score == pytest.approx(
        combat
    )
    assert (
        0.0
        <= result.objective_score
        <= 1.0
    )
    assert (
        result.discipline_penalty
        == pytest.approx(discipline)
    )
    assert result.bot_skill == 4
    assert result.bot_count == 7


def test_incomplete_match_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Completed match index",
    ):
        extract_match_performance(
            [
                event(
                    "match_started",
                    {
                        "game_mode": "kh",
                    },
                )
            ]
        )


def test_missing_controlled_scoreboard_is_rejected() -> None:
    records = match(
        "kh",
        {
            "kills": 1,
            "deaths": 1,
        },
        {
            "kills": 1,
            "deaths": 1,
        },
        10,
        5,
        "BLUE",
    )

    records = [
        record
        for record in records
        if record["data"].get(
            "player_name"
        )
        != "Noobnog"
    ]

    with pytest.raises(
        ValueError,
        match="Final scoreboard",
    ):
        extract_match_performance(
            records
        )


def test_path_uses_filename_as_session_fallback(
    tmp_path: Path,
) -> None:
    records = match(
        "ctf",
        {
            "caps": 1,
            "kills": 2,
            "deaths": 1,
        },
        {
            "caps": 1,
            "kills": 1,
            "deaths": 1,
        },
        2,
        1,
        "BLUE",
    )

    records[0]["data"].pop(
        "session_id"
    )

    path = (
        tmp_path
        / "session_fallback-id.jsonl"
    )

    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )

    result = (
        extract_match_performance_from_path(
            path
        )
    )

    assert result.session_id == "fallback-id"
