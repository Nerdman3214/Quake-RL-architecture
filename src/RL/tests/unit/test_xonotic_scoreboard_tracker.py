"""Tests for generic mode-specific scoreboard interpretation."""

from RL.engine.server.eventlog_reader import XonoticEventLogReader
from RL.engine.server.scoreboard_tracker import (
    XonoticScoreboardTracker,
)
from RL.events import Event


def test_real_deathmatch_score_fields() -> None:
    reader = XonoticEventLogReader()
    tracker = XonoticScoreboardTracker()

    labels = reader.parse_line(
        ":labels:player:"
        "score!!,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,"
        "kills,deaths<,,suicides<,dmg,dmgtaken<,"
        "rounds_pl,elo,,,,,,,,,"
    )

    assert labels is not None
    tracker.process(labels)

    snapshot = reader.parse_line(
        ":player:see-labels:"
        "30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
        "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
        "30,10,0,0,5244.713867,1708.388062,0,-2,"
        "0,0,0,0,0,0,0,0,0:"
        "520:-1:1:Noobnog"
    )

    assert snapshot is not None
    enriched = tracker.process(snapshot)

    assert enriched.data["score_fields"]["score"] == 30
    assert enriched.data["score_fields"]["kills"] == 30
    assert enriched.data["score_fields"]["deaths"] == 10
    assert enriched.data["score_fields"]["suicides"] == 0
    assert enriched.data["primary_score_name"] == "score"
    assert enriched.data["primary_score"] == 30
    assert not enriched.data["primary_lower_is_better"]


def test_race_lower_time_is_better() -> None:
    tracker = XonoticScoreboardTracker()

    tracker.process(
        Event(
            type="player_score_labels",
            data={"labels": ["time<!!", "laps", "deaths<"]},
        )
    )

    enriched = tracker.process(
        Event(
            type="player_score_snapshot",
            data={
                "scores": ["42.125", "3", "1"],
                "player_name": "Noobnog",
            },
        )
    )

    assert enriched.data["score_fields"]["time"] == 42.125
    assert enriched.data["score_fields"]["laps"] == 3
    assert enriched.data["primary_score"] == 42.125
    assert enriched.data["primary_lower_is_better"]


def test_ctf_mode_specific_fields() -> None:
    tracker = XonoticScoreboardTracker()

    tracker.process(
        Event(
            type="player_score_labels",
            data={
                "labels": [
                    "score!!",
                    "caps",
                    "pickups",
                    "returns",
                    "deaths<",
                ]
            },
        )
    )

    enriched = tracker.process(
        Event(
            type="player_score_snapshot",
            data={
                "scores": ["25", "2", "4", "1", "7"],
                "player_name": "Noobnog",
            },
        )
    )

    assert enriched.data["score_fields"] == {
        "score": 25,
        "caps": 2,
        "pickups": 4,
        "returns": 1,
        "deaths": 7,
    }


def test_team_score_fields() -> None:
    tracker = XonoticScoreboardTracker()

    tracker.process(
        Event(
            type="team_score_labels",
            data={"labels": ["score!!", "caps"]},
        )
    )

    enriched = tracker.process(
        Event(
            type="team_score_snapshot",
            data={
                "scores": ["10", "3"],
                "team": "RED",
            },
        )
    )

    assert enriched.data["score_fields"]["score"] == 10
    assert enriched.data["score_fields"]["caps"] == 3
