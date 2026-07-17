"""Tests for atomic adaptive session updates."""

import json
from pathlib import Path

from RL.matchmaking import (
    AdaptiveState,
    load_state,
    save_state,
    update_adaptive_state_from_session,
)


def record(
    event_type: str,
    data: dict,
) -> dict:
    return {
        "type": event_type,
        "data": data,
    }


def match_records() -> list[dict]:
    return [
        record(
            "match_started",
            {
                "game_mode": "kh",
            },
        ),
        record(
            "player_score_snapshot",
            {
                "player_id": "1",
                "player_name": "Noobnog",
                "team": "13",
                "score_fields": {
                    "caps": 3,
                    "kckills": 4,
                    "losses": 2,
                    "pickups": 5,
                    "kills": 12,
                    "deaths": 8,
                    "teamkills": 0,
                    "suicides": 1,
                },
            },
        ),
        record(
            "player_score_snapshot",
            {
                "player_id": "2",
                "player_name": "Teammate",
                "team": "13",
                "score_fields": {
                    "caps": 3,
                    "kckills": 2,
                    "losses": 2,
                    "pickups": 3,
                    "kills": 8,
                    "deaths": 8,
                    "teamkills": 0,
                    "suicides": 0,
                },
            },
        ),
        record(
            "team_score_snapshot",
            {
                "team_id": "5",
                "team": "RED",
                "primary_score": 500,
                "primary_lower_is_better": False,
            },
        ),
        record(
            "team_score_snapshot",
            {
                "team_id": "13",
                "team": "YELLOW",
                "primary_score": 1000,
                "primary_lower_is_better": False,
            },
        ),
        record(
            "team_match_win",
            {
                "team": "YELLOW",
            },
        ),
        record(
            "match_ended",
            {},
        ),
    ]


def session_records(
    *,
    session_id: str = "adaptive-session",
    matchmaking: str = "adaptive",
    match_count: int = 1,
) -> list[dict]:
    records = [
        record(
            "session_started",
            {
                "session_id": session_id,
                "matchmaking": matchmaking,
                "bot_skill": 6,
                "bot_count": 7,
            },
        )
    ]

    for _ in range(match_count):
        records.extend(match_records())

    records.append(
        record(
            "session_ended",
            {
                "session_id": session_id,
            },
        )
    )

    return records


def write_records(
    path: Path,
    records: list[dict],
) -> None:
    path.write_text(
        "\n".join(
            json.dumps(item)
            for item in records
        )
        + "\n",
        encoding="utf-8",
    )


def test_nonadaptive_session_is_ignored(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session.jsonl"
    state_path = tmp_path / "state.json"

    write_records(
        session_path,
        session_records(
            matchmaking="fixed",
        ),
    )

    result = update_adaptive_state_from_session(
        session_path=session_path,
        state_path=state_path,
    )

    assert result.status == "not_adaptive"
    assert not state_path.exists()


def test_incomplete_match_does_not_create_state(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session.jsonl"
    state_path = tmp_path / "state.json"

    records = session_records()
    records = [
        item
        for item in records
        if item["type"] != "match_ended"
    ]

    write_records(session_path, records)

    result = update_adaptive_state_from_session(
        session_path=session_path,
        state_path=state_path,
    )

    assert result.status == "no_completed_match"
    assert not state_path.exists()


def test_completed_match_updates_once(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session.jsonl"
    state_path = tmp_path / "state.json"

    write_records(session_path, session_records())

    save_state(
        AdaptiveState(
            rating=5.75,
            current_skill=6,
            matches=12,
            last_adjustment_match=9,
        ),
        state_path,
    )

    first = update_adaptive_state_from_session(
        session_path=session_path,
        state_path=state_path,
    )

    assert first.status == "updated"
    assert first.updated_matches == 1

    updated = load_state(state_path)

    assert updated.matches == 13
    assert updated.current_skill == 6
    assert updated.processed_match_keys == (
        "adaptive-session:0",
    )

    before = state_path.read_bytes()

    second = update_adaptive_state_from_session(
        session_path=session_path,
        state_path=state_path,
    )

    assert second.status == "duplicate"
    assert second.duplicate_matches == 1
    assert state_path.read_bytes() == before


def test_two_completed_matches_use_distinct_keys(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session.jsonl"
    state_path = tmp_path / "state.json"

    write_records(
        session_path,
        session_records(
            session_id="two-matches",
            match_count=2,
        ),
    )

    result = update_adaptive_state_from_session(
        session_path=session_path,
        state_path=state_path,
    )

    assert result.status == "updated"
    assert result.completed_matches == 2
    assert result.updated_matches == 2

    state = load_state(state_path)

    assert state.matches == 2
    assert state.processed_match_keys == (
        "two-matches:0",
        "two-matches:1",
    )


def test_extraction_failure_does_not_modify_state(
    tmp_path: Path,
) -> None:
    session_path = tmp_path / "session.jsonl"
    state_path = tmp_path / "state.json"

    records = session_records()
    records = [
        item
        for item in records
        if not (
            item["type"]
            == "player_score_snapshot"
            and item["data"].get(
                "player_name"
            )
            == "Noobnog"
        )
    ]

    write_records(session_path, records)

    save_state(
        AdaptiveState(
            rating=5.0,
            current_skill=5,
            matches=4,
        ),
        state_path,
    )

    before = state_path.read_bytes()

    result = update_adaptive_state_from_session(
        session_path=session_path,
        state_path=state_path,
    )

    assert result.status == "extraction_failed"
    assert state_path.read_bytes() == before
